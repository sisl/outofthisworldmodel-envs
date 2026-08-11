"""Reference orbit: elements, epoch, and the world<->RTN mapping.

`OrbitConfig` records the chief (ISS) orbital elements and epoch, plus the
per-episode sampling ranges consumed by later tasks (epoch offsets, start
state). `ReferenceOrbit` wraps astrojax to turn those elements into the
chief's ECI state and the world<->ECI rotation at a given time offset from
the epoch, drifting the elements at their first-order J2 secular rates on the
way. What that drift does depends on the consumer. iss-numerical calls
`chief_state_eci` once, at reset, and integrates the chief through the full
perturbed force model from there, so the drift decides only which orbit plane
an episode starts in. iss-hcw integrates no chief at all: it evaluates this
analytically every step, for the |r_chief| in the gravity-gradient term, and
every rendered frame, for the sun direction, eclipse and moon vector. Its
chief therefore does precess -- across a 360 s episode by 0.02 deg of RAAN,
but at an epoch offset days into the window by tens of degrees, which is what
puts its lighting on the plane the ISS would really be in. The module-level
sun/moon/eclipse helpers take a chief ECI state and an `Epoch` directly, so
they compose with `ReferenceOrbit` without depending on its instance.

Dtype policy: astrojax's own float dtype config (`astrojax.config`) defaults
to f32 and is deliberately NOT flipped here. `state_koe_to_eci`,
`rotation_eci_to_rtn`, and `eclipse_conical` honour that config and return
f32 regardless of this package's `jax_enable_x64` flag. `sun_position` and
`moon_position` evaluate at that dtype too, but still hand back f64 under
x64: their closing ecliptic->equatorial rotation is built by `Rx` from a
Python-float obliquity, so the matmul promotes. That extra width is an
artefact of the last operation, not information.

Every public function in this module therefore narrows to the astrojax dtype
on the way out, so none of them advertises f64 for an f32-information value.
Intermediate arithmetic still runs at the widest width available -- see
`sun_direction_world`, where the chief offset is subtracted before the
narrowing. The f64 element math in `ReferenceOrbit` feeding into astrojax is
fine on its own terms, but callers must not expect f64 precision back out.

Error budget -- two independent terms, and only the first is about dtype:

* Numerical. Against an f64 reference over one orbit, chief ECI
  reconstruction differs by ~0.6 m median / ~4 m max, and the sun direction
  by ~6e-6 rad median / ~2e-5 rad max. The narrowing above adds ~1e-7 rad to
  that, i.e. nothing.
* Model. astrojax's sun and moon are Montenbruck & Gill low-precision
  analytical ephemerides, documented at ~0.1 deg (~2e-3 rad). This dominates
  the numerical term by two orders of magnitude and is the real accuracy of
  every sun/moon quantity here. It is ample for lighting, eclipse, and the
  moon's apparent-size swing, none of which are metrology -- but nothing
  downstream should treat these directions as better than ~0.1 deg.

The dominant gravity term for dynamics always comes from
`envs/common/zonal_gravity.py` (f64), never from astrojax's point-mass
helpers, which cast to astrojax's dtype.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from astrojax import config as astrojax_config
from astrojax.constants import J2_EARTH, R_EARTH, R_SUN
from astrojax.coordinates.keplerian import state_koe_to_eci
from astrojax.epoch import Epoch
from astrojax.orbit_dynamics.third_body import moon_position, sun_position
from astrojax.orbits.keplerian import mean_motion as _mean_motion
from astrojax.relative_motion.eci_rtn import rotation_eci_to_rtn
from pydantic import Field, field_validator

from ...core.models import ConfigModel


class OrbitConfig(ConfigModel):
    """Chief (ISS) osculating elements + epoch, and per-episode sampling
    ranges consumed by dynamics/dataset code once orbit dynamics are wired
    in (Tasks 2-4). Defaults reproduce today's behaviour exactly: every
    sampling knob a single fixed value or zero-width range."""

    # ISO 8601, parsed by astrojax.epoch.Epoch. UTC.
    epoch: str = "2026-08-01T00:00:00Z"
    # ISS-like low Earth orbit. gt 6.4e6 keeps the chief above the Earth's
    # surface (R_EARTH ~= 6.378e6 m) with margin.
    sma_m: float = Field(default=6_795_000.0, gt=6.4e6, allow_inf_nan=False)
    ecc: float = Field(default=0.0005, ge=0, lt=1, allow_inf_nan=False)
    inc_deg: float = 51.64
    raan_deg: float = 0.0
    argp_deg: float = 0.0
    mean_anomaly_deg: float = 0.0
    # Per-episode start-time offset from `epoch`, sampled uniformly at
    # reset (Task 3). (0.0, 0.0) reproduces today's single fixed epoch.
    # A window of a few orbits only re-phases the chief along a fixed plane;
    # a multi-day one moves the plane itself, since `ReferenceOrbit` carries
    # the elements forward at their J2 secular rates (-4.955 deg/day of RAAN
    # for these elements). That plane motion, against a sun which moves ~1
    # deg/day, is what varies the beta angle and so the lighting.
    epoch_offset_range_s: tuple[float, float] = (0.0, 0.0)
    # Initial-state sampling knobs (Task 3). Defaults match the current
    # reset exactly: a fixed start radius, zero initial speed, and a
    # nose-to-ISS attitude with zero error and zero rates.
    start_radius_range_m: tuple[float, float] = (100.0, 100.0)
    start_speed_max_m_s: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    start_attitude_error_max_deg: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    start_rate_max_rad_s: float = Field(default=0.0, ge=0, allow_inf_nan=False)

    @field_validator("epoch_offset_range_s")
    @classmethod
    def _epoch_offset_range_is_valid(
        cls, value: tuple[float, float]
    ) -> tuple[float, float]:
        lo, hi = value
        if lo < 0.0 or hi < 0.0 or lo > hi:
            raise ValueError(
                f"epoch_offset_range_s must be (lo, hi) with 0 <= lo <= hi, got {value}"
            )
        return value

    @field_validator("start_radius_range_m")
    @classmethod
    def _start_radius_range_is_valid(
        cls, value: tuple[float, float]
    ) -> tuple[float, float]:
        lo, hi = value
        if lo < 0.0 or hi < 0.0 or lo > hi:
            raise ValueError(
                f"start_radius_range_m must be (lo, hi) with 0 <= lo <= hi, got {value}"
            )
        return value


# RTN (radial, along-track, normal) axes expressed in world coordinates, one
# per row: rtn_vector = RTN_FROM_WORLD @ world_vector.
#
# The renderer places the Earth's center at world (0, 0, -(R_earth + alt))
# (see render/iss_scene.py's `earth_group.local.position`), i.e. straight
# below the chaser along -z. The chief's radial direction R (away from
# Earth, "up") is therefore +z_world. The dock port faces the ram direction
# (T, along-track) at -y_world (DockConfig.quaternion maps body +z, the
# capsule's forward axis, onto world +y -- the ISS itself flies "backwards"
# relative to its velocity in this convention, so along-track is -y_world).
# N completes a right-handed RTN triad: R x T = N => z x (-y) = x, so
# N = +x_world.
RTN_FROM_WORLD: np.ndarray = np.array(
    [
        [0.0, 0.0, 1.0],  # R = +z_world
        [0.0, -1.0, 0.0],  # T = -y_world
        [1.0, 0.0, 0.0],  # N = +x_world
    ]
)


class ReferenceOrbit:
    """Keplerian chief under J2 secular drift + world<->ECI mapping."""

    def __init__(self, cfg: OrbitConfig) -> None:
        self.cfg = cfg
        self.epoch0 = Epoch(cfg.epoch)
        # n = sqrt(mu / a^3); astrojax.orbits.keplerian.mean_motion wraps
        # GM_EARTH from astrojax.constants.
        self.mean_motion: float = float(_mean_motion(cfg.sma_m))
        self._elements0 = jnp.asarray(
            [
                cfg.sma_m,
                cfg.ecc,
                np.deg2rad(cfg.inc_deg),
                np.deg2rad(cfg.raan_deg),
                np.deg2rad(cfg.argp_deg),
                np.deg2rad(cfg.mean_anomaly_deg),
            ],
            dtype=jnp.float64,
        )
        # J2 secular rates. The chief's initial state is the only thing this
        # class produces -- once an episode starts, iss-numerical integrates
        # the chief through the full perturbed force model. But that in-episode
        # propagation covers 360 s, which is 0.02 deg of RAAN, while the epoch
        # window spans a week. Advancing only the mean anomaly across that
        # window would start every episode in the SAME orbital plane, and the
        # lighting diversity the window exists for would come from solar and
        # lunar motion alone. Measured across the seven-day window from four
        # starting epochs, the beta angle spans 16-27 deg with the drift and
        # 2.2-2.6 deg without it -- an order of magnitude, and a span that
        # itself varies with where the window starts, which is part of why
        # sampling within it helps.
        #
        # First-order secular theory, from the same J2 as
        # `envs/common/zonal_gravity`, so the initial condition and the force
        # model that receives it agree about the Earth's oblateness. Short-
        # period and higher-order terms are not modelled: they are bounded and
        # small next to a week of secular regression, and the chief's exact
        # phase within an orbit is already dispersed by the epoch offset
        # itself.
        inclination = float(np.deg2rad(cfg.inc_deg))
        semi_latus_rectum = cfg.sma_m * (1.0 - cfg.ecc**2)
        j2_rate = 1.5 * self.mean_motion * J2_EARTH * (R_EARTH / semi_latus_rectum) ** 2
        sin_inclination_squared = np.sin(inclination) ** 2
        self.secular_rates: tuple[float, float, float] = (
            -j2_rate * np.cos(inclination),
            j2_rate * (2.0 - 2.5 * sin_inclination_squared),
            self.mean_motion
            + j2_rate
            * np.sqrt(1.0 - cfg.ecc**2)
            * (1.0 - 1.5 * sin_inclination_squared),
        )

    def chief_state_eci(self, t_s) -> jnp.ndarray:
        """Chief ECI state at epoch0 + t_s: advance the secularly-drifting
        elements and convert. First-order J2 secular theory, not a numerical
        propagation -- perturbed propagation is iss-numerical's job, not this
        class's. Returns a (6,) astrojax-dtype (f32 by default) array."""
        raan_dot, argp_dot, m_dot = self.secular_rates
        t = jnp.asarray(t_s, jnp.float64)
        elements = self._elements0
        elements = elements.at[3].set(jnp.mod(elements[3] + raan_dot * t, 2.0 * jnp.pi))
        elements = elements.at[4].set(jnp.mod(elements[4] + argp_dot * t, 2.0 * jnp.pi))
        elements = elements.at[5].set(jnp.mod(elements[5] + m_dot * t, 2.0 * jnp.pi))
        return state_koe_to_eci(elements)

    def world_from_eci(self, chief_state_eci: jnp.ndarray) -> jnp.ndarray:
        """R such that v_world = R @ v_eci: through RTN at the chief. Returns
        a (3, 3) astrojax-dtype (f32 by default) array."""
        return world_from_eci(chief_state_eci)


def world_from_eci(chief_state_eci: jnp.ndarray) -> jnp.ndarray:
    """R such that v_world = R @ v_eci, from the chief's ECI state alone.

    The instance method above is the usual way in; this module-level form
    exists for callers that carry a chief state but no `ReferenceOrbit` --
    the rotation depends on the state, never on the orbit's elements."""
    rtn_from_eci = rotation_eci_to_rtn(chief_state_eci)
    # RTN_FROM_WORLD's entries are exactly 0/+-1, so narrowing it to the
    # astrojax dtype is lossless and keeps the product from advertising f64
    # for a value that only ever carried f32 information.
    return jnp.asarray(RTN_FROM_WORLD.T, rtn_from_eci.dtype) @ rtn_from_eci


def earth_rotation_world(chief_state_eci: jnp.ndarray, epoch: Epoch) -> jnp.ndarray:
    """R such that v_world = R @ v_ecef: where the rotating Earth's own axes
    point in world coordinates at `epoch`.

    The renderer holds the globe as a fixed body and needs its attitude, not a
    spin angle: the world frame is RTN at the chief, so the planet turns under
    the scene for two independent reasons -- the Earth's own rotation, and the
    chief's motion around it, which is much the faster of the two. Handing back
    the composed rotation covers both, and the sub-satellite point falls out of
    it rather than having to be tracked: the chief's own ECEF position maps to
    +z_world by construction, which is exactly where the scene hangs the
    station over the planet.

    ECEF here is the ECI frame turned by Greenwich mean sidereal time about the
    pole, with no precession, nutation or polar motion. Those reach ~0.4 deg of
    surface longitude at present epochs -- tens of km of ground track, which is
    not resolvable against a globe drawn from a 16k-wide texture, and nothing
    downstream of this measures a position on the surface. Scalar `Epoch` and a
    single (6,) chief state -- NOT batched; vmap for batches. Returns a (3, 3)
    astrojax-dtype (f32 by default) array."""
    gmst = jnp.asarray(epoch.gmst(), jnp.float64)
    cos_gmst, sin_gmst = jnp.cos(gmst), jnp.sin(gmst)
    zero, one = jnp.zeros_like(gmst), jnp.ones_like(gmst)
    eci_from_ecef = jnp.stack(
        [
            jnp.stack([cos_gmst, -sin_gmst, zero]),
            jnp.stack([sin_gmst, cos_gmst, zero]),
            jnp.stack([zero, zero, one]),
        ]
    )
    rotated = world_from_eci(chief_state_eci).astype(jnp.float64) @ eci_from_ecef
    return rotated.astype(astrojax_config.get_dtype())


def sun_direction_world(chief_state_eci: jnp.ndarray, epoch: Epoch) -> jnp.ndarray:
    """Unit vector from the chief toward the sun, expressed in world
    coordinates. This one IS translated to the chief -- it feeds a light
    direction at the chief, and the ~3.5e-5 rad parallax against the
    geocentric direction is the physically right thing there. Scalar `Epoch`
    and a single (6,) chief state -- NOT batched; vmap for batches. Returns a
    (3,) astrojax-dtype (f32 by default) array."""
    # The chief offset is subtracted at whatever width sun_position hands
    # back (f64 today) before the result is narrowed: 6.8e6 m against 1.5e11
    # m is only ~400 f32 ulps, so doing it wide keeps the parallax clean.
    rel = sun_position(epoch) - chief_state_eci[:3]
    unit = world_from_eci(chief_state_eci) @ (rel / jnp.linalg.norm(rel))
    return unit.astype(astrojax_config.get_dtype())


def moon_vector_world(chief_state_eci: jnp.ndarray, epoch: Epoch) -> jnp.ndarray:
    """Geocentric lunar position vector expressed in WORLD AXES (not
    translated to the chief): direction AND distance, so the true-scale moon
    renders with its real +/-7% apparent-size swing. The geocentric origin is
    the contract: the moon is meant to hang off the Earth-center scene node,
    at earth_center_world + this, which is why the chief offset must NOT be
    subtracted here -- unlike `sun_direction_world`, whose consumer wants a
    direction at the chief. `ISSScene.update` places the moon in exactly that
    earth_center + offset form when the frame carries `Lighting`, and from its
    static `moon_direction_from_earth_world` / `earth_moon_distance_m` config
    when it does not. `chief_state_eci` enters only through the world rotation.
    Scalar `Epoch` and a single (6,) chief state -- NOT batched; vmap for
    batches. Returns a (3,) astrojax-dtype (f32 by default) array."""
    rotated = world_from_eci(chief_state_eci) @ moon_position(epoch)
    return rotated.astype(astrojax_config.get_dtype())


def illumination(chief_r_eci: jnp.ndarray, epoch: Epoch) -> jnp.ndarray:
    """Conical-shadow illumination fraction in [0, 1] at the chief: the
    fraction of the sun's disc still visible past the Earth's. Returns a
    scalar astrojax-dtype (f32 by default) array.

    From the chief's ~6.8e6 m radius the sun subtends a = 4.65e-3 rad and the
    Earth b = 1.2196 rad, and the penumbra is the 2a-wide band of separations c
    where the two discs overlap only partially. Across a band that narrow the
    Earth's limb is a straight chord over the sun's disc, so the lit fraction
    is a circular segment of that disc, cut at the signed distance
    (b**2 - c**2) / 2c from its centre. Clipping that distance to +/- a covers
    both extremes -- fully lit and full umbra -- so the branch-free form below
    is the whole function.

    The lens-area form (Montenbruck & Gill, and `astrojax.orbit_dynamics.srp
    .eclipse_conical`) is exact for discs of any size, and the curvature term
    this drops is worth ~6e-4 of illumination here -- three orders below the
    ~0.1 deg sun direction it is evaluated against. What it costs is
    conditioning: it resolves a difference of two ~5.6e-3 terms against
    pi*a**2 = 6.8e-5, through an arccos evaluated at ~0.999993 where the
    derivative is -270, so an input ulp reaches the result amplified about
    5.9e6 times. At astrojax's default f32 that is O(0.3) of error on a
    quantity bounded by 1, unbounded in both directions once amplified, and
    different frame to frame -- a rendered eclipse transition strobes rather
    than fades. Nothing in the segment form cancels worse than b - c over a
    range of 2a, so it holds to ~1e-4 even at f32 and to rounding in the f64
    it is evaluated at here.
    """
    r = jnp.asarray(chief_r_eci, jnp.float64)[:3]
    to_sun = jnp.asarray(sun_position(epoch), jnp.float64) - r
    r_norm = jnp.linalg.norm(r)
    to_sun_norm = jnp.linalg.norm(to_sun)

    sun_radius_rad = jnp.arcsin(R_SUN / to_sun_norm)
    earth_radius_rad = jnp.arcsin(R_EARTH / r_norm)
    separation_rad = jnp.arccos(-jnp.dot(r, to_sun) / (r_norm * to_sun_norm))

    chord = jnp.clip(
        (earth_radius_rad**2 - separation_rad**2)
        / (2.0 * separation_rad * sun_radius_rad),
        -1.0,
        1.0,
    )
    lit = (jnp.arccos(chord) - chord * jnp.sqrt(1.0 - chord**2)) / jnp.pi
    return lit.astype(astrojax_config.get_dtype())
