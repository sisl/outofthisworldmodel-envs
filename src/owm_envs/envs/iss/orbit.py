"""ISS reference orbit: elements, epoch, and the world<->RTN mapping.

`OrbitConfig` records the chief (ISS) orbital elements and epoch, plus the
per-episode sampling ranges consumed by later tasks (epoch offsets, start
state). `ReferenceOrbit` wraps astrojax to turn those elements into the
chief's ECI state and the sun direction at a given time offset from the
epoch, expressed in the sim's world frame -- everything downstream (CW
dynamics, gravity-gradient torque, renderer lighting) consumes only
`mean_motion` and these two methods.

Disabled by default (`OrbitConfig.enabled = False`): nothing in this module
is imported or evaluated by the physics/render paths unless a caller opts in.
"""

from __future__ import annotations

import numpy as np
from astrojax.coordinates.keplerian import state_koe_to_eci
from astrojax.epoch import Epoch
from astrojax.orbit_dynamics.planetary_ephemerides import emb_position_jpl_approx
from astrojax.orbits.keplerian import mean_motion as _mean_motion
from pydantic import Field, field_validator

from ...core.models import ConfigModel


class OrbitConfig(ConfigModel):
    """Chief (ISS) osculating elements + epoch, and per-episode sampling
    ranges consumed by dynamics/dataset code once orbit dynamics are wired
    in (Tasks 2-4). Defaults reproduce today's behaviour exactly: disabled,
    and every sampling knob a single fixed value or zero-width range."""

    enabled: bool = False
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
    """The chief's (ISS) reference orbit: two-body Keplerian propagation of
    the configured elements, plus the geocentric sun direction, both
    expressed in world/RTN frames as needed by dynamics and rendering."""

    def __init__(self, cfg: OrbitConfig) -> None:
        self.cfg = cfg
        self.epoch0 = Epoch(cfg.epoch)
        # n = sqrt(mu / a^3); astrojax.orbits.keplerian.mean_motion wraps
        # GM_EARTH from astrojax.constants.
        self.mean_motion: float = float(_mean_motion(cfg.sma_m))
        self._elements0 = np.array(
            [
                cfg.sma_m,
                cfg.ecc,
                np.deg2rad(cfg.inc_deg),
                np.deg2rad(cfg.raan_deg),
                np.deg2rad(cfg.argp_deg),
                np.deg2rad(cfg.mean_anomaly_deg),
            ]
        )

    def chief_eci_state(self, t_offset_s: float) -> np.ndarray:
        """Chief ECI state `[x, y, z, vx, vy, vz]` (m, m/s) at `t_offset_s`
        seconds past `epoch0`, via mean-anomaly propagation M(t) = M0 + n*t
        and the standard perifocal->ECI conversion (two-body Keplerian --
        no perturbations)."""
        elements = self._elements0.copy()
        elements[5] = self._elements0[5] + self.mean_motion * t_offset_s
        return np.asarray(state_koe_to_eci(elements))

    def sun_direction_world(self, t_offset_s: float) -> np.ndarray:
        """Unit vector from the chief toward the sun, expressed in the sim
        world frame at `t_offset_s`.

        The sun's geocentric direction is (to the precision of astrojax's
        JPL-approximate ephemerides) independent of the chief's position --
        Earth-Moon-barycenter distance to the sun (~1 AU) dwarfs LEO
        altitudes -- so it is computed once in ECI from the epoch, then
        rotated into world coordinates via the chief's *own* instantaneous
        RTN basis (which is what makes the direction vary across the orbit
        in world/LVLH coordinates, even though it is fixed in ECI)."""
        epoch_t = self.epoch0 + t_offset_s
        # emb_position_jpl_approx is heliocentric (sun->EMB); the geocentric
        # sun direction is the negated, normalized EMB position.
        emb_pos = np.asarray(emb_position_jpl_approx(epoch_t))
        sun_eci_hat = -emb_pos / np.linalg.norm(emb_pos)

        state = self.chief_eci_state(t_offset_s)
        r, v = state[:3], state[3:6]
        r_hat = r / np.linalg.norm(r)
        h = np.cross(r, v)
        n_hat = h / np.linalg.norm(h)
        t_hat = np.cross(n_hat, r_hat)
        rtn_from_eci = np.stack([r_hat, t_hat, n_hat])  # rows R, T, N

        sun_rtn = rtn_from_eci @ sun_eci_hat
        return RTN_FROM_WORLD.T @ sun_rtn
