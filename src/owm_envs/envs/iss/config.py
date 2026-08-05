"""ISS environment configuration.

Frozen Pydantic models for the ISS docking environment.

These round-trip through YAML/TOML (see core.models.ConfigModel) so a specific
experiment's settings can be committed under configs/ as a versioned input and
written next to the dataset it produced as an as-run record.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import Field, field_validator

from ...core.models import ConfigModel
from .sensing import SensorNoiseConfig

# Directory holding geometry shipped with this package. A relative
# collision_boxes_path is resolved against the caller's cwd first (so a
# user-supplied file wins), falling back to this directory so the packaged
# default stays valid regardless of where it's loaded from.
_RESOURCES_DIR = Path(__file__).resolve().parent / "resources"

# Filename, within _RESOURCES_DIR, of the 318-box ISS geometry shipped with
# this package. This is the default for PhysicsConfig.collision_boxes_path:
# a real, relative path rather than a magic sentinel, so it stays portable
# across machines/installs (an absolute path would make configs/iss_default
# .toml install-location-dependent) while still naming an actual file.
DEFAULT_COLLISION_BOXES_FILENAME = "collision_boxes.yaml"


class PhysicsConfig(ConfigModel):
    """Chaser mass properties, damping, and collision geometry."""

    # Strictly positive and finite: _eom divides force/torque by mass and
    # inertia, so a non-positive or non-finite value produces inf/NaN/
    # reversed-sign accelerations rather than a load error.
    mass: float = Field(default=12_000.0, gt=0, allow_inf_nan=False)
    inertia_diag: tuple[float, float, float] = (80_000.0, 80_000.0, 50_000.0)
    # Free-body motion in vacuum has no medium to damp against, so the
    # physical default is zero. Non-zero values are a deliberate, unphysical
    # artificial-stabilisation knob, not a default choice.
    #
    # Non-negative and finite: the orbit policy's rate bound consumes
    # linear_damping (holding a circle has to cancel drag as well as supply
    # centripetal force), so a NaN would propagate silently into every
    # sampled orbit, and a negative value would inject energy rather than
    # remove it.
    # The upper bound is not arithmetic defensiveness: a coefficient of 1e6
    # damps all motion within a microsecond, so anything beyond it describes
    # no scenario this environment can represent, and capping it keeps the
    # orbit rate bound's intermediates trivially in range.
    linear_damping: float = Field(default=0.0, ge=0, le=1e6, allow_inf_nan=False)
    angular_damping: float = Field(default=0.0, ge=0, le=1e6, allow_inf_nan=False)

    # Radii, not divisors -- a negative value is geometrically meaningless
    # (it would shrink the collision box or flip the start position to the
    # far side of the sphere) but zero is a valid degenerate case. Still
    # excludes inf/NaN, which are as meaningless here as they are for mass.
    dragon_collision_radius_m: float = Field(default=2.25, ge=0, allow_inf_nan=False)
    # Episodes start at a uniformly sampled radius in this closed interval,
    # so some trajectories begin near the station and others must first fly
    # in from up to 500 m out. A degenerate (r, r) range reproduces the old
    # fixed start sphere.
    start_radius_range_m: tuple[float, float] = (100.0, 500.0)

    # A path to a YAML file (relative paths resolve against the cwd first,
    # then this package's resources directory, so the default below finds
    # the shipped geometry without hardcoding an install-location-dependent
    # absolute path), an already-loaded list of box dicts (useful in tests),
    # or None to opt out
    # of collision geometry entirely -- which emits a warning, since a
    # silently collision-free environment can never terminate on collision.
    collision_boxes_path: str | list[dict] | None = DEFAULT_COLLISION_BOXES_FILENAME

    @field_validator("start_radius_range_m")
    @classmethod
    def _start_radius_range_is_ordered(
        cls, value: tuple[float, float]
    ) -> tuple[float, float]:
        # reset() feeds the interval to a uniform sample; a reversed, negative,
        # or non-finite bound would silently place the chaser somewhere
        # geometrically meaningless rather than fail to load.
        low, high = value
        if not (math.isfinite(low) and math.isfinite(high)) or low < 0 or high < low:
            raise ValueError("start_radius_range_m must be finite with 0 <= min <= max")
        return value

    @field_validator("inertia_diag")
    @classmethod
    def _inertia_diag_is_positive(
        cls, value: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        # The angular equation of motion divides by each component; a
        # non-positive or non-finite moment of inertia produces inf/NaN/
        # reversed-sign angular acceleration rather than a load error.
        if any(not math.isfinite(component) or component <= 0 for component in value):
            raise ValueError("inertia_diag components must be finite and positive")
        return value


class ControlConfig(ConfigModel):
    """Actuator limits.

    Sized to a real Dragon's Draco RCS: one Draco produces ~400 N, and a
    proximity-ops translation fires ~4 of the 16 in a given body direction,
    so 1600 N per axis (~0.13 m/s^2 at 12 t). Torque from a Draco couple at
    a ~2.5 m arm gives ~2000 N*m (~0.025 rad/s^2 about the transverse axes).
    """

    # Strictly positive and finite: the orbit policy divides its rate bound by
    # the force limit, so zero would command a stationary "orbit" and a
    # negative or non-finite value would poison every sampled rate. A chaser
    # with no thrust is also not a scenario any policy here can express.
    limit_force_n: float = Field(default=1600.0, gt=0, allow_inf_nan=False)
    limit_torque_nm: float = Field(default=2000.0, gt=0, allow_inf_nan=False)


class DockConfig(ConfigModel):
    """Dock pose and the success criteria for reaching it."""

    position: tuple[float, float, float] = (0.225, -24.5, -2.5)
    # Maps body +z onto world +y: 90 deg rotation about world -x.
    quaternion: tuple[float, float, float, float] = (0.7071068, -0.7071068, 0.0, 0.0)
    enabled: bool = True
    max_distance_m: float = 0.1
    max_velocity_m_s: float = 0.5
    # Optional gates: None admits any attitude/rate. Note the unit mismatch
    # between the two fields -- max_attitude_error_deg is DEGREES,
    # max_body_rate_rad_s is RADIANS per second. 0.008727 rad/s = 0.5 deg/s.
    max_attitude_error_deg: float | None = 5.0
    max_body_rate_rad_s: float | None = 0.008727


class RewardWeights(ConfigModel):
    """Penalty weights for the five reward terms.

    Negative so that `iss_reward` is a plain weighted sum of these against
    non-negative error terms -- no separate negation needed at the call site.

    NOTE: `collision` at -1e6 was tuned for MPPI planning, where rewards are
    averaged over a horizon. As a per-step Gymnasium reward it is an enormous
    negative spike that will dominate RL gradients. Retuning is deliberately
    left to the consumer.
    """

    position: float = -1.0
    velocity: float = -0.35
    angular_velocity: float = -0.1
    control_effort: float = -0.05
    collision: float = -1_000_000.0


class ObservationConfig(ConfigModel):
    """What the emitted observation contains beyond the 13-dim state."""

    # Appends [pos_err(3), vel_err(3), att_err_axis_angle(3), rate_err(3)]
    # (obs 13 -> 25). Gym envs use the dock pose as the goal; the rollout
    # drivers resolve the goal per policy (dock pose / orbit desired state /
    # zeros for random).
    goal_error: bool = False


class ISSConfig(ConfigModel):
    # Strictly positive and finite: dt scales every RK4 stage, so zero freezes
    # the simulation and a negative value integrates backwards in time. It is
    # also a divisor -- the recorded frame rate is 1/dt -- so zero would raise
    # rather than produce a wrong number.
    dt: float = Field(default=0.05, gt=0, allow_inf_nan=False)
    # 12000 * 0.05 = 600 s. Sized for the orbit policy's slowest commanded
    # circle at Draco-class thrust: a 150 m orbit at the minimum speed
    # fraction has a ~450 s period, plus the fly-in from a start as far as
    # 500 m out, so anything much shorter ends before a revolution
    # completes. A critically-damped dock approach settles into the gate in
    # ~250 s from the nearest starts and ~300 s median from the farthest,
    # comfortably inside the same horizon.
    #
    # The renderer draws Earth as a full textured globe, so however far the
    # ground track runs there is always surface under it; episode length is
    # bounded by the control problem alone, not by the map.
    max_steps: int = 12000
    # Radius of the spherical domain, measured from the ISS origin. A chaser
    # whose post-step position lies outside it has left the scenario, and the
    # episode terminates there rather than running out the horizon.
    #
    # 750 m is 1.5x the farthest start radius (500 m), so even an episode
    # that begins at the outer edge has room to maneuver before it is out of
    # bounds. Its purpose is the random policy, whose undirected walk
    # otherwise spends the full horizon drifting away from the station
    # recording nothing but empty space. It also bounds the non-cooperative
    # sensor noise, whose position error is proportional to range
    # (sigma_pos_frac_of_range) and so grows without limit on a runaway
    # trajectory.
    #
    # None removes the bound entirely, for callers who want unbounded flight.
    # Strictly positive and finite when set: zero or negative would put every
    # reachable state outside the domain, and inf/NaN describe no boundary at
    # all -- None is how the bound is turned off.
    max_range_m: float | None = Field(default=750.0, gt=0, allow_inf_nan=False)

    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    control: ControlConfig = Field(default_factory=ControlConfig)
    dock: DockConfig = Field(default_factory=DockConfig)
    sensor_noise: SensorNoiseConfig = Field(default_factory=SensorNoiseConfig)
    observation: ObservationConfig = Field(default_factory=ObservationConfig)

    reward_weights: RewardWeights = Field(default_factory=RewardWeights)
    # Overrides the reward's position-error target. When None (default),
    # iss_reward measures distance to `dock.position`, as it should for a
    # docking task. The ISS origin [0, 0, 0] is INSIDE the station's
    # collision hull (dock.position, ~24.63 m away, is not), so setting this
    # field to the origin would pull a controller toward a position it can
    # never reach without incurring the -1e6 collision penalty. This field
    # exists to deliberately target the origin (or some other point) for
    # callers who want that trade-off -- not because the origin is a
    # sensible target.
    reward_goal_position: tuple[float, float, float] | None = None

    # Which of the renderer's six fixed views `ISSEnv.render()` uses:
    # "DRAGON_ISO", "DRAGON_TOP", "DRAGON_FPV", "ISS_ISO", "ISS_TOP", or
    # "ISS_FPV" (renderer.ViewName). The first-person view is what datasets
    # record as observation.images.fpv, the key training reads whatever else a
    # run rendered, so live rendering through the Gymnasium adapters shows the
    # same view a world model trains on.
    render_view: str = "DRAGON_FPV"
    # Graphics settings for `ISSEnv.render()`, as a plain dict of
    # `owm_envs.render.iss_scene.RenderConfig` fields (e.g. {"image_width":
    # 640}). A dict rather than `RenderConfig` itself so this module -- part
    # of the core, always-imported package -- never has to import the
    # `render` package, whose modules pull in pygfx at the top level. `None`
    # (the default) renders with `RenderConfig`'s own defaults.
    render: dict[str, Any] | None = None


def dock_target(cfg: ISSConfig) -> np.ndarray:
    """The (7,) [position, quaternion] row `DockConfig` names.

    The single-target form of what `policies.dock_target_table` produces for a
    port set, so both paths hand the same layout to the dynamics and the
    goal-error block.
    """
    return np.concatenate(
        [
            np.asarray(cfg.dock.position, dtype=np.float32),
            np.asarray(cfg.dock.quaternion, dtype=np.float32),
        ]
    )


def default_collision_boxes_path() -> str:
    """Absolute path to the 318-AABB ISS geometry shipped with this package."""
    return str(_RESOURCES_DIR / DEFAULT_COLLISION_BOXES_FILENAME)


def load_collision_boxes(source: Any) -> tuple[np.ndarray, np.ndarray]:
    """Load axis-aligned collision boxes.

    `source` may be:
      - None            -> no geometry: empty box set, with a warning, since a
                            silently collision-free environment can never
                            terminate on collision
      - a relative path -> resolved against the current working directory
                            first (a user-supplied file, e.g. "my_boxes.yaml"),
                            then against this package's resources directory
                            (DEFAULT_COLLISION_BOXES_FILENAME there is the
                            shipped 318-box ISS geometry)
      - an absolute path -> a YAML file containing a list of {center, size} dicts
      - an already-loaded list of {center, size} or {center, half_extents} dicts

    Returns (centers (N,3) float32, half_extents (N,3) float32).
    """
    if source is None:
        warnings.warn(
            "collision_boxes_path is None: no collision geometry will be "
            "loaded, so episodes can never terminate on collision.",
            stacklevel=2,
        )
        return _empty_boxes()

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_absolute():
            cwd_path = Path.cwd() / path
            package_path = _RESOURCES_DIR / path
            # The cwd must win: a user-supplied relative path (e.g. "./my_
            # boxes.yaml") means "relative to where I'm running from"
            # everywhere else, and if the package directory won instead, a
            # user file that happens to share a name with a packaged asset
            # would silently load the wrong geometry.
            if cwd_path.exists():
                path = cwd_path
            elif package_path.exists():
                path = package_path
            else:
                raise FileNotFoundError(
                    "collision_boxes file not found in either location "
                    f"tried: {cwd_path} or {package_path}"
                )
        elif not path.exists():
            raise FileNotFoundError(f"collision_boxes file not found: {path}")
        # An empty file (or one that's just `null`) parses to None. That's a
        # real, specified path holding zero boxes, not "no path configured"
        # -- route it to the empty box set directly rather than recursing
        # into the source-is-None branch above, which would misattribute it
        # and warn about a path that was in fact given.
        loaded = yaml.safe_load(path.read_text())
        return load_collision_boxes(loaded if loaded is not None else [])

    centers: list[np.ndarray] = []
    half_extents: list[np.ndarray] = []
    for raw in source:
        center = np.asarray(raw["center"], dtype=np.float32).reshape(3)
        if "half_extents" in raw:
            half = np.asarray(raw["half_extents"], dtype=np.float32).reshape(3)
        else:
            half = 0.5 * np.asarray(raw["size"], dtype=np.float32).reshape(3)
        centers.append(center)
        half_extents.append(np.maximum(half, 0.0))

    if not centers:
        return _empty_boxes()
    return np.stack(centers, axis=0), np.stack(half_extents, axis=0)


def _empty_boxes() -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
