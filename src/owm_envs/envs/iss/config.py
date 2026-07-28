"""ISS environment configuration.

Frozen Pydantic models replacing seamstress's Hydra YAML. Values are carried
from seamstress branch iss2,
conf/environments/environment_international_space_station.yaml.

These round-trip through YAML (see core.config_io.YamlModel) so a specific
experiment's settings can be committed under configs/ as a versioned input and
written next to the dataset it produced as an as-run record.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import Field

from ...core.config_io import YamlModel

# Sentinel for ISSConfig.collision_boxes_path meaning "use the 318-box ISS
# geometry shipped with this package". A plain sentinel (rather than baking
# default_collision_boxes_path()'s absolute, install-location-dependent path
# into the field default) keeps the default portable across machines and
# installs, and keeps configs/iss_default.yaml diffable/reviewable.
DEFAULT_COLLISION_BOXES = "default"


class RewardWeights(YamlModel):
    """Weights for the five reward terms. Carried from seamstress iss2.

    NOTE: `collision` at 1e6 was tuned for MPPI planning, where rewards are
    averaged over a horizon. As a per-step Gymnasium reward it is an enormous
    negative spike that will dominate RL gradients. Retuning is deliberately
    left to the consumer.
    """

    position: float = 1.0
    velocity: float = 0.35
    angular_velocity: float = 0.1
    control_effort: float = 0.05
    collision: float = 1_000_000.0


class ISSConfig(YamlModel):
    dt: float = 0.05
    # 2000 * 0.05 = 100 s. Past this the Earth-rotation texture sampling in the
    # renderer visibly glitches, so iss2 caps episodes here.
    max_steps: int = 2000

    mass: float = 12_000.0
    inertia_diag: tuple[float, float, float] = (80_000.0, 80_000.0, 50_000.0)
    linear_damping: float = 0.005
    angular_damping: float = 0.02

    dragon_collision_radius_m: float = 2.25
    start_radius_m: float = 100.0

    dock_position: tuple[float, float, float] = (0.225, -24.5, -2.5)
    # Maps body +z onto world +y: 90 deg rotation about world -x.
    dock_quaternion: tuple[float, float, float, float] = (0.7071068, -0.7071068, 0.0, 0.0)
    dock_enabled: bool = True
    dock_max_distance_m: float = 0.1
    dock_max_velocity_m_s: float = 0.5

    # A path to a YAML file, an already-loaded list of box dicts (useful in
    # tests), DEFAULT_COLLISION_BOXES to use the 318-box ISS geometry shipped
    # with this package (the default), or None to opt out of collision
    # geometry entirely.
    collision_boxes_path: str | list[dict] | None = DEFAULT_COLLISION_BOXES

    # 9x the baseline (was 2000 N / 10000 N*m). Traversal time scales as
    # 1/sqrt(F_max), so 9x force gives ~3x faster chaser motion. Not physically
    # realistic for a real Dragon -- purely a synthetic-dataset variety knob.
    control_limit_force_n: float = 18_000.0
    control_limit_torque_nm: float = 90_000.0

    reward_weights: RewardWeights = Field(default_factory=RewardWeights)
    # Overrides the reward's position-error target. When None (default),
    # iss_reward measures distance to `dock_position`, as it should for a
    # docking task. Seamstress's own control config instead sets the reward
    # goal to the ISS origin [0, 0, 0] -- but that point is INSIDE the
    # station's collision hull (dock_position, ~24.63 m away, is not), so
    # seamstress's MPPI was being pulled toward a position it can never reach
    # without incurring the -1e6 collision penalty. This field exists to
    # deliberately reproduce that seamstress behaviour, or to target some
    # other point -- not because the origin is a sensible target.
    reward_goal_position: tuple[float, float, float] | None = None


def default_collision_boxes_path() -> str:
    """Path to the 318-AABB ISS geometry shipped with this package."""
    return str(Path(__file__).resolve().parent / "resources" / "collision_boxes.yaml")


def load_collision_boxes(source: Any) -> tuple[np.ndarray, np.ndarray]:
    """Load axis-aligned collision boxes.

    `source` may be:
      - None                     -> empty box set
      - DEFAULT_COLLISION_BOXES  -> the 318-box ISS geometry shipped with this package
      - a path to a YAML file containing a list of {center, size} dicts
      - an already-loaded list of {center, size} or {center, half_extents} dicts

    Returns (centers (N,3) float32, half_extents (N,3) float32).
    """
    if source is None:
        return _empty_boxes()

    if source == DEFAULT_COLLISION_BOXES:
        return load_collision_boxes(default_collision_boxes_path())

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"collision_boxes file not found: {path}")
        return load_collision_boxes(yaml.safe_load(path.read_text()))

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
