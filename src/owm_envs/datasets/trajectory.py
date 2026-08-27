"""One episode of any docking harness, as a directory of two files.

`trajectory.npz` holds the arrays and `meta.json` everything a reader needs to
rebuild the environment the episode flew on: the env name and its inline
config, the step spacing, the port and seed, and the outcome. The rows are
recorded at the environment's own integration step, whatever cadence the
policy decided at, so a clip rendered from the file moves at the simulation's
rate rather than the controller's.

Every writer in the suite -- a reinforcement-learning checkpoint, a
world-model planner, a scripted policy -- targets this layout, and the render
and plot commands read nothing else. That is the whole point of the file: two
harnesses that agree on `(env_config, port, seed)` produce episodes this
module can put side by side.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from ..envs import ENV_REGISTRY
from ..envs.common.layout import VIEW_DIM
from .video import VIEW_NAMES

ARRAY_FILE = "trajectory.npz"
META_FILE = "meta.json"

META_KEYS: tuple[str, ...] = (
    "method",
    "port",
    "seed",
    "env",
    "env_config",
    "dt",
    "rate_hz",
    "action_repeat",
    "steps",
    "outcome",
    "ever_collided",
    "min_range_m",
    "start_fingerprint",
    "lighting",
    "produced_by",
)

OUTCOMES: tuple[str, ...] = ("docked", "collision", "escaped", "truncated")

# The file-name tail each rendered view is written under: the short names
# people type (`fpv`, `dragon_iso`, ...) collapsed to what fits in a clip name.
SHORT_VIEW_NAMES: dict[str, str] = {
    "fpv": "fpv",
    "dragon_iso": "iso",
    "dragon_top": "top",
    "iss_fpv": "iss_fpv",
    "iss_iso": "iss_iso",
    "iss_top": "iss_top",
    "composite": "composite",
}
assert set(SHORT_VIEW_NAMES) == set(VIEW_NAMES)

# Arrays with T+1 rows (one per state, including the reset state) and those
# with T rows (one per transition).
_PER_STATE = ("epoch", "state", "rel_view", "measured_state", "observation")
_PER_STEP = ("action_norm", "action_phys", "reward", "collision")


@dataclass
class Trajectory:
    epoch: np.ndarray
    state: np.ndarray
    rel_view: np.ndarray
    measured_state: np.ndarray
    observation: np.ndarray
    action_norm: np.ndarray
    action_phys: np.ndarray
    reward: np.ndarray
    collision: np.ndarray
    dock_target: np.ndarray
    meta: dict

    @property
    def steps(self) -> int:
        return int(self.action_norm.shape[0])

    @property
    def dt(self) -> float:
        return float(self.meta["dt"])

    def validate(self) -> None:
        missing = [key for key in META_KEYS if key not in self.meta]
        if missing:
            raise ValueError(f"meta.json is missing {missing}")
        env = self.meta["env"]
        if env not in ENV_REGISTRY:
            raise ValueError(
                f"meta.json names env '{env}', which this build does not register "
                f"({', '.join(ENV_REGISTRY)})"
            )
        if self.meta["outcome"] not in OUTCOMES:
            raise ValueError(
                f"meta.json outcome '{self.meta['outcome']}' is not one of {OUTCOMES}"
            )
        steps = self.steps
        for name in _PER_STATE:
            rows = getattr(self, name).shape[0]
            if rows != steps + 1:
                raise ValueError(f"{name} has {rows} rows, expected steps + 1 = {steps + 1}")
        for name in _PER_STEP:
            rows = getattr(self, name).shape[0]
            if rows != steps:
                raise ValueError(f"{name} has {rows} rows, expected steps = {steps}")
        if self.epoch.shape[1:] != (2,):
            raise ValueError(f"epoch must be (T+1, 2), got {self.epoch.shape}")
        if self.rel_view.shape[1:] != (VIEW_DIM,):
            raise ValueError(f"rel_view must be (T+1, {VIEW_DIM}), got {self.rel_view.shape}")
        if self.action_norm.shape[1:] != (6,) or self.action_phys.shape[1:] != (6,):
            raise ValueError("actions must be (T, 6)")
        if self.dock_target.shape != (7,):
            raise ValueError(f"dock_target must be (7,), got {self.dock_target.shape}")
        if int(self.meta["steps"]) != steps:
            raise ValueError(f"meta.json steps={self.meta['steps']} but arrays hold {steps}")


def save_trajectory(traj: Trajectory, directory: str | Path) -> None:
    traj.validate()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {f.name: getattr(traj, f.name) for f in fields(traj) if f.name != "meta"}
    np.savez_compressed(directory / ARRAY_FILE, **arrays)
    (directory / META_FILE).write_text(json.dumps(traj.meta, indent=2, sort_keys=True) + "\n")


def load_trajectory(directory: str | Path) -> Trajectory:
    directory = Path(directory)
    array_path, meta_path = directory / ARRAY_FILE, directory / META_FILE
    for path in (array_path, meta_path):
        if not path.is_file():
            raise FileNotFoundError(f"{directory} holds no {path.name}")
    with np.load(array_path) as data:
        names = [f.name for f in fields(Trajectory) if f.name != "meta"]
        absent = [name for name in names if name not in data]
        if absent:
            raise ValueError(f"{array_path} is missing arrays {absent}")
        arrays = {name: data[name] for name in names}
    traj = Trajectory(**arrays, meta=json.loads(meta_path.read_text()))
    traj.validate()
    return traj
