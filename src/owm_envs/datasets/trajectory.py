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

The dtypes are part of that agreement, and `validate()` enforces them. The
truth channels -- `epoch`, `state`, `measured_state`, `rel_view`, `reward`
and `dock_target` -- are float64, because they are what one harness's episode
is compared against another's: the epoch prefix alone carries a Julian date
near 2.46e6, which float32 cannot hold to better than a tenth of a second,
and the ECI positions a relative view is differenced from are ~6.8e6 m, where
float32 costs a quarter of a metre. `observation`, `action_norm` and
`action_phys` are float32, which is what a policy saw and emitted, and
`collision` is bool.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from ..envs import ENV_REGISTRY, EnvSpec
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
if set(SHORT_VIEW_NAMES) != set(VIEW_NAMES):
    raise RuntimeError(
        f"SHORT_VIEW_NAMES covers {sorted(SHORT_VIEW_NAMES)}, but the renderable views "
        f"are {sorted(VIEW_NAMES)}"
    )

# Arrays with T+1 rows (one per state, including the reset state) and those
# with T rows (one per transition).
_PER_STATE = ("epoch", "state", "rel_view", "measured_state", "observation")
_PER_STEP = ("action_norm", "action_phys", "reward", "collision")

# The stored dtype and rank of every array, as the module docstring states them.
_DTYPES: dict[str, type] = {
    "epoch": np.float64,
    "state": np.float64,
    "rel_view": np.float64,
    "measured_state": np.float64,
    "observation": np.float32,
    "action_norm": np.float32,
    "action_phys": np.float32,
    "reward": np.float64,
    "collision": np.bool_,
    "dock_target": np.float64,
}
_RANKS: dict[str, int] = {
    "epoch": 2, "state": 2, "rel_view": 2, "measured_state": 2, "observation": 2,
    "action_norm": 2, "action_phys": 2, "reward": 1, "collision": 1, "dock_target": 1,
}
_FLOAT_ARRAYS = tuple(name for name, dtype in _DTYPES.items() if dtype is not np.bool_)

# `method` names the clips and plots a file renders to, as a bare stem.
_METHOD_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# float32 actions that came back through a normalisation divide can land a few
# ulps outside the closed unit box; an action_phys stored in the wrong field
# misses it by orders of magnitude.
_ACTION_TOL = 1e-6


def start_fingerprint(state0: np.ndarray) -> str:
    """A digest of an episode's initial TRUE state, over its raw float64 bytes.

    The cross-harness handshake: two harnesses that seed the same environment
    the same way start from a bit-identical state and so agree on this digest,
    which is what lets episodes they flew independently be paired. Exact rather
    than tolerant, because the start is drawn from the seed alone and never
    from the timing -- a tolerance here would only hide a harness that had
    broken that.
    """
    return hashlib.sha256(np.asarray(state0, dtype=np.float64).tobytes()).hexdigest()[:16]


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
        """Raise `ValueError` unless this episode meets the file's contract.

        Every reader in the suite goes through here, on save and on load, so a
        file another harness wrote either matches this module's idea of an
        episode or fails naming the array and the value that did not.
        """
        self._check_meta_keys()
        spec = self._check_env()
        self._check_dtypes_and_ranks()
        self._check_shapes(spec)
        self._check_values(spec)

    def _check_meta_keys(self) -> None:
        missing = [key for key in META_KEYS if key not in self.meta]
        if missing:
            raise ValueError(f"meta.json is missing {missing}")
        method = self.meta["method"]
        if not isinstance(method, str) or not _METHOD_RE.match(method):
            raise ValueError(
                f"meta.json method {method!r} is not a bare file stem; it names the "
                f"clips and plots this episode renders to, so it must match "
                f"{_METHOD_RE.pattern}"
            )
        if self.meta["outcome"] not in OUTCOMES:
            raise ValueError(
                f"meta.json outcome '{self.meta['outcome']}' is not one of {OUTCOMES}"
            )

    def _check_env(self) -> EnvSpec:
        env = self.meta["env"]
        if env not in ENV_REGISTRY:
            raise ValueError(
                f"meta.json names env '{env}', which this build does not register "
                f"({', '.join(ENV_REGISTRY)})"
            )
        spec = ENV_REGISTRY[env]
        # The inline config is the file's promise that the scene can be rebuilt,
        # so it is checked here rather than left to fail inside a renderer half
        # an episode later.
        try:
            spec.config_cls.model_validate(self.meta["env_config"])
        except ValidationError as exc:
            raise ValueError(
                f"meta.json env_config does not rebuild {env}'s config: {exc}"
            ) from exc
        return spec

    def _check_dtypes_and_ranks(self) -> None:
        for name, dtype in _DTYPES.items():
            array = getattr(self, name)
            if array.dtype != dtype:
                raise ValueError(
                    f"{name} must be stored as {np.dtype(dtype).name}, got {array.dtype}"
                )
            if array.ndim != _RANKS[name]:
                raise ValueError(
                    f"{name} must be {_RANKS[name]}-D, got shape {array.shape}"
                )

    def _check_shapes(self, spec: EnvSpec) -> None:
        steps = self.steps
        for name in _PER_STATE:
            rows = getattr(self, name).shape[0]
            if rows != steps + 1:
                raise ValueError(f"{name} has {rows} rows, expected steps + 1 = {steps + 1}")
        for name in _PER_STEP:
            rows = getattr(self, name).shape[0]
            if rows != steps:
                raise ValueError(f"{name} has {rows} rows, expected steps = {steps}")
        state_dim = spec.layout.state_dim
        for name in ("state", "measured_state"):
            width = getattr(self, name).shape[1]
            if width != state_dim:
                raise ValueError(
                    f"{name} is {width} wide, but env '{spec.name}' has a "
                    f"{state_dim}-element state"
                )
        if self.epoch.shape[1] != 2:
            raise ValueError(f"epoch must be (T+1, 2), got {self.epoch.shape}")
        if self.rel_view.shape[1] != VIEW_DIM:
            raise ValueError(f"rel_view must be (T+1, {VIEW_DIM}), got {self.rel_view.shape}")
        if self.action_norm.shape[1] != 6 or self.action_phys.shape[1] != 6:
            raise ValueError("actions must be (T, 6)")
        if self.dock_target.shape != (7,):
            raise ValueError(f"dock_target must be (7,), got {self.dock_target.shape}")

    def _check_values(self, spec: EnvSpec) -> None:
        steps = self.steps
        if steps < 1:
            raise ValueError("a trajectory must hold at least one step")
        if int(self.meta["steps"]) != steps:
            raise ValueError(f"meta.json steps={self.meta['steps']} but arrays hold {steps}")
        dt = self.dt
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"meta.json dt={dt} must be finite and > 0")
        config_dt = self.meta["env_config"].get("dt")
        if config_dt is None:
            raise ValueError("meta.json env_config carries no dt")
        if abs(dt - float(config_dt)) >= 1e-12:
            raise ValueError(
                f"meta.json dt={dt} disagrees with env_config dt={config_dt}; rows are "
                "recorded at the environment's own integration step"
            )
        for name in _FLOAT_ARRAYS:
            array = getattr(self, name)
            if not np.isfinite(array).all():
                raise ValueError(f"{name} holds non-finite values")
        largest = float(np.abs(self.action_norm).max())
        if largest > 1.0 + _ACTION_TOL:
            raise ValueError(
                f"action_norm reaches {largest:g}, outside the normalised [-1, 1] a "
                "policy emits"
            )
        epoch_slice = spec.layout.epoch
        if epoch_slice is not None and not np.array_equal(
            self.epoch, self.state[:, epoch_slice]
        ):
            raise ValueError(
                f"epoch does not match state[:, {epoch_slice}]; the two must be the "
                f"same numbers for a clip to be lit at the episode's own time"
            )
        expected = start_fingerprint(self.state[0])
        if self.meta["start_fingerprint"] != expected:
            raise ValueError(
                f"meta.json start_fingerprint '{self.meta['start_fingerprint']}' is not "
                f"the digest of state[0] ('{expected}'); results keyed on it would pair "
                "the wrong episodes"
            )


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
