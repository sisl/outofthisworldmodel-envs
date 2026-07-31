"""LeRobot dataset writer -- the ISOLATED lerobot API surface.

Every lerobot call in this package lives in this file, so a version bump of a
library whose API drifts across releases touches one place.

Targets the lerobot 0.4.4 API:
  LeRobotDataset.create(repo_id, fps, root, features, use_videos)
  .add_frame(frame)   # frame dict carries "task"
  .save_episode()

lerobot is an optional extra. The import is function-local so the rest of the
package imports and tests without it.

Schema: `observation_vector` (float32, obs_dim) and `action` (float32,
act_dim) carry the trajectory itself. Five more features carry outcome
metadata that would otherwise be lost once a TrajectoryBatch is discarded:
`reward` (the per-frame reward; zero on an episode's final frame, whose
action slot is a zero pad rather than a real action), `is_last`,
`terminated`, `truncated`, and `policy_id`.

`is_last` is per-FRAME and true only on an episode's final frame. It marks
both the terminal observation and the one frame whose action is the zero pad,
which must be dropped when forming (observation, action, next observation)
transitions. `terminated`, `truncated` and `policy_id` are per-EPISODE facts
written onto every frame of that episode -- deliberate redundancy so a
dataset can be filtered frame-wise (e.g. "give me every frame of every
collision episode") without joining back to episode boundaries. Combining
them, `is_last & terminated` selects exactly the collision and dock frames,
and `is_last & truncated` exactly the horizon cuts.

Episode boundaries are recoverable from lerobot's own bookkeeping too --
`meta.episodes["length"]`, and `frame_index` restarting at 0 -- and that
agrees with `is_last` by construction: only real frames are written, so a
lerobot episode's length equals that episode's `TrajectoryBatch.lengths`
entry.

`reward` and `policy_id` use shape (1, 1), not the seemingly more natural
scalar shape (1,). This is not a style choice: lerobot 0.4.4's schema-to-HF
conversion (`get_hf_features_from_features`) special-cases any feature whose
shape is exactly `(1,)` into a plain HuggingFace `datasets.Value`, whose
`encode_example` calls Python's `float()`/`int()` on the stored array. Under
numpy>=2.0 (installed: 2.5.1) those raise `TypeError: only 0-dimensional
arrays can be converted to Python scalars` for any array with `ndim != 0`,
which every value `add_frame` accepts is (its own validator requires an
`np.ndarray` with `.shape` matching the declared feature shape, i.e. `(1,)`,
which always has `ndim == 1`). The boolean features `is_last`, `terminated`
and `truncated` survive at shape `(1,)` only because `bool()` -- unlike
`float()`/`int()` -- does not raise for a single-element array;
`reward`/`policy_id` do not, for either float32 or int64. Shape `(1, 1)`
takes the `Array2D` path instead, which round-trips correctly. It reads back
as a (1, 1) array rather than a bare scalar; callers should `.reshape(())` or
index `[0, 0]`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..drivers.types import TrajectoryBatch


def write_lerobot_split(
    root: str | Path,
    repo_id: str,
    batch: TrajectoryBatch,
    fps: int,
    task_name: str = "iss_docking",
) -> Path:
    """Write one TrajectoryBatch as a LeRobotDataset on disk. Returns its root.

    Only real frames are written; padding past each episode's length is skipped.
    """
    batch.validate()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    features = {
        "observation_vector": {
            "dtype": "float32",
            "shape": (batch.observations.shape[-1],),
            "names": None,
        },
        "action": {"dtype": "float32", "shape": (batch.actions.shape[-1],), "names": None},
        # (1, 1), not (1,): see the module docstring for why the scalar shape
        # is unwritable for non-boolean features on this lerobot version.
        "reward": {"dtype": "float32", "shape": (1, 1), "names": None},
        "is_last": {"dtype": "bool", "shape": (1,), "names": None},
        "terminated": {"dtype": "bool", "shape": (1,), "names": None},
        "truncated": {"dtype": "bool", "shape": (1,), "names": None},
        "policy_id": {"dtype": "int64", "shape": (1, 1), "names": None},
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=fps, root=root, features=features, use_videos=False
    )

    for episode in range(batch.num_episodes):
        length = int(batch.lengths[episode])
        terminated = bool(batch.terminated[episode])
        truncated = bool(batch.truncated[episode])
        policy_id = int(batch.policy_ids[episode]) if batch.policy_ids is not None else 0
        for t in range(length):
            dataset.add_frame(
                {
                    "observation_vector": np.asarray(
                        batch.observations[episode, t], dtype=np.float32
                    ),
                    "action": np.asarray(batch.actions[episode, t], dtype=np.float32),
                    "reward": np.array([[batch.rewards[episode, t]]], dtype=np.float32),
                    "is_last": np.array([t == length - 1]),
                    "terminated": np.array([terminated]),
                    "truncated": np.array([truncated]),
                    "policy_id": np.array([[policy_id]], dtype=np.int64),
                    "task": task_name,
                }
            )
        dataset.save_episode()

    return root
