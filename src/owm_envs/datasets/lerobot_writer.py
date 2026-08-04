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
act_dim) carry the trajectory itself. Six more features carry outcome
metadata that would otherwise be lost once a TrajectoryBatch is discarded:
`reward` (the per-frame reward; zero on an episode's final frame, whose
action slot is a zero pad rather than a real action), `is_last`,
`terminated`, `truncated`, `policy_id`, and `dock_target`.

`is_last` is per-FRAME and true only on an episode's final frame. It marks
both the terminal observation and the one frame whose action is the zero pad,
which must be dropped when forming (observation, action, next observation)
transitions. `terminated`, `truncated`, `policy_id` and `dock_target` are
per-EPISODE facts written onto every frame of that episode -- deliberate
redundancy so a dataset can be filtered frame-wise (e.g. "give me every frame
of every collision episode") without joining back to episode boundaries.
Combining them, `is_last & terminated` selects exactly the collision and dock
frames, and `is_last & truncated` exactly the horizon cuts.

`dock_target` is the (1, 7) [position, quaternion] pose the episode was flying
to: the port it was assigned under a port set, or the `DockConfig` pose for a
run that configured none. The pose itself rather than an index into the port
table, because the table's indices shift whenever a port is added or removed,
so a stored index stops reproducing the episode as soon as the table changes;
a pose is self-contained and stays comparable across table versions. A batch
whose `dock_targets` is None -- a driver whose policy source cannot supply one
-- writes all-NaN rows, which no real pose can collide with.

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
index `[0, 0]`. `dock_target` takes the same `Array2D` path at (1, 7), which
is not a special case at all -- it reads back as a (1, 7) array, and callers
want the row rather than a scalar anyway.

One further feature, `observation.images.fpv`, carries an egocentric video
clip per episode. It is only declared when the caller passes `frames` -- a
list of per-episode `(L, H, W, 3)` uint8 clips, one per episode in `batch`,
each `L` matching that episode's `lengths[i]` exactly. Frame rendering itself
does not happen here; it lives in `datasets/video.py`, kept out of this file
because rendering needs `owm_envs.render` (an optional extra of its own) and
this file's only job is staying the sole lerobot call site.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ..drivers.types import TrajectoryBatch


def write_lerobot_split(
    root: str | Path,
    repo_id: str,
    batch: TrajectoryBatch,
    fps: int,
    task_name: str = "iss_docking",
    frames: Sequence[np.ndarray] | None = None,
) -> Path:
    """Write one TrajectoryBatch as a LeRobotDataset on disk. Returns its root.

    Only real frames are written; padding past each episode's length is skipped.

    `frames`, when given, is one `(L, H, W, 3)` uint8 clip per episode in
    `batch`, aligned 1:1 with that episode's stored observations -- `L` must
    equal `batch.lengths[i]`, or a ValueError names the mismatching episode.
    """
    batch.validate()
    if frames is not None:
        _validate_frames(frames, batch)

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
        "dock_target": {"dtype": "float32", "shape": (1, 7), "names": None},
    }
    if frames is not None:
        # Derived from the actual clip, not assumed square: the renderer
        # returns (H, W, 3), and H need not equal W.
        height, width, channels = np.asarray(frames[0]).shape[1:]
        features["observation.images.fpv"] = {
            "dtype": "video",
            "shape": (int(height), int(width), int(channels)),
            "names": ["height", "width", "channels"],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=fps, root=root, features=features, use_videos=frames is not None
    )

    for episode in range(batch.num_episodes):
        length = int(batch.lengths[episode])
        terminated = bool(batch.terminated[episode])
        truncated = bool(batch.truncated[episode])
        policy_id = int(batch.policy_ids[episode]) if batch.policy_ids is not None else 0
        dock_target = (
            np.asarray(batch.dock_targets[episode], dtype=np.float32).reshape(1, 7)
            if batch.dock_targets is not None
            else np.full((1, 7), np.nan, dtype=np.float32)
        )
        clip = np.asarray(frames[episode]) if frames is not None else None
        for t in range(length):
            frame = {
                "observation_vector": np.asarray(
                    batch.observations[episode, t], dtype=np.float32
                ),
                "action": np.asarray(batch.actions[episode, t], dtype=np.float32),
                "reward": np.array([[batch.rewards[episode, t]]], dtype=np.float32),
                "is_last": np.array([t == length - 1]),
                "terminated": np.array([terminated]),
                "truncated": np.array([truncated]),
                "policy_id": np.array([[policy_id]], dtype=np.int64),
                "dock_target": dock_target,
                "task": task_name,
            }
            if clip is not None:
                frame["observation.images.fpv"] = np.asarray(clip[t], dtype=np.uint8)
            dataset.add_frame(frame)
        dataset.save_episode()

    return root


def _validate_frames(frames: Sequence[np.ndarray], batch: TrajectoryBatch) -> None:
    """Raise ValueError if `frames` doesn't align 1:1 with `batch`'s episodes.

    A clip shorter or longer than its episode would silently desynchronise
    video from state for every downstream consumer, so this compares length
    per episode, not just episode count.
    """
    if len(frames) != batch.num_episodes:
        raise ValueError(
            f"frames has {len(frames)} episodes, batch has {batch.num_episodes}"
        )
    # The declared feature shape comes from episode 0's clip, so every other
    # clip's frame shape (H, W, C) must agree with it too.
    expected_frame_shape = np.asarray(frames[0]).shape[1:]
    for episode in range(batch.num_episodes):
        clip = np.asarray(frames[episode])
        clip_length = int(clip.shape[0])
        expected_length = int(batch.lengths[episode])
        if clip_length != expected_length:
            raise ValueError(
                f"episode {episode}: frame clip length {clip_length} does not "
                f"match batch length {expected_length}"
            )
        if clip.shape[1:] != expected_frame_shape:
            raise ValueError(
                f"episode {episode}: frame shape {clip.shape[1:]} does not "
                f"match episode 0's frame shape {expected_frame_shape}"
            )
