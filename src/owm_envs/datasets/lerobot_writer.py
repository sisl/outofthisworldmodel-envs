"""LeRobot dataset writer -- the ISOLATED lerobot API surface.

Every lerobot call in this package lives in this file. quickdraw's own writer
carries the same note: the lerobot API "has drifted across versions", so
containing it to one function means a version bump touches one place.

Verified against lerobot 0.4.4:
  LeRobotDataset.create(repo_id, fps, root, features, use_videos)
  .add_frame(frame)   # frame dict carries "task"
  .save_episode()

lerobot is an optional extra. The import is function-local so the rest of the
package imports and tests without it.
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
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=fps, root=root, features=features, use_videos=False
    )

    for episode in range(batch.num_episodes):
        length = int(batch.lengths[episode])
        for t in range(length):
            dataset.add_frame(
                {
                    "observation_vector": np.asarray(
                        batch.observations[episode, t], dtype=np.float32
                    ),
                    "action": np.asarray(batch.actions[episode, t], dtype=np.float32),
                    "task": task_name,
                }
            )
        dataset.save_episode()

    return root
