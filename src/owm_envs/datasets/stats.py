"""Normalization statistics and the run-directory metadata.

Layout mirrors quickdraw's data_generation output so a generated run can be
consumed by that training stack without an adapter: normalization_stats.json,
dataset_card.json, summary.json. Added here: env_config.yaml and
policy_config.yaml, the as-run record of exactly what produced the data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from ..core.config_io import YamlModel
from ..drivers.types import TrajectoryBatch
from ..envs.iss.config import ISSConfig
from ..envs.iss.policies import PolicyConfig

_STD_FLOOR = 1e-6


class SplitSpec(YamlModel):
    num_episodes: int = 64
    max_steps: int = 2000
    seed: int = 0


class GenerationConfig(YamlModel):
    splits: dict[str, SplitSpec] = Field(
        default_factory=lambda: {
            "train": SplitSpec(num_episodes=64, seed=0),
            "val": SplitSpec(num_episodes=8, seed=1),
        }
    )
    num_envs: int = 8
    fps: int = 24
    driver: Literal["auto", "scan", "vector"] = "auto"


def _real_rows(arr: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Concatenate only the unpadded rows: (E, T, D) + lengths -> (sum(lengths), D)."""
    return np.concatenate([arr[i, : int(n)] for i, n in enumerate(lengths)], axis=0)


def compute_norm_stats(batch: TrajectoryBatch) -> dict:
    """Per-dimension mean and std over REAL transitions only.

    Padding is zeros; including it would drag every mean toward zero in
    proportion to how often episodes terminate early, silently corrupting
    normalization for anything trained downstream.

    Observations use all `lengths[i]` entries per episode. Actions do not:
    per TrajectoryBatch's convention, the action at index `lengths[i] - 1` is
    a zero-padded no-op past the terminal state, not a real action -- it sits
    inside `lengths` but isn't real. Including it would mix one zero action
    into every episode's statistics, the same padding-bias bug this function
    exists to avoid, just smaller. So actions use `lengths[i] - 1` entries.
    """
    obs = _real_rows(batch.observations, batch.lengths)
    # NOT a typo / NOT the same call as above with `lengths` swapped for
    # `lengths - 1` by mistake: the last action slot inside `lengths` is a
    # zero pad, not a taken action (see TrajectoryBatch's docstring), so it
    # must be dropped here even though the observation at that same index is
    # real and IS kept above. rewards are not touched by this function at
    # all -- no run metadata currently reports reward statistics.
    act = _real_rows(batch.actions, np.maximum(batch.lengths - 1, 0))
    return {
        "observation_vector": {
            "mean": obs.mean(0).tolist(),
            "std": (obs.std(0) + _STD_FLOOR).tolist(),
        },
        "action": {
            "mean": act.mean(0).tolist(),
            "std": (act.std(0) + _STD_FLOOR).tolist(),
        },
    }


def write_run_metadata(
    run_dir: str | Path,
    *,
    cfg: ISSConfig,
    policy_cfg: PolicyConfig,
    batches: dict[str, TrajectoryBatch],
    fps: int,
    seed: int,
) -> None:
    """Write stats, card, summary, and the as-run configs into `run_dir`."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    reference = batches.get("train") or next(iter(batches.values()))
    stats = compute_norm_stats(reference)
    (run_dir / "normalization_stats.json").write_text(json.dumps(stats, indent=2))

    counts = {}
    for name, batch in batches.items():
        transitions = batch.total_transitions
        seconds = transitions * cfg.dt
        counts[name] = {
            "episodes": batch.num_episodes,
            "transitions": transitions,
            "terminated": int(batch.terminated.sum()),
            "truncated": int(batch.truncated.sum()),
            "seconds": round(seconds, 2),
            "minutes": round(seconds / 60.0, 3),
            "hours": round(seconds / 3600.0, 5),
        }

    card = {
        "env": "iss",
        "fps": fps,
        "seed": seed,
        "dt": cfg.dt,
        "splits": {
            name: {"episodes": b.num_episodes, "transitions": b.total_transitions}
            for name, b in batches.items()
        },
        "policy_type": policy_cfg.type,
        "union_weights": list(policy_cfg.union_weights),
    }
    (run_dir / "dataset_card.json").write_text(json.dumps(card, indent=2))

    (run_dir / "summary.json").write_text(
        json.dumps(
            {"dataset_root": str(run_dir), "counts": counts, "normalization_stats": stats},
            indent=2,
        )
    )

    # The as-run record: exactly the configuration that produced this data.
    cfg.to_yaml(run_dir / "env_config.yaml")
    policy_cfg.to_yaml(run_dir / "policy_config.yaml")
