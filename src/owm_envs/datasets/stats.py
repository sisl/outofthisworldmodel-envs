"""Normalization statistics and the run-directory metadata.

A run directory holds normalization_stats.json, dataset_card.json and
summary.json, alongside env_config.yaml and policy_config.yaml -- the as-run
record of exactly what produced the data.

Those files are built and written in two steps (`build_run_metadata` then
`RunMetadata.write`) so their presence means the run finished, not merely
that it started.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from ..core.models import ConfigModel
from ..drivers.types import TrajectoryBatch
from ..envs.iss.config import ISSConfig
from ..envs.iss.policies import PolicyConfig

_STD_FLOOR = 1e-6


class SplitSpec(ConfigModel):
    num_episodes: int = 64
    max_steps: int = 2000
    seed: int = 0


class GenerationConfig(ConfigModel):
    splits: dict[str, SplitSpec] = Field(
        default_factory=lambda: {
            "train": SplitSpec(num_episodes=64, seed=0),
            "val": SplitSpec(num_episodes=8, seed=1),
        }
    )
    num_envs: int = 8
    # None means "use the environment's own simulation rate, 1/dt", which is
    # the rate the recorded frames actually occur at. A fixed default here
    # would silently disagree with dt for any dt but one.
    fps: int | None = None
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
    if act.shape[0] == 0:
        # Every episode is a lone observation with no step taken, so there is
        # nothing to take action statistics over. numpy would return NaN for
        # mean and std of an empty array, and json.dumps writes that as the
        # bare token NaN -- invalid JSON that a strict parser rejects and a
        # lenient one silently propagates into training as a NaN normalizer.
        raise ValueError(
            "batch holds no real actions: every episode has length 1, which "
            "stores a single observation and no transition"
        )
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


@dataclass(frozen=True)
class RunMetadata:
    """Built run metadata, not yet on disk.

    Building and writing are separate so a caller can compute this as soon as
    the rollout is done -- surfacing a bad batch before any expensive
    downstream work -- while `write` stays the last thing that touches the
    run directory. These files are what marks a run complete, so a run that
    fails partway leaves a directory visibly missing them rather than one
    that looks finished.
    """

    stats: dict
    card: dict
    counts: dict
    cfg: ISSConfig
    policy_cfg: PolicyConfig

    def write(self, run_dir: str | Path) -> None:
        """Flush stats, card, summary, and the as-run configs into `run_dir`."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "normalization_stats.json").write_text(json.dumps(self.stats, indent=2))
        (run_dir / "dataset_card.json").write_text(json.dumps(self.card, indent=2))
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "dataset_root": str(run_dir),
                    "counts": self.counts,
                    "normalization_stats": self.stats,
                },
                indent=2,
            )
        )

        # The as-run record: exactly the configuration that produced this data.
        self.cfg.to_yaml(run_dir / "env_config.yaml")
        self.policy_cfg.to_yaml(run_dir / "policy_config.yaml")


def build_run_metadata(
    *,
    cfg: ISSConfig,
    policy_cfg: PolicyConfig,
    batches: dict[str, TrajectoryBatch],
    fps: int,
    seed: int,
) -> RunMetadata:
    """Compute the run metadata. Touches no files -- see RunMetadata.write."""
    reference = batches.get("train") or next(iter(batches.values()))
    stats = compute_norm_stats(reference)

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

    return RunMetadata(
        stats=stats, card=card, counts=counts, cfg=cfg, policy_cfg=policy_cfg
    )
