"""Normalization statistics and the run-directory metadata.

A run directory holds normalization_stats.json, dataset_card.json and
summary.json, alongside env_config.yaml and policy_config.yaml -- the as-run
record of exactly what produced the data.

Those files are built and written in two steps (`build_run_metadata` then
`RunMetadata.write`) so their presence means the run finished, not merely
that it started. `SUMMARY_FILENAME` is the marker a consumer should test
for; it lands last, by a single atomic rename.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from ..core.models import ConfigModel
from ..drivers.types import TrajectoryBatch
from ..envs.iss.config import ISSConfig
from ..envs.iss.policies import PolicyConfig

_STD_FLOOR = 1e-6

# The file whose presence means a run finished. See RunMetadata.write.
SUMMARY_FILENAME = "summary.json"


def _staging_path(final: Path) -> Path:
    """Temporary sibling of `final` to build content at before moving it.

    A sibling rather than a system temp file: `os.replace` is only atomic
    within one filesystem, and /tmp is routinely a different one.
    """
    return final.with_name(f".{final.name}.partial")


def _place(final: Path, text: str) -> None:
    """Put `text` at `final` without it ever being visible half-written.

    Writing directly would leave the path present but truncated for the
    duration of the write, so a reader could parse an incomplete file. The
    content is built under a temporary name and moved in one operation.
    """
    staged = _staging_path(final)
    staged.write_text(text)
    os.replace(staged, final)


def code_provenance() -> dict:
    """Version and git state of the code that produced a run.

    Best-effort by design: an installed wheel has no git checkout, and git
    may be absent entirely, so missing pieces are None rather than errors --
    provenance must never be the reason a data run fails.
    """
    try:
        version: str | None = importlib_metadata.version("owm-envs")
    except importlib_metadata.PackageNotFoundError:
        version = None

    package_dir = str(Path(__file__).resolve().parent)

    def _git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", package_dir, *args],
                capture_output=True, text=True, timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain") if commit is not None else None
    return {
        "owm_envs_version": version,
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
    }


class SplitSpec(ConfigModel):
    # Exactly one of these two sizes the split: episode count, or usable
    # (state, action, next-state) transitions (an episode of length L
    # contributes L - 1). See `_validate_mode`.
    num_episodes: int | None = Field(default=None, ge=1)
    min_transitions: int | None = Field(default=None, ge=1)
    max_steps: int = Field(default=7200, ge=1)
    seed: int = Field(default=0, ge=0)
    # None inherits the run-level policy. Set it to give this split its own
    # -- e.g. a dock-only val split against a union-policy train split.
    policy: PolicyConfig | None = None

    @model_validator(mode="after")
    def _validate_mode(self) -> "SplitSpec":
        if (self.num_episodes is None) == (self.min_transitions is None):
            raise ValueError(
                "exactly one of num_episodes or min_transitions must be set"
            )
        return self


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

    @field_validator("splits")
    @classmethod
    def _validate_splits(cls, v: dict[str, SplitSpec]) -> dict[str, SplitSpec]:
        for name in v:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or ".." in name:
                raise ValueError(
                    f"split name {name!r} must be a plain slug "
                    "(letters, digits, '_', '-', '.'; no path separators)"
                )
        # Normalization statistics are computed from the train split alone
        # (the downstream trainer's contract), so a run without one has no
        # valid stats at all -- reject at config load, not at metadata time.
        if "train" not in v:
            raise ValueError("splits must include a 'train' split")
        seeds = [spec.seed for spec in v.values()]
        if len(set(seeds)) != len(seeds):
            # Two splits sharing a seed produce identical trajectories --
            # silent train/val leakage rather than an error downstream.
            raise ValueError(f"split seeds must be unique, got {seeds}")
        return v


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
    gen_cfg: GenerationConfig

    def write(self, run_dir: str | Path) -> None:
        """Flush stats, card, summary, and the as-run configs into `run_dir`.

        SUMMARY_FILENAME is the completion marker. Five files cannot be
        created in one filesystem operation, so instead each is staged under
        a temporary name and moved into place by `os.replace`, and the marker
        is moved last. A reader therefore never observes a truncated file,
        and the marker's own move -- a single atomic rename -- is the instant
        the run becomes complete. Everything else is already in place by
        then, so `summary.json` present means all of it is.

        Consumers should test for the marker, not for any other file: the
        others exist during the flush, before the run is finished.
        """
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Writing over an earlier run: drop its marker before touching
        # anything else. Left in place it would vouch for a mixture of the
        # old files and the new ones for the length of the flush, and if a
        # replacement below failed it would go on vouching for that mixture
        # indefinitely -- the exact state the marker exists to rule out.
        (run_dir / SUMMARY_FILENAME).unlink(missing_ok=True)

        _place(run_dir / "normalization_stats.json", json.dumps(self.stats, indent=2))
        _place(run_dir / "dataset_card.json", json.dumps(self.card, indent=2))

        # The as-run record: exactly the configuration that produced this data.
        # ConfigModel.to_yaml writes straight to the path it is given, so it
        # gets staged the same way rather than serialized in place.
        for model, name in (
            (self.cfg, "env_config.yaml"),
            (self.policy_cfg, "policy_config.yaml"),
            (self.gen_cfg, "generation_config.yaml"),
        ):
            staged = _staging_path(run_dir / name)
            model.to_yaml(staged)
            os.replace(staged, run_dir / name)

        _place(
            run_dir / SUMMARY_FILENAME,
            json.dumps(
                {
                    "dataset_root": str(run_dir),
                    "counts": self.counts,
                    "normalization_stats": self.stats,
                },
                indent=2,
            ),
        )


def build_run_metadata(
    *,
    cfg: ISSConfig,
    policy_cfg: PolicyConfig,
    gen_cfg: GenerationConfig,
    batches: dict[str, TrajectoryBatch],
    fps: int,
) -> RunMetadata:
    """Compute the run metadata. Touches no files -- see RunMetadata.write."""
    if "train" not in batches:
        raise ValueError("batches must include 'train': normalization stats are train-only")
    missing = set(batches) - set(gen_cfg.splits)
    if missing:
        raise ValueError(f"batches {sorted(missing)} have no matching entry in gen_cfg.splits")
    unbatched = set(gen_cfg.splits) - set(batches)
    if unbatched:
        # Otherwise the as-run generation_config.yaml would claim splits
        # that were never generated.
        raise ValueError(
            f"configured splits {sorted(unbatched)} have no generated batch"
        )
    stats = compute_norm_stats(batches["train"])

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

    def _split_policy(name: str) -> PolicyConfig:
        return gen_cfg.splits[name].policy or policy_cfg

    card = {
        "env": "iss",
        "fps": fps,
        "dt": cfg.dt,
        "splits": {
            name: {
                "episodes": b.num_episodes,
                "transitions": b.total_transitions,
                "seed": gen_cfg.splits[name].seed,
                "max_steps": gen_cfg.splits[name].max_steps,
                "min_transitions": gen_cfg.splits[name].min_transitions,
                "num_episodes_requested": gen_cfg.splits[name].num_episodes,
                "policy_type": _split_policy(name).type,
                "union_weights": list(_split_policy(name).union_weights),
            }
            for name, b in batches.items()
        },
        "provenance": code_provenance(),
    }

    return RunMetadata(
        stats=stats, card=card, counts=counts,
        cfg=cfg, policy_cfg=policy_cfg, gen_cfg=gen_cfg,
    )
