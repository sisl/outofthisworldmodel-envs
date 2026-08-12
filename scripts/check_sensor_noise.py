#!/usr/bin/env python
"""Compare a run's recorded observations to its true states.

For each split: per-channel total-RMS residuals (position, velocity, attitude
angle, body rate) against the sigmas in the run's env_config.yaml, plus
range-binned position error (the non-cooperative preset's error grows with
range). Exits non-zero when any measured RMS is off its configured value by
more than the tolerance, so a generated run can be checked before it is
published.

The residual is the measured 13D relative view minus the true state's view --
exactly the sensor-noise draw the simulator made for that frame -- so this
measures the noise that actually landed in the dataset rather than trusting
the config that produced it. Which env generated the run (and so which config
class parses its env_config.yaml, where the view sits in `observation_vector`,
and how `state_vector` maps to a view) is resolved from the run's own
dataset_card.json; see run_view.py.

Usage:
    uv run --extra datasets python scripts/check_sensor_noise.py RUN_DIR
    uv run --extra datasets python scripts/check_sensor_noise.py RUN_DIR --tolerance 0.05
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from run_view import observed_views, run_env, state_views

from owm_envs.envs.common.config import BaseTaskConfig
from owm_envs.envs.common.sensing import Sigma

# Position-error range bins, and the number of frames one needs before its RMS
# is worth comparing to anything: at 100 samples the relative standard error of
# a 3-axis RMS estimate is ~4%, already a fraction of the default tolerance.
NUM_RANGE_BINS = 5
MIN_BIN_SAMPLES = 100

CHANNELS = ("pos_rms_m", "vel_rms_m_s", "att_rms_rad", "rate_rms_rad_s")


def _quaternion_angles(q_measured: np.ndarray, q_true: np.ndarray) -> np.ndarray:
    """Rotation angle between two quaternion arrays, in radians.

    2*atan2(|v|, |w|) of the relative quaternion, NOT 2*arccos(|dot|). The
    datasets store float32 and the cooperative preset's attitude sigma is
    5e-5 rad, so the dot product of two stored quaternions sits at
    1 - 3e-10 -- three orders of magnitude inside float32's spacing at 1.0.
    The dot of a real 800-frame split takes four distinct values, and arccos,
    whose slope diverges there, turns that quantization into an attitude RMS
    of 6.4e-4 rad: 13x the true residual, enough to fail a correct run. The
    atan2 form reads the angle off the vector part instead, which is ~2e-5 in
    magnitude and so nowhere near its own rounding point, and is indifferent
    to both quaternion sign and normalisation.
    """
    w_t, v_t = q_true[:, 0:1], q_true[:, 1:4]
    w_m, v_m = q_measured[:, 0:1], q_measured[:, 1:4]
    # conj(q_true) (x) q_measured, Hamilton convention, scalar first.
    w = w_t * w_m + np.sum(v_t * v_m, axis=1, keepdims=True)
    v = w_t * v_m - w_m * v_t - np.cross(v_t, v_m)
    return 2.0 * np.arctan2(np.linalg.norm(v, axis=1), np.abs(w[:, 0]))


def residual_stats(obs: np.ndarray, truth: np.ndarray) -> dict:
    """Total-RMS residuals per channel, plus position error by true range.

    `obs` may be wider than 13 (a goal-error run appends a block that has no
    truth counterpart); only the measured-state dims take part.
    """
    obs = np.asarray(obs, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    delta = obs[:, :13] - truth

    def total_rms(block: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.sum(block**2, axis=1))))

    angles = _quaternion_angles(obs[:, 6:10], truth[:, 6:10])
    ranges = np.linalg.norm(truth[:, 0:3], axis=1)
    edges = np.linspace(0.0, max(float(ranges.max()), 1.0), NUM_RANGE_BINS + 1)
    # Upper edge inclusive on the last bin: the farthest frames are exactly the
    # ones a range-proportional error model is measured on.
    index = np.clip(np.digitize(ranges, edges) - 1, 0, NUM_RANGE_BINS - 1)

    by_range = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = index == i
        if mask.sum() >= MIN_BIN_SAMPLES:
            by_range.append({
                "range_lo": float(lo),
                "range_hi": float(hi),
                "range_rms_m": float(np.sqrt(np.mean(ranges[mask] ** 2))),
                "count": int(mask.sum()),
                "pos_rms_m": total_rms(delta[mask, 0:3]),
            })

    return {
        "pos_rms_m": total_rms(delta[:, 0:3]),
        "vel_rms_m_s": total_rms(delta[:, 3:6]),
        "att_rms_rad": float(np.sqrt(np.mean(angles**2))),
        "rate_rms_rad_s": total_rms(delta[:, 10:13]),
        "pos_err_by_range": by_range,
        "range_rms_m": float(np.sqrt(np.mean(ranges**2))),
        "count": int(obs.shape[0]),
    }


def total_sigma(value: Sigma) -> float:
    """Total-RMS of a sigma written either as a total or as per-axis values."""
    if isinstance(value, tuple):
        return float(np.linalg.norm(value))
    return float(value)


def expected_sigmas(cfg: BaseTaskConfig, range_rms: float = 0.0) -> dict[str, float]:
    """Total-RMS residual each channel should show, per the config.

    Position combines its constant and range-proportional terms as independent
    variances, so the expectation over a set of frames uses the RMS of their
    true ranges (E[||e||^2] = sigma^2 + frac^2 * E[range^2]) rather than the
    mean range, which would understate a run that spans a range of distances.
    """
    noise = cfg.sensor_noise
    if not noise.enabled:
        return dict.fromkeys(CHANNELS, 0.0)
    return {
        "pos_rms_m": math.hypot(
            total_sigma(noise.sigma_pos_m), noise.sigma_pos_frac_of_range * range_rms
        ),
        "vel_rms_m_s": total_sigma(noise.sigma_vel_m_s),
        "att_rms_rad": total_sigma(noise.sigma_att_rad),
        "rate_rms_rad_s": total_sigma(noise.sigma_rate_rad_s),
    }


def split_names(run_dir: Path) -> list[str]:
    return sorted(p.name for p in run_dir.iterdir() if (p / "data").is_dir())


def load_split(run_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """(observations, true states) for one split, concatenated over its files.

    Globs `**/*.parquet` under the split's data directory so a change in
    lerobot's chunk/file layout does not silently read nothing.

    Only the two columns this compares are read, one file at a time and stacked
    into dense arrays as they go: a 500k-frame split holds a numpy object per
    frame per vector column, so reading the actions, rewards, targets and
    bookkeeping alongside them costs gigabytes to answer a question about two.
    """
    files = sorted((run_dir / split / "data").rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {run_dir / split / 'data'}")
    if "state_vector" not in pq.ParquetFile(files[0]).schema_arrow.names:
        raise SystemExit(
            f"split '{split}' has no state_vector column: this run predates true-state "
            "recording, so its observations cannot be compared against truth. Regenerate "
            "it with the current owm-envs."
        )

    obs, truth = [], []
    for path in files:
        frame = pd.read_parquet(path, columns=["observation_vector", "state_vector"])
        obs.append(np.stack(frame["observation_vector"].to_numpy()))
        truth.append(np.stack(frame["state_vector"].to_numpy()))
    return np.concatenate(obs), np.concatenate(truth)


def _print_row(label: str, measured: float, target: float, tolerance: float, suffix: str = "") -> bool:
    """Print one measured-vs-expected line. True when it is within tolerance.

    A zero expectation admits no relative tolerance: nothing was injected on
    that channel, so any residual at all is a defect.
    """
    if target == 0.0:
        passed, relative = measured == 0.0, "-"
    else:
        passed = abs(measured - target) <= tolerance * target
        relative = f"{(measured - target) / target:+.1%}"
    print(f"  {label:<18} {measured:>12.6g} {target:>12.6g} {relative:>9}  "
          f"{'ok' if passed else 'FAIL'}{suffix}")
    return passed


def report_split(split: str, stats: dict, cfg: BaseTaskConfig, tolerance: float) -> bool:
    """Print the measured-vs-expected table. True when the split passes."""
    print(f"\n{split}: {stats['count']} frames, true range RMS {stats['range_rms_m']:.1f} m")
    print(f"  {'channel':<18} {'measured':>12} {'expected':>12} {'rel err':>9}  result")

    expected = expected_sigmas(cfg, stats["range_rms_m"])
    ok = True
    for channel in CHANNELS:
        ok &= _print_row(channel, stats[channel], expected[channel], tolerance)

    for row in stats["pos_err_by_range"]:
        target = expected_sigmas(cfg, row["range_rms_m"])["pos_rms_m"]
        label = f"pos {row['range_lo']:.0f}-{row['range_hi']:.0f} m"
        ok &= _print_row(label, row["pos_rms_m"], target, tolerance, f" ({row['count']} frames)")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="run directory written by owm-envs generate")
    parser.add_argument(
        "--tolerance", type=float, default=0.1,
        help="allowed relative deviation of a measured RMS from its configured "
             "sigma (default 0.1). A no-noise run is held to exact zeros regardless.",
    )
    parser.add_argument(
        "--split", action="append", dest="splits",
        help="split to check (repeatable); default is every split in the run",
    )
    args = parser.parse_args()

    env_config = args.run_dir / "env_config.yaml"
    if not env_config.is_file():
        raise SystemExit(f"{env_config} not found; is {args.run_dir} a run directory?")
    spec, cfg = run_env(args.run_dir)
    splits = args.splits or split_names(args.run_dir)
    if not splits:
        raise SystemExit(f"no splits found under {args.run_dir}")

    noise = cfg.sensor_noise
    print(f"{args.run_dir}: sensor noise {'enabled' if noise.enabled else 'disabled'}, "
          f"tolerance {args.tolerance:.0%}")

    ok = True
    for split in splits:
        obs, truth = load_split(args.run_dir, split)
        stats = residual_stats(observed_views(spec, cfg, obs), state_views(spec, truth))
        ok &= report_split(split, stats, cfg, args.tolerance)

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
