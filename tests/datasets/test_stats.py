import json

import numpy as np
import pytest

from owm_envs.datasets.stats import GenerationConfig, SplitSpec, compute_norm_stats, write_run_metadata
from owm_envs.drivers.types import TrajectoryBatch
from owm_envs.envs.iss.config import ISSConfig
from owm_envs.envs.iss.policies import PolicyConfig


def batch_with_padding():
    """2 episodes of width 4; the second is only 2 steps long, rest is padding."""
    obs = np.zeros((2, 4, 13), dtype=np.float32)
    act = np.zeros((2, 4, 6), dtype=np.float32)
    obs[0, :, 0] = [1.0, 1.0, 1.0, 1.0]
    obs[1, :2, 0] = [3.0, 3.0]
    act[0, :, 0] = 2.0
    act[1, :2, 0] = 4.0
    return TrajectoryBatch(
        observations=obs,
        actions=act,
        rewards=np.zeros((2, 4), dtype=np.float32),
        lengths=np.array([4, 2], dtype=np.int32),
        terminated=np.array([False, True]),
        truncated=np.array([True, False]),
        policy_ids=None,
    )


def test_stats_ignore_padding():
    # Real obs[...,0] values are [1,1,1,1,3,3] -> mean 10/6 = 1.6667.
    # Averaging the zero-padded array instead gives [1,1,1,1,3,3,0,0] = 10/8 = 1.25,
    # so this assertion fails loudly if padding leaks into the statistics.
    stats = compute_norm_stats(batch_with_padding())
    assert np.isclose(stats["observation_vector"]["mean"][0], 10.0 / 6.0, atol=1e-4)


def test_stats_have_one_entry_per_dimension():
    stats = compute_norm_stats(batch_with_padding())
    assert len(stats["observation_vector"]["mean"]) == 13
    assert len(stats["observation_vector"]["std"]) == 13
    assert len(stats["action"]["mean"]) == 6
    assert len(stats["action"]["std"]) == 6


def test_std_is_never_zero():
    # A constant dimension would give std 0 and produce div-by-zero downstream.
    stats = compute_norm_stats(batch_with_padding())
    assert all(s > 0.0 for s in stats["observation_vector"]["std"])
    assert all(s > 0.0 for s in stats["action"]["std"])


def test_write_run_metadata_emits_every_expected_file(tmp_path):
    write_run_metadata(
        tmp_path,
        cfg=ISSConfig(),
        policy_cfg=PolicyConfig(type="union"),
        batches={"train": batch_with_padding()},
        fps=24,
        seed=0,
    )
    for name in (
        "normalization_stats.json",
        "dataset_card.json",
        "summary.json",
        "env_config.yaml",
        "policy_config.yaml",
    ):
        assert (tmp_path / name).exists(), f"missing {name}"


def test_as_run_configs_round_trip(tmp_path):
    # The whole point of the as-run record: reload it and get the same config.
    cfg = ISSConfig(start_radius_m=250.0)
    policy_cfg = PolicyConfig(type="orbit")
    write_run_metadata(tmp_path, cfg=cfg, policy_cfg=policy_cfg, batches={"train": batch_with_padding()}, fps=24, seed=0)
    assert ISSConfig.from_yaml(tmp_path / "env_config.yaml") == cfg
    assert PolicyConfig.from_yaml(tmp_path / "policy_config.yaml") == policy_cfg


def test_summary_counts_real_transitions_not_padded_ones(tmp_path):
    write_run_metadata(tmp_path, cfg=ISSConfig(), policy_cfg=PolicyConfig(), batches={"train": batch_with_padding()}, fps=24, seed=0)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["counts"]["train"]["episodes"] == 2
    assert summary["counts"]["train"]["transitions"] == 6  # 4 + 2, not 8


def test_generation_config_round_trips(tmp_path):
    gen = GenerationConfig(
        splits={"train": SplitSpec(num_episodes=8, max_steps=100, seed=0),
                "val": SplitSpec(num_episodes=2, max_steps=100, seed=1)},
        num_envs=4,
        fps=24,
        driver="auto",
    )
    path = tmp_path / "gen.yaml"
    gen.to_yaml(path)
    assert GenerationConfig.from_yaml(path) == gen


def test_generation_config_rejects_an_unknown_driver():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        GenerationConfig(splits={}, num_envs=4, fps=24, driver="teleport")
