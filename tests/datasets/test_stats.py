import json
import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from owm_envs.datasets.stats import (
    SUMMARY_FILENAME,
    GenerationConfig,
    SplitSpec,
    build_run_metadata,
    code_provenance,
    compute_norm_stats,
)
from owm_envs.drivers.types import TrajectoryBatch
from owm_envs.envs.iss.config import ISSConfig, PhysicsConfig
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


def test_action_stats_exclude_the_zero_pad_slot():
    """Two episodes of DIFFERENT lengths, each built to actually follow the
    episode convention: `lengths[i] - 1` real actions, then one genuine zero
    pad at index `lengths[i] - 1`. `batch_with_padding()` above does NOT model
    this (it fills every slot up to `lengths[i]`, pad slot included, with the
    "real" value), so it cannot catch a regression here -- this fixture can.
    """
    obs = np.zeros((2, 5, 13), dtype=np.float32)
    act = np.zeros((2, 5, 6), dtype=np.float32)
    # Episode 0: lengths=3 -> 2 real actions (indices 0-1), pad at index 2.
    act[0, 0:2, 0] = 5.0
    # Episode 1: lengths=5 -> 4 real actions (indices 0-3), pad at index 4.
    act[1, 0:4, 0] = 5.0
    batch = TrajectoryBatch(
        observations=obs,
        actions=act,
        rewards=np.zeros((2, 5), dtype=np.float32),
        lengths=np.array([3, 5], dtype=np.int32),
        terminated=np.array([True, True]),
        truncated=np.array([False, False]),
        policy_ids=None,
    )
    stats = compute_norm_stats(batch)
    # Real actions only: (2 + 4) rows all equal to 5.0 -> mean 5.0 exactly.
    # Including the pad slots would instead average in two zeros: rows would
    # be [5,5,0] + [5,5,5,5,0] = 8 rows summing to 30 -> mean 3.75.
    assert np.isclose(stats["action"]["mean"][0], 5.0, atol=1e-4)


def test_building_metadata_touches_no_files(tmp_path):
    # The separation exists so nothing lands on disk until every artifact of
    # the run has succeeded. If building wrote anything, a run that failed
    # later would still leave a directory that looks like a finished one.
    run_dir = tmp_path / "run"
    build_run_metadata(
        cfg=ISSConfig(), policy_cfg=PolicyConfig(),
        gen_cfg=GenerationConfig(splits={"train": SplitSpec(num_episodes=2, seed=0)}),
        batches={"train": batch_with_padding()}, fps=20,
    )
    assert not run_dir.exists()
    assert list(tmp_path.iterdir()) == []


def test_summary_marker_lands_last(tmp_path):
    # summary.json is what a consumer tests for, so it must not appear while
    # any other metadata file is still missing.
    seen_when_summary_appeared = {}
    real_replace = os.replace

    def watched_replace(src, dst):
        real_replace(src, dst)
        if Path(dst).name == SUMMARY_FILENAME:
            seen_when_summary_appeared["files"] = sorted(
                p.name for p in Path(dst).parent.iterdir() if not p.name.startswith(".")
            )

    with mock.patch.object(os, "replace", watched_replace):
        build_run_metadata(
            cfg=ISSConfig(), policy_cfg=PolicyConfig(),
            gen_cfg=GenerationConfig(splits={"train": SplitSpec(num_episodes=2, seed=0)}),
            batches={"train": batch_with_padding()}, fps=20,
        ).write(tmp_path)

    assert seen_when_summary_appeared["files"] == [
        "dataset_card.json",
        "env_config.yaml",
        "generation_config.yaml",
        "normalization_stats.json",
        "policy_config.yaml",
        SUMMARY_FILENAME,
    ]


def test_a_failure_mid_flush_leaves_no_summary_marker(tmp_path):
    # If the flush dies partway, the directory holds some metadata -- but not
    # the marker, so nothing reads it as a finished run.
    metadata = build_run_metadata(
        cfg=ISSConfig(), policy_cfg=PolicyConfig(),
        gen_cfg=GenerationConfig(splits={"train": SplitSpec(num_episodes=2, seed=0)}),
        batches={"train": batch_with_padding()}, fps=20,
    )
    with mock.patch.object(
        PolicyConfig, "to_yaml", side_effect=OSError("disk full")
    ), pytest.raises(OSError):
        metadata.write(tmp_path)

    assert (tmp_path / "normalization_stats.json").exists()  # got that far
    assert not (tmp_path / SUMMARY_FILENAME).exists()


def test_rewriting_a_run_drops_the_old_marker_first(tmp_path):
    # Writing over a finished run: the previous summary.json must not survive
    # a failed rewrite, or it advertises a directory that is now half old
    # metadata and half new as a complete, consistent run.
    build_run_metadata(
        cfg=ISSConfig(), policy_cfg=PolicyConfig(),
        gen_cfg=GenerationConfig(splits={"train": SplitSpec(num_episodes=2, seed=0)}),
        batches={"train": batch_with_padding()}, fps=20,
    ).write(tmp_path)
    assert (tmp_path / SUMMARY_FILENAME).exists()

    rewrite = build_run_metadata(
        cfg=ISSConfig(), policy_cfg=PolicyConfig(type="orbit"),
        gen_cfg=GenerationConfig(splits={"train": SplitSpec(num_episodes=2, seed=1)}),
        batches={"train": batch_with_padding()}, fps=50,
    )
    with mock.patch.object(
        PolicyConfig, "to_yaml", side_effect=OSError("disk full")
    ), pytest.raises(OSError):
        rewrite.write(tmp_path)

    assert not (tmp_path / SUMMARY_FILENAME).exists()
    # And the rewrite did get far enough to replace earlier files, which is
    # what makes the surviving marker a lie rather than merely stale.
    assert json.loads((tmp_path / "dataset_card.json").read_text())["fps"] == 50


def test_no_staging_files_survive_a_successful_write(tmp_path):
    build_run_metadata(
        cfg=ISSConfig(), policy_cfg=PolicyConfig(),
        gen_cfg=GenerationConfig(splits={"train": SplitSpec(num_episodes=2, seed=0)}),
        batches={"train": batch_with_padding()}, fps=20,
    ).write(tmp_path)
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []


def test_run_metadata_write_emits_every_expected_file(tmp_path):
    build_run_metadata(
        cfg=ISSConfig(),
        policy_cfg=PolicyConfig(type="union"),
        gen_cfg=GenerationConfig(splits={"train": SplitSpec(num_episodes=2, seed=0)}),
        batches={"train": batch_with_padding()},
        fps=24,
    ).write(tmp_path)
    for name in (
        "normalization_stats.json",
        "dataset_card.json",
        "summary.json",
        "env_config.yaml",
        "policy_config.yaml",
        "generation_config.yaml",
    ):
        assert (tmp_path / name).exists(), f"missing {name}"


def test_as_run_configs_round_trip(tmp_path):
    # The whole point of the as-run record: reload it and get the same config.
    cfg = ISSConfig(physics=PhysicsConfig(start_radius_m=250.0))
    policy_cfg = PolicyConfig(type="orbit")
    build_run_metadata(
        cfg=cfg, policy_cfg=policy_cfg,
        gen_cfg=GenerationConfig(splits={"train": SplitSpec(num_episodes=2, seed=0)}),
        batches={"train": batch_with_padding()}, fps=24,
    ).write(tmp_path)
    assert ISSConfig.from_yaml(tmp_path / "env_config.yaml") == cfg
    assert PolicyConfig.from_yaml(tmp_path / "policy_config.yaml") == policy_cfg


def test_summary_counts_real_transitions_not_padded_ones(tmp_path):
    build_run_metadata(
        cfg=ISSConfig(), policy_cfg=PolicyConfig(),
        gen_cfg=GenerationConfig(splits={"train": SplitSpec(num_episodes=2, seed=0)}),
        batches={"train": batch_with_padding()}, fps=24,
    ).write(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["counts"]["train"]["episodes"] == 2
    # lengths [4, 2] -> (4-1) + (2-1) = 4 real transitions, not 8 (padded
    # width) and not 6 (naive sum of lengths, which double-counts each
    # episode's terminal-observation slot as a transition).
    assert summary["counts"]["train"]["transitions"] == 4


def test_batch_with_no_real_actions_is_rejected():
    # lengths == 1 is a lone observation with no step taken, so there is
    # nothing to take action statistics over. numpy yields NaN for an empty
    # mean/std and json.dumps writes the bare token NaN, which is invalid
    # JSON -- unusable as a normalizer either way, so fail here instead.
    batch = TrajectoryBatch(
        observations=np.zeros((2, 1, 13), dtype=np.float32),
        actions=np.zeros((2, 1, 6), dtype=np.float32),
        rewards=np.zeros((2, 1), dtype=np.float32),
        lengths=np.array([1, 1], dtype=np.int32),
        terminated=np.array([True, False]),
        truncated=np.array([False, True]),
        policy_ids=None,
    )
    batch.validate()  # structurally valid; the problem is that it is empty
    with pytest.raises(ValueError, match="no real actions"):
        compute_norm_stats(batch)


def test_one_real_action_is_enough(tmp_path):
    # The guard above must not reject the shortest batch that does carry a
    # transition, and the stats it writes must stay finite JSON.
    obs = np.zeros((1, 2, 13), dtype=np.float32)
    act = np.zeros((1, 2, 6), dtype=np.float32)
    act[0, 0, 0] = 5.0
    batch = TrajectoryBatch(
        observations=obs,
        actions=act,
        rewards=np.zeros((1, 2), dtype=np.float32),
        lengths=np.array([2], dtype=np.int32),
        terminated=np.array([True]),
        truncated=np.array([False]),
        policy_ids=None,
    )
    stats = compute_norm_stats(batch)
    assert stats["action"]["mean"][0] == 5.0
    assert all(np.isfinite(stats["action"]["std"]))

    build_run_metadata(
        cfg=ISSConfig(), policy_cfg=PolicyConfig(),
        gen_cfg=GenerationConfig(splits={"train": SplitSpec(num_episodes=1, seed=0)}),
        batches={"train": batch}, fps=20,
    ).write(tmp_path)
    # json.loads rejects the bare NaN token only with parse_constant raising.
    text = (tmp_path / "normalization_stats.json").read_text()
    assert "NaN" not in text


def batch_with_constant_observations(value):
    """One episode whose every observation and real action equals `value`.

    Lets a test that mixes a train and a val batch assert which one a result
    actually came from, rather than merely that some result exists.
    """
    obs = np.full((1, 2, 13), value, dtype=np.float32)
    act = np.zeros((1, 2, 6), dtype=np.float32)
    act[0, 0, :] = value
    return TrajectoryBatch(
        observations=obs,
        actions=act,
        rewards=np.zeros((1, 2), dtype=np.float32),
        lengths=np.array([2], dtype=np.int32),
        terminated=np.array([True]),
        truncated=np.array([False]),
        policy_ids=None,
    )


def gen_cfg_for(batches):
    return GenerationConfig(
        splits={
            name: SplitSpec(num_episodes=b.num_episodes, max_steps=100, seed=i)
            for i, (name, b) in enumerate(batches.items())
        }
    )


def test_stats_come_from_the_train_batch_only():
    train = batch_with_constant_observations(2.0)
    val = batch_with_constant_observations(100.0)
    batches = {"train": train, "val": val}
    metadata = build_run_metadata(
        cfg=ISSConfig(), policy_cfg=PolicyConfig(), gen_cfg=gen_cfg_for(batches),
        batches=batches, fps=20,
    )
    assert metadata.stats == compute_norm_stats(train)


def test_metadata_requires_a_train_batch():
    val = batch_with_constant_observations(1.0)
    gen = GenerationConfig(splits={
        "train": SplitSpec(num_episodes=val.num_episodes, seed=0),
        "val": SplitSpec(num_episodes=val.num_episodes, seed=1),
    })
    with pytest.raises(ValueError, match="train"):
        build_run_metadata(cfg=ISSConfig(), policy_cfg=PolicyConfig(),
                           gen_cfg=gen, batches={"val": val}, fps=20)


def test_metadata_requires_a_batch_for_every_configured_split():
    train = batch_with_constant_observations(1.0)
    gen = GenerationConfig(splits={
        "train": SplitSpec(num_episodes=train.num_episodes, seed=0),
        "val": SplitSpec(num_episodes=2, seed=1),
    })
    with pytest.raises(ValueError, match="no generated batch"):
        build_run_metadata(cfg=ISSConfig(), policy_cfg=PolicyConfig(),
                           gen_cfg=gen, batches={"train": train}, fps=20)


def test_card_records_per_split_seed_and_provenance():
    train = batch_with_constant_observations(1.0)
    batches = {"train": train}
    metadata = build_run_metadata(
        cfg=ISSConfig(), policy_cfg=PolicyConfig(), gen_cfg=gen_cfg_for(batches),
        batches=batches, fps=20,
    )
    assert metadata.card["splits"]["train"]["seed"] == 0
    assert metadata.card["splits"]["train"]["max_steps"] == 100
    assert "seed" not in metadata.card
    assert "policy_type" not in metadata.card
    assert set(metadata.card["provenance"]) == {"owm_envs_version", "git_commit", "git_dirty"}


def test_card_resolves_the_per_split_policy():
    train = batch_with_constant_observations(1.0)
    val = batch_with_constant_observations(1.0)
    gen = GenerationConfig(splits={
        "train": SplitSpec(num_episodes=train.num_episodes, seed=0),
        "val": SplitSpec(num_episodes=val.num_episodes, seed=1,
                         policy=PolicyConfig(type="dock")),
    })
    metadata = build_run_metadata(
        cfg=ISSConfig(), policy_cfg=PolicyConfig(type="union"), gen_cfg=gen,
        batches={"train": train, "val": val}, fps=20,
    )
    assert metadata.card["splits"]["train"]["policy_type"] == "union"
    assert metadata.card["splits"]["val"]["policy_type"] == "dock"


def test_write_records_the_generation_config(tmp_path):
    train = batch_with_constant_observations(1.0)
    batches = {"train": train}
    metadata = build_run_metadata(
        cfg=ISSConfig(), policy_cfg=PolicyConfig(), gen_cfg=gen_cfg_for(batches),
        batches=batches, fps=20,
    )
    metadata.write(tmp_path)
    assert GenerationConfig.from_yaml(tmp_path / "generation_config.yaml") == metadata.gen_cfg


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


def test_generation_config_requires_a_train_split():
    with pytest.raises(ValueError, match="train"):
        GenerationConfig(splits={"val": SplitSpec(num_episodes=2, seed=1)})


def test_generation_config_rejects_duplicate_split_seeds():
    with pytest.raises(ValueError, match="unique"):
        GenerationConfig(
            splits={
                "train": SplitSpec(num_episodes=4, seed=3),
                "val": SplitSpec(num_episodes=2, seed=3),
            }
        )


def test_split_spec_policy_round_trips_through_yaml(tmp_path):
    gen = GenerationConfig(splits={
        "train": SplitSpec(num_episodes=4, seed=0, policy=PolicyConfig(type="union")),
        "val": SplitSpec(num_episodes=2, seed=1, policy=PolicyConfig(type="dock")),
    })
    path = tmp_path / "gen.yaml"
    gen.to_yaml(path)
    loaded = GenerationConfig.from_yaml(path)
    assert loaded == gen
    assert loaded.splits["val"].policy.type == "dock"
    assert loaded.splits["train"].policy.union_weights == (0.3, 0.35, 0.35)


def test_code_provenance_reports_version_and_git_state():
    prov = code_provenance()
    assert set(prov) == {"owm_envs_version", "git_commit", "git_dirty"}
    # This test runs from the git checkout, so the commit must resolve.
    assert isinstance(prov["git_commit"], str) and len(prov["git_commit"]) == 40
    assert isinstance(prov["git_dirty"], bool)


def test_code_provenance_degrades_when_git_is_unavailable(monkeypatch):
    import owm_envs.datasets.stats as stats_mod

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git not on PATH")

    monkeypatch.setattr(stats_mod.subprocess, "run", no_git)
    prov = code_provenance()
    assert prov["git_commit"] is None
    assert prov["git_dirty"] is None


def test_negative_split_seed_is_rejected():
    with pytest.raises(ValueError):
        SplitSpec(num_episodes=2, seed=-1)


def test_non_positive_split_sizes_are_rejected():
    with pytest.raises(ValueError):
        SplitSpec(num_episodes=0, seed=0)
    with pytest.raises(ValueError):
        SplitSpec(num_episodes=2, max_steps=0, seed=0)


def test_path_like_split_names_are_rejected():
    with pytest.raises(ValueError, match="slug"):
        GenerationConfig(splits={
            "train": SplitSpec(num_episodes=2, seed=0),
            "../evil": SplitSpec(num_episodes=2, seed=1),
        })
