import json

import pytest
from typer.testing import CliRunner

from owm_envs.cli import app
from owm_envs.envs.common.config import DockConfig
from owm_envs.envs.iss.config import ISSConfig

runner = CliRunner()


def run(*args):
    """Invoke `rollout` against the iss env with rendering off.

    The iss env rather than iss-numerical: same task, same reward, same
    outcome classification, and a rollout finishes in a fraction of the time.
    Rendering is what makes a rollout slow and needs a GPU, and none of these
    tests are about the video.

    `--no-lerobot` unless a test is about the dataset: writing one costs a
    parquet encode per episode and none of the manifest tests read it.
    """
    return runner.invoke(
        app, ["rollout", "--env", "iss", "--no-render", "--no-lerobot", *args]
    )


def ports_flown(tmp_path):
    """The per-episode port column of a manifest, in the order written."""
    manifest = json.loads((tmp_path / "rollout.json").read_text())
    return [episode["port"] for episode in manifest["episodes"]]


def test_rollout_writes_a_manifest(tmp_path):
    result = run("--out", str(tmp_path), "--policy", "dock",
                 "--port", "harmony_fwd_pma2", "--episodes", "2", "--steps", "600")
    assert result.exit_code == 0, result.output
    manifest = json.loads((tmp_path / "rollout.json").read_text())
    assert len(manifest["episodes"]) == 2
    assert manifest["policy"] == "dock"
    assert manifest["port"] == "harmony_fwd_pma2"
    for episode in manifest["episodes"]:
        assert set(episode) >= {"seed", "docked", "steps", "position_error_m"}


def test_rollout_writes_the_as_run_env_config(tmp_path):
    result = run("--out", str(tmp_path), "--policy", "dock",
                 "--port", "harmony_fwd_pma2", "--episodes", "1", "--steps", "600")
    assert result.exit_code == 0, result.output
    assert (tmp_path / "env_config.toml").exists()


def test_require_dock_keeps_only_successful_episodes(tmp_path):
    # The full horizon, because this is the one test that needs episodes to
    # actually reach the gate. Do not shorten it to speed the file up -- a
    # --require-dock that silently keeps a non-docked episode is the failure
    # this whole command exists to prevent.
    result = run("--out", str(tmp_path), "--policy", "dock",
                 "--port", "harmony_fwd_pma2", "--episodes", "2",
                 "--require-dock", "--steps", "7200")
    assert result.exit_code == 0, result.output
    manifest = json.loads((tmp_path / "rollout.json").read_text())
    assert len(manifest["episodes"]) == 2
    assert all(e["docked"] for e in manifest["episodes"])


def test_without_require_dock_every_episode_is_kept(tmp_path):
    result = run("--out", str(tmp_path), "--policy", "random",
                 "--episodes", "3", "--steps", "200")
    assert result.exit_code == 0, result.output
    manifest = json.loads((tmp_path / "rollout.json").read_text())
    assert len(manifest["episodes"]) == 3


def test_exhausting_attempts_fails_with_the_counts(tmp_path):
    # A random policy essentially never docks, so the retry loop must give up
    # and say how hard it tried rather than spin.
    result = run("--out", str(tmp_path), "--policy", "random",
                 "--episodes", "1", "--require-dock", "--max-attempts", "4",
                 "--steps", "200")
    assert result.exit_code != 0
    assert "4" in result.output
    assert "docked" in result.output.lower()


def test_a_port_is_rejected_for_a_policy_that_ignores_it(tmp_path):
    result = run("--out", str(tmp_path), "--policy", "orbit",
                 "--port", "harmony_fwd_pma2", "--episodes", "1", "--steps", "200")
    assert result.exit_code != 0
    assert "orbit" in result.output


def test_union_is_not_an_offered_policy(tmp_path):
    result = run("--out", str(tmp_path), "--policy", "union",
                 "--episodes", "1", "--steps", "200")
    assert result.exit_code != 0


def test_frame_stride_is_rejected_without_rendering(tmp_path):
    result = run("--out", str(tmp_path), "--policy", "random",
                 "--episodes", "1", "--steps", "200", "--frame-stride", "10")
    assert result.exit_code != 0


def test_env_dock_ports_the_single_target_does_not_share_is_a_usage_error(tmp_path):
    # A rollout flies the policy's port, so an env config naming a port set
    # beside a single --port would silently roll one target and record it as
    # if the set had been honoured -- the same disagreement `generate` refuses.
    cfg_path = tmp_path / "env.yaml"
    ISSConfig(dock=DockConfig(ports=("zvezda_aft", "poisk_zenith"))).to_yaml(cfg_path)
    result = run("--out", str(tmp_path / "run"), "--env-config", str(cfg_path),
                 "--policy", "dock", "--port", "harmony_fwd_pma2",
                 "--episodes", "1", "--steps", "200")
    assert result.exit_code != 0
    assert "dock ports disagree" in result.output
    assert "zvezda_aft" in result.output and "harmony_fwd_pma2" in result.output


def test_an_unknown_port_is_rejected(tmp_path):
    result = run("--out", str(tmp_path), "--policy", "dock",
                 "--port", "not_a_port", "--episodes", "1", "--steps", "200")
    assert result.exit_code != 0


def test_a_port_list_splits_the_episode_budget_evenly(tmp_path):
    result = run("--out", str(tmp_path), "--policy", "dock",
                 "--port", "harmony_fwd_pma2,zvezda_aft", "--episodes", "4",
                 "--steps", "200")
    assert result.exit_code == 0, result.output
    assert ports_flown(tmp_path) == [
        "harmony_fwd_pma2", "harmony_fwd_pma2", "zvezda_aft", "zvezda_aft"
    ]


def test_an_uneven_budget_gives_the_remainder_to_the_earlier_ports(tmp_path):
    # 5 across 3 is 2/2/1, not 1/1/3 and not a refusal: a sweep asks for a
    # round total and wants it spread as evenly as the total allows.
    result = run("--out", str(tmp_path), "--policy", "dock",
                 "--port", "harmony_fwd_pma2,zvezda_aft,rassvet_nadir",
                 "--episodes", "5", "--steps", "200")
    assert result.exit_code == 0, result.output
    flown = ports_flown(tmp_path)
    assert [flown.count(p) for p in
            ("harmony_fwd_pma2", "zvezda_aft", "rassvet_nadir")] == [2, 2, 1]


def test_all_expands_to_every_port(tmp_path):
    from owm_envs.envs.common.docking_ports import PORT_NAMES

    result = run("--out", str(tmp_path), "--policy", "dock", "--port", "all",
                 "--episodes", str(len(PORT_NAMES)), "--steps", "200")
    assert result.exit_code == 0, result.output
    assert ports_flown(tmp_path) == list(PORT_NAMES)


def test_fewer_episodes_than_ports_is_a_usage_error(tmp_path):
    # Silently dropping ports would report an eight-port sweep that flew four.
    result = run("--out", str(tmp_path), "--policy", "dock", "--port", "all",
                 "--episodes", "4", "--steps", "200")
    assert result.exit_code != 0
    assert "8" in result.output and "4" in result.output


def test_the_manifest_records_the_resolved_port_set(tmp_path):
    result = run("--out", str(tmp_path), "--policy", "dock",
                 "--port", "zvezda_aft,poisk_zenith", "--episodes", "2",
                 "--steps", "200")
    assert result.exit_code == 0, result.output
    manifest = json.loads((tmp_path / "rollout.json").read_text())
    # `port` stays what was asked for, so a single-port manifest reads exactly
    # as it always has; `ports` is what that expanded to.
    assert manifest["port"] == "zvezda_aft,poisk_zenith"
    assert manifest["ports"] == ["zvezda_aft", "poisk_zenith"]


def test_a_single_port_still_records_itself_per_episode(tmp_path):
    result = run("--out", str(tmp_path), "--policy", "dock",
                 "--port", "zvezda_aft", "--episodes", "2", "--steps", "200")
    assert result.exit_code == 0, result.output
    manifest = json.loads((tmp_path / "rollout.json").read_text())
    assert manifest["port"] == "zvezda_aft"
    assert manifest["ports"] == ["zvezda_aft"]
    assert ports_flown(tmp_path) == ["zvezda_aft", "zvezda_aft"]


def test_a_portless_rollout_records_no_port(tmp_path):
    result = run("--out", str(tmp_path), "--policy", "random",
                 "--episodes", "2", "--steps", "200")
    assert result.exit_code == 0, result.output
    manifest = json.loads((tmp_path / "rollout.json").read_text())
    assert manifest["ports"] == []
    assert ports_flown(tmp_path) == [None, None]


def test_an_unknown_port_inside_a_list_is_rejected(tmp_path):
    result = run("--out", str(tmp_path), "--policy", "dock",
                 "--port", "zvezda_aft,not_a_port", "--episodes", "2",
                 "--steps", "200")
    assert result.exit_code != 0
    assert "not_a_port" in result.output


@pytest.mark.parametrize("spec", [",", "+", "zvezda_aft,", ",zvezda_aft",
                                  "zvezda_aft,,poisk_zenith"])
def test_a_malformed_port_list_is_rejected(tmp_path, spec):
    # "," would otherwise fly the env config's own dock pose while the manifest
    # recorded --port as given: a run claiming a port set it never flew.
    result = run("--out", str(tmp_path), "--policy", "dock", "--port", spec,
                 "--episodes", "1", "--steps", "200")
    assert result.exit_code != 0
    assert "empty entry" in result.output


def test_a_repeated_port_is_rejected(tmp_path):
    # Otherwise the quota split would silently give that port a double share.
    result = run("--out", str(tmp_path), "--policy", "dock",
                 "--port", "zvezda_aft,zvezda_aft", "--episodes", "2",
                 "--steps", "200")
    assert result.exit_code != 0
    assert "zvezda_aft" in result.output


def run_with_dataset(*args):
    """`rollout` with the LeRobot split left on, which is the default."""
    return runner.invoke(app, ["rollout", "--env", "iss", "--no-render", *args])


def test_a_lerobot_split_is_written_by_default(tmp_path):
    result = run_with_dataset("--out", str(tmp_path), "--policy", "dock",
                              "--port", "zvezda_aft,poisk_zenith",
                              "--episodes", "2", "--steps", "200")
    assert result.exit_code == 0, result.output
    info = json.loads((tmp_path / "rollout" / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2


def test_the_split_holds_the_same_episodes_the_manifest_lists(tmp_path):
    # The manifest is what a reader uses to pick an episode out of the
    # dataset, so a split whose lengths disagree with it is unusable however
    # well-formed it is.
    result = run_with_dataset("--out", str(tmp_path), "--policy", "dock",
                              "--port", "zvezda_aft,poisk_zenith",
                              "--episodes", "2", "--steps", "200")
    assert result.exit_code == 0, result.output
    manifest = json.loads((tmp_path / "rollout.json").read_text())
    info = json.loads((tmp_path / "rollout" / "meta" / "info.json").read_text())
    # The manifest counts transitions and the dataset counts the rows they run
    # between, so an episode of N steps is N+1 frames.
    assert info["total_frames"] == sum(e["steps"] + 1 for e in manifest["episodes"])


def test_render_without_lerobot_stays_allowed(tmp_path, monkeypatch):
    """`generate` refuses --render --no-lerobot; rollout must not.

    There the clips are a tee off the dataset writer, so with no writer to
    pull them every rendered frame is thrown away. Here the clips are an
    output in their own right and the command drains the stream itself, which
    is the whole of what rollout did before it wrote datasets at all. Pinned
    because the two commands now share the flag and reading across from
    generate's guard would silently delete that behaviour.
    """
    pytest.importorskip("lerobot", reason="--render requires the datasets extra")

    def stop(index):
        raise RuntimeError("gpu probed")

    monkeypatch.setattr("owm_envs.render.device.select_gpu", stop)
    result = runner.invoke(app, [
        "rollout", "--env", "iss", "--out", str(tmp_path), "--policy", "dock",
        "--port", "zvezda_aft", "--episodes", "1", "--steps", "200",
        "--render", "--no-lerobot",
    ])
    # Reaching the GPU probe is what says the flag combination was accepted:
    # a refusal is a usage error raised well before it.
    assert isinstance(result.exception, RuntimeError), result.output


def test_a_failed_dataset_write_leaves_no_manifest(tmp_path, monkeypatch):
    """rollout.json is the completion marker, so a failure must not write one.

    Nothing lerobot writes can serve as that marker: it creates meta/info.json
    with the dataset, updates the count as episodes are saved, and flushes
    meta/episodes every ten of them. So a reader that wants to know whether a
    rollout finished has only this file to go on, and it must mean it.
    """
    from owm_envs.datasets import lerobot_writer as writer

    def boom(*args, **kwargs):
        raise RuntimeError("writer exploded")

    monkeypatch.setattr(writer, "write_lerobot_split", boom)
    result = run_with_dataset("--out", str(tmp_path), "--policy", "dock",
                              "--port", "zvezda_aft", "--episodes", "1",
                              "--steps", "200")
    assert result.exit_code != 0
    assert not (tmp_path / "rollout.json").exists()
    assert not (tmp_path / "env_config.toml").exists()


def test_an_out_holding_a_split_is_refused_before_anything_is_overwritten(tmp_path):
    # The writer refuses an existing split, but only once the rollout is spent
    # and the manifest already replaced -- which leaves rollout.json describing
    # episodes the surviving split does not hold. The refusal has to come first.
    (tmp_path / "rollout").mkdir()
    (tmp_path / "rollout.json").write_text('{"sentinel": true}\n')
    result = run_with_dataset("--out", str(tmp_path), "--policy", "dock",
                              "--port", "zvezda_aft", "--episodes", "1",
                              "--steps", "200")
    assert result.exit_code != 0
    assert json.loads((tmp_path / "rollout.json").read_text()) == {"sentinel": True}


def test_no_lerobot_may_reuse_an_out_that_holds_a_split(tmp_path):
    # With no split to write there is nothing to collide with, and replacing
    # an earlier run's clips and manifest is what a plain re-roll has always
    # done.
    (tmp_path / "rollout").mkdir()
    result = run("--out", str(tmp_path), "--policy", "dock", "--port",
                 "zvezda_aft", "--episodes", "1", "--steps", "200")
    assert result.exit_code == 0, result.output


def test_no_lerobot_writes_no_split(tmp_path):
    result = run_with_dataset("--out", str(tmp_path), "--no-lerobot",
                              "--policy", "dock", "--port", "zvezda_aft",
                              "--episodes", "1", "--steps", "200")
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "rollout").exists()
    assert (tmp_path / "rollout.json").exists()


def test_every_episode_is_rolled_under_its_own_seed(tmp_path):
    # Ports share one advancing seed rather than each restarting at --seed:
    # restarting would fly every port from the same initial conditions.
    result = run("--out", str(tmp_path), "--policy", "dock",
                 "--port", "harmony_fwd_pma2,zvezda_aft", "--episodes", "2",
                 "--steps", "200")
    assert result.exit_code == 0, result.output
    manifest = json.loads((tmp_path / "rollout.json").read_text())
    seeds = [e["seed"] for e in manifest["episodes"]]
    assert len(set(seeds)) == len(seeds), seeds


def test_a_retried_sweep_keeps_every_episode_re_rollable(tmp_path):
    # The full horizon, because this needs real retries: quotas above one mean
    # several kept episodes can share an attempt, and (seed, wanted,
    # batch_index) is the only thing that picks one back out of it. A
    # duplicated pair would make two clips re-roll to the same episode.
    result = run("--out", str(tmp_path), "--policy", "dock",
                 "--port", "harmony_fwd_pma2,zvezda_aft", "--episodes", "4",
                 "--require-dock", "--steps", "7200")
    assert result.exit_code == 0, result.output
    manifest = json.loads((tmp_path / "rollout.json").read_text())
    rows = manifest["episodes"]
    assert all(e["docked"] for e in rows)
    assert [e["port"] for e in rows].count("harmony_fwd_pma2") == 2
    assert [e["port"] for e in rows].count("zvezda_aft") == 2
    keys = [(e["seed"], e["wanted"], e["batch_index"]) for e in rows]
    assert len(set(keys)) == len(keys), keys
    # An attempt rolls `wanted` lanes, so a batch_index outside it names a lane
    # the re-roll would never produce.
    assert all(0 <= e["batch_index"] < e["wanted"] for e in rows)
