"""Hub repo naming and the dataset card, derived from a run's own artifacts.

Nothing here touches the network. Naming and the card are pure reads of a run
directory, and `push_run`'s Hub calls are exercised against a stand-in api
object -- what is being checked there is the call sequence, not the transfer.

The naming tests are load-bearing beyond this module: the six committed
configs/iss_*.toml must map onto exactly the six published dataset names, so a
config edit that changes what a dataset IS cannot leave it published under the
old name.
"""

import json
from pathlib import Path

import huggingface_hub
import pytest

from owm_envs.datasets.hub import _dataset_card, dataset_name, push_preview, push_run
from owm_envs.datasets.stats import GenerationConfig
from owm_envs.envs.iss.config import ISSConfig, ObservationConfig
from owm_envs.envs.iss.policies import PolicyConfig
from owm_envs.envs.iss.sensing import PRESETS, SensorNoiseConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

# The six variants and the dataset each one becomes. Spelled out literally
# rather than derived: deriving them from the same tag tables the naming code
# uses would make this test agree with any renaming, which is the one thing it
# exists to catch.
VARIANT_NAMES = {
    "iss_nonoise_nogoal": "owm-iss-nonoise-nogoal-dt50ms",
    "iss_nonoise_goal": "owm-iss-nonoise-goal-dt50ms",
    "iss_coop_nogoal": "owm-iss-coop-nogoal-dt50ms",
    "iss_coop_goal": "owm-iss-coop-goal-dt50ms",
    "iss_noncoop_nogoal": "owm-iss-noncoop-nogoal-dt50ms",
    "iss_noncoop_goal": "owm-iss-noncoop-goal-dt50ms",
}

COMMIT = "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c"

# The schema a run with truth and no video writes, as lerobot records it in
# each split's meta/info.json.
FEATURES = {
    "observation_vector": {"dtype": "float32", "shape": [25], "names": None},
    "action": {"dtype": "float32", "shape": [6], "names": None},
    "reward": {"dtype": "float32", "shape": [1, 1], "names": None},
    "is_last": {"dtype": "bool", "shape": [1], "names": None},
    "terminated": {"dtype": "bool", "shape": [1], "names": None},
    "truncated": {"dtype": "bool", "shape": [1], "names": None},
    "policy_id": {"dtype": "int64", "shape": [1, 1], "names": None},
    "dock_target": {"dtype": "float32", "shape": [1, 7], "names": None},
    "state_vector": {"dtype": "float32", "shape": [13], "names": None},
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}

COUNTS = {
    "train": {"episodes": 96, "transitions": 500_012, "terminated": 71,
              "truncated": 25, "seconds": 25000.6, "minutes": 416.677, "hours": 6.94461},
    "val": {"episodes": 11, "transitions": 50_004, "terminated": 8,
            "truncated": 3, "seconds": 2500.2, "minutes": 41.67, "hours": 0.6945},
}


def _write_run(
    tmp_path: Path,
    env_cfg: ISSConfig | None = None,
    *,
    features: dict | None = None,
    video: bool = False,
    finished: bool = True,
) -> Path:
    """A run directory holding exactly the artifacts RunMetadata.write leaves."""
    run = tmp_path / "run"
    run.mkdir()
    env_cfg = env_cfg or ISSConfig.from_toml(CONFIGS / "iss_noncoop_goal.toml")
    env_cfg.to_yaml(run / "env_config.yaml")
    GenerationConfig.from_yaml(CONFIGS / "generation_trial.yaml").to_yaml(
        run / "generation_config.yaml"
    )
    PolicyConfig().to_yaml(run / "policy_config.yaml")
    (run / "normalization_stats.json").write_text("{}")
    (run / "dataset_card.json").write_text(json.dumps({
        "env": "iss",
        "fps": round(1.0 / env_cfg.dt),
        "dt": env_cfg.dt,
        "splits": {
            name: {"episodes": c["episodes"], "transitions": c["transitions"],
                   "seed": 0 if name == "train" else 1, "max_steps": 7200,
                   "min_transitions": 500_000, "num_episodes_requested": None,
                   "policy_type": "union" if name == "train" else "dock",
                   "union_weights": [0.3, 0.35, 0.35]}
            for name, c in COUNTS.items()
        },
        "provenance": {"owm_envs_version": "0.1.0", "git_commit": COMMIT,
                       "git_dirty": False},
    }))
    for split in COUNTS:
        meta = run / split / "meta"
        meta.mkdir(parents=True)
        (meta / "info.json").write_text(json.dumps({
            "fps": round(1.0 / env_cfg.dt),
            "video_path": "videos/{video_key}/file-000.mp4" if video else None,
            "features": features if features is not None else FEATURES,
        }))
        data = run / split / "data" / "chunk-000"
        data.mkdir(parents=True)
        (data / "file-000.parquet").write_bytes(b"")
    if finished:
        (run / "summary.json").write_text(json.dumps({
            "dataset_root": str(run), "counts": COUNTS, "normalization_stats": {}
        }))
    return run


def _card(run: Path) -> str:
    env_cfg = ISSConfig.from_yaml(run / "env_config.yaml")
    return _dataset_card(dataset_name(env_cfg), run, env_cfg)


def test_names_cover_the_matrix():
    assert dataset_name(ISSConfig()) == "owm-iss-nonoise-nogoal-dt50ms"
    assert dataset_name(
        ISSConfig(sensor_noise=PRESETS["cooperative"],
                  observation=ObservationConfig(goal_error=True))
    ) == "owm-iss-coop-goal-dt50ms"
    assert dataset_name(
        ISSConfig(sensor_noise=PRESETS["noncooperative"])
    ) == "owm-iss-noncoop-nogoal-dt50ms"


def test_custom_noise_is_named_custom():
    cfg = ISSConfig(sensor_noise=SensorNoiseConfig(enabled=True, sigma_pos_m=9.9))
    assert dataset_name(cfg) == "owm-iss-custom-nogoal-dt50ms"


def test_dt_tag_scales():
    assert dataset_name(ISSConfig(dt=0.1)) == "owm-iss-nonoise-nogoal-dt100ms"


def test_the_env_name_is_a_component_not_a_prefix():
    assert dataset_name(ISSConfig(), env="lunar") == "owm-lunar-nonoise-nogoal-dt50ms"


@pytest.mark.parametrize("config_name, expected", sorted(VARIANT_NAMES.items()))
def test_each_committed_variant_names_its_published_dataset(config_name, expected):
    assert dataset_name(ISSConfig.from_toml(CONFIGS / f"{config_name}.toml")) == expected


def test_the_six_variants_get_six_distinct_names():
    # Two variants collapsing onto one name would publish one over the other.
    names = {dataset_name(ISSConfig.from_toml(CONFIGS / f"{c}.toml")) for c in VARIANT_NAMES}
    assert len(names) == len(VARIANT_NAMES)


def test_the_card_gives_the_viewer_one_config_per_split(tmp_path):
    card = _card(_write_run(tmp_path))
    for split in COUNTS:
        assert f"  - config_name: {split}\n    data_files: {split}/data/**/*.parquet" in card


def test_the_card_documents_the_truth_channel(tmp_path):
    card = _card(_write_run(tmp_path))
    assert "`state_vector`" in card
    # The obs-minus-truth identity is the whole reason the column is written.
    assert "observation_vector[:13] - state_vector" in card


def test_the_card_omits_the_truth_channel_when_the_run_has_none(tmp_path):
    features = {k: v for k, v in FEATURES.items() if k != "state_vector"}
    card = _card(_write_run(tmp_path, features=features))
    assert "state_vector" not in card


def test_the_card_reports_the_split_sizes(tmp_path):
    card = _card(_write_run(tmp_path))
    assert "| `train` | 96 | 500012 |" in card
    assert "| `val` | 11 | 50004 |" in card


def test_the_card_describes_the_policy_split(tmp_path):
    card = _card(_write_run(tmp_path))
    # The shipped recipe: a union-policy train split over five ports, a
    # dock-policy val split over all seven.
    assert "union" in card and "harmony_fwd_pma2" in card
    for held_out in ("harmony_zenith_cbm", "poisk_zenith"):
        assert held_out in card


def test_the_card_carries_the_sensor_noise_block(tmp_path):
    card = _card(_write_run(tmp_path))
    assert "[sensor_noise]" in card
    assert "sigma_pos_frac_of_range = 0.01" in card
    assert "sigma_vel_m_s = 0.03" in card


def test_a_noise_free_run_says_the_channels_agree(tmp_path):
    cfg = ISSConfig.from_toml(CONFIGS / "iss_nonoise_nogoal.toml")
    features = {**FEATURES, "observation_vector": {"dtype": "float32", "shape": [13],
                                                   "names": None}}
    card = _card(_write_run(tmp_path, cfg, features=features))
    assert "noise-free" in card
    # A run with no goal-error block must not claim one.
    assert "goal-error block" not in card


def test_the_card_asserts_no_licence(tmp_path):
    # The station geometry these episodes fly against, and the video rendered
    # from it, are derived from third-party assets whose upstream terms were
    # never recorded (see the repository README). A `license:` key in the
    # frontmatter would be an assertion nobody can back.
    frontmatter = _card(_write_run(tmp_path)).split("---")[1]
    assert "license" not in frontmatter


def test_the_card_names_the_asset_sources_and_their_unrecorded_terms(tmp_path):
    card = _card(_write_run(tmp_path))
    assert "## Assets and attribution" in card
    for source in ("ISS_stationary.glb", "Sketchfab", "CGTrader", "NASA SVS"):
        assert source in card, f"{source} unattributed"
    assert "never recorded" in card


def test_the_card_names_the_source_commit(tmp_path):
    assert COMMIT in _card(_write_run(tmp_path))


def test_the_card_gives_a_regeneration_command(tmp_path):
    card = _card(_write_run(tmp_path))
    assert "owm-envs generate" in card
    # The as-run configs ride along in the repo, so the command names them
    # rather than guessing which committed variant produced the run.
    assert "--config env_config.yaml" in card
    assert "--gen-config generation_config.yaml" in card
    assert "--render" not in card


def test_the_regeneration_command_renders_when_the_run_has_video(tmp_path):
    features = {**FEATURES, "observation.images.fpv": {
        "dtype": "video", "shape": [256, 256, 3],
        "names": ["height", "width", "channels"]}}
    card = _card(_write_run(tmp_path, features=features, video=True))
    assert "--render" in card
    assert "observation.images.fpv" in card


def test_a_run_with_no_lerobot_splits_is_not_publishable(tmp_path):
    run = _write_run(tmp_path)
    (run / "train" / "meta" / "info.json").unlink()
    with pytest.raises(FileNotFoundError, match="no LeRobot"):
        _card(run)


class _FakeApi:
    """Stands in for HfApi, recording the call sequence and what it could see."""

    def __init__(self):
        self.calls: list[str] = []
        self.create_kwargs: dict = {}
        self.settings_kwargs: dict = {}
        self.upload_kwargs: dict = {}
        self.card_at_upload: str | None = None

    def whoami(self) -> dict:
        self.calls.append("whoami")
        return {"name": "acct"}

    def create_repo(self, repo_id, **kwargs):
        self.calls.append("create_repo")
        self.create_kwargs = {"repo_id": repo_id, **kwargs}

    def update_repo_settings(self, repo_id, **kwargs):
        self.calls.append("update_repo_settings")
        self.settings_kwargs = {"repo_id": repo_id, **kwargs}

    def upload_folder(self, **kwargs):
        self.calls.append("upload_folder")
        self.upload_kwargs = kwargs
        readme = Path(kwargs["folder_path"]) / "README.md"
        self.card_at_upload = readme.read_text() if readme.exists() else None


@pytest.fixture
def api(monkeypatch) -> _FakeApi:
    fake = _FakeApi()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: fake)
    return fake


def test_push_creates_the_repo_then_uploads_a_carded_folder(tmp_path, api):
    run = _write_run(tmp_path)
    repo_id = push_run(run)

    assert repo_id == "acct/owm-iss-noncoop-goal-dt50ms"
    # The repo has to exist before anything is uploaded into it, and the card
    # has to be on disk before the upload or it is not in that upload.
    assert api.calls == ["whoami", "create_repo", "upload_folder"]
    assert api.create_kwargs == {"repo_id": repo_id, "repo_type": "dataset",
                                 "private": False, "exist_ok": True}
    assert api.upload_kwargs["repo_id"] == repo_id
    assert api.upload_kwargs["repo_type"] == "dataset"
    assert Path(api.upload_kwargs["folder_path"]) == run
    assert api.card_at_upload is not None
    assert repo_id.split("/")[1] in api.card_at_upload


def test_push_mirrors_the_run_rather_than_merging_into_what_is_there(tmp_path, api):
    # A regenerated run must REPLACE its repo. Merging would leave the previous
    # run's parquet behind, still matching the split's viewer glob, so one repo
    # would serve two runs' frames under a README describing only the newer.
    push_run(_write_run(tmp_path))
    assert api.upload_kwargs["delete_patterns"] == "*"


def test_push_honours_an_explicit_name_and_namespace(tmp_path, api):
    run = _write_run(tmp_path)
    assert push_run(run, name="trial", namespace="org") == "org/trial"
    # A given namespace must not cost a whoami round-trip.
    assert "whoami" not in api.calls


@pytest.mark.parametrize("private", [True, False])
def test_push_applies_an_asked_for_visibility_before_uploading(tmp_path, api, private):
    push_run(_write_run(tmp_path), private=private)
    # create_repo's `private` is ignored for a repo that already exists, so the
    # setting has to be applied on its own -- and applied before the data
    # lands, or a repo asked to be private is briefly world-readable.
    assert api.calls == ["whoami", "create_repo", "update_repo_settings", "upload_folder"]
    assert api.settings_kwargs == {"repo_id": "acct/owm-iss-noncoop-goal-dt50ms",
                                   "repo_type": "dataset", "private": private}


def test_push_leaves_an_existing_repos_visibility_alone_by_default(tmp_path, api):
    # Neither flag given: a new repo is public, and a repo someone deliberately
    # made private is not exposed by a routine re-push.
    push_run(_write_run(tmp_path))
    assert "update_repo_settings" not in api.calls
    assert api.create_kwargs["private"] is False


def test_push_preview_names_the_repo_and_the_counts_a_push_would_replace(tmp_path, api):
    # The caller has to be able to see what an upload would mirror over before
    # it happens; asking must itself change nothing.
    repo_id, counts = push_preview(_write_run(tmp_path))
    assert repo_id == "acct/owm-iss-noncoop-goal-dt50ms"
    assert counts["train"]["episodes"] == 96
    assert counts["val"]["transitions"] == 50_004
    assert api.calls == ["whoami"]


def test_push_preview_refuses_a_run_that_did_not_finish(tmp_path, api):
    with pytest.raises(FileNotFoundError, match="did not finish"):
        push_preview(_write_run(tmp_path, finished=False))


def test_push_refuses_a_run_that_did_not_finish(tmp_path, api):
    run = _write_run(tmp_path, finished=False)
    with pytest.raises(FileNotFoundError, match="did not finish"):
        push_run(run)
    assert api.calls == []
