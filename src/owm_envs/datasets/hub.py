"""Publish a finished run directory to the HuggingFace Hub as one dataset repo.

The repo name is derived from the run's OWN as-run config rather than taken
from a flag, so it cannot drift from the data it describes:
`owm-{env}-{noise}-{goal}-dt{ms}ms`. The noise tag is the name of the PRESETS
entry the config's sensor_noise equals, or "custom" when it equals none of
them -- a hand-tuned sensor model therefore publishes under a name that says
so rather than borrowing a preset's.

The card is built the same way, out of the artifacts the run itself left:
`summary.json` for the split sizes, `dataset_card.json` for the per-split
policy and the code provenance, `env_config.yaml` for the measurement model,
`generation_config.yaml` for the port sets, and each split's own lerobot
`meta/info.json` for the column schema. Nothing is assumed about how the run
was invoked, so the card cannot describe a dataset other than the one being
uploaded.

The whole run directory goes up as a single repo -- every split, plus the
as-run configs and normalization statistics -- so a dataset is one thing to
browse, and one split loads back with
`LeRobotDataset("iss/train", root="<downloaded-repo>/train")`.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..envs import ENV_REGISTRY
from ..envs.common.config import BaseTaskConfig
from ..envs.common.policies import PolicyConfig
from ..envs.common.sensing import PRESETS
from .stats import SUMMARY_FILENAME, GenerationConfig

_NOISE_TAGS = {"off": "nonoise", "cooperative": "coop", "noncooperative": "noncoop"}

_NOISE_PROSE = {
    "nonoise": "The measurement model is disabled, so the recorded observations are "
               "noise-free -- they are the simulator's own state.",
    "coop": "Observations carry differential-GNSS-class relative navigation error: the "
            "accuracy available against a **cooperative** target, one that shares its "
            "own navigation solution.",
    "noncoop": "Observations carry chaser-derived (vision/LIDAR-class) relative "
               "navigation error: the accuracy available against a **non-cooperative** "
               "target, so position error grows with range and the velocity estimate is "
               "coarser.",
    "custom": "Observations carry the measurement error configured below.",
}

_POLICY_PROSE = {
    "random": "uniform random forces and torques within the actuator limits",
    "orbit": "a PD controller holding a circular station-relative orbit, its radius and "
             "rate drawn per episode",
    "dock": "a PD controller flying to the episode's assigned docking port",
}

_COLUMN_DOC = {
    "action": "commanded `[force (3, N), torque (3, N*m)]`",
    "reward": "per-frame reward; zero on an episode's last frame, whose action slot is a "
              "pad rather than a taken action",
    "is_last": "true only on an episode's final frame -- the frame to drop when forming "
               "`(observation, action, next observation)` transitions",
    "terminated": "per-episode: the episode ended on docking success or collision, "
                  "written onto every one of its frames",
    "truncated": "per-episode: the episode ran out of steps, written onto every one of "
                 "its frames",
    "policy_id": "per-episode: which member of a union policy drove it (0 random, "
                 "1 orbit, 2 dock); 0 throughout a run driven by a single policy",
    "dock_target": "per-episode: the `[position (3), quaternion (4)]` port pose it was "
                   "flying to; all-NaN when the driver could not supply one",
    # A rendered run writes all of these unless --render-views narrowed it;
    # only the first is the training view.
    "observation.images.fpv": "egocentric RGB video from the chaser, MP4-encoded, "
                              "aligned 1:1 with the vector frames",
    "observation.images.dragon_iso": "isometric RGB video following the chaser, for "
                                     "review rather than training; aligned 1:1 with "
                                     "the vector frames",
    "observation.images.dragon_top": "top-down RGB video following the chaser, for "
                                     "review rather than training; aligned 1:1 with "
                                     "the vector frames",
    "observation.images.iss_fpv": "RGB video from the station looking back at the "
                                  "chaser, for review rather than training; aligned "
                                  "1:1 with the vector frames",
    "observation.images.iss_iso": "isometric RGB video of the station, for review "
                                  "rather than training; aligned 1:1 with the vector "
                                  "frames",
    "observation.images.iss_top": "top-down RGB video of the station, for review "
                                  "rather than training; aligned 1:1 with the vector "
                                  "frames",
    "observation.images.composite": "all six camera views tiled into one frame per "
                                    "step (chaser and station, each first-person, "
                                    "isometric and top-down), for review rather than "
                                    "training; aligned 1:1 with the vector frames",
    "timestamp": "LeRobot bookkeeping: seconds since the start of the episode",
    "frame_index": "LeRobot bookkeeping: index within the episode, restarting at 0",
    "episode_index": "LeRobot bookkeeping: index of the episode within the split",
    "index": "LeRobot bookkeeping: index of the frame within the split",
    "task_index": "LeRobot bookkeeping: index of the task string",
}

_STATE_VECTOR_DOC = (
    "the TRUE dynamics state at that frame, before the sensor model touched it. "
    "`observation_vector[:13] - state_vector` is exactly the realized noise draw, so "
    "the measurement model can be measured back off the data rather than trusted "
    "from the config"
)

_ASSETS_SECTION = """## Assets and attribution

This dataset is a derived work of third-party 3D assets. The station geometry every
episode flies against -- the collision hull and the docking-port poses -- is derived
from `ISS_base.glb`, and a rendered video additionally shows the chaser, the
Moon, the starfield and the Earth:

| Asset | Source |
|---|---|
| ISS mesh | NASA 3D Resources / science.nasa.gov |
| Chaser mesh | Sketchfab -- "SpaceX Dragon 2 Exterior" |
| Starfield | NASA SVS #4851 |
| Moon | texture from NASA SVS #14959 (CGI Moon Kit); mesh geometry source not recorded |
| Earth maps | downsampled from equirectangular imagery collected from one of \
sketchfab.com, science.nasa.gov, cgtrader.com and maps.drsys.eu -- which one produced \
each map was not recorded |

**No licence is asserted for this dataset**, and none is declared in this card's
metadata: the upstream terms of the assets above were never recorded when they were
collected. Sketchfab and CGTrader items carry per-item terms, some requiring
attribution, and NASA imagery has its own usage guidelines. Resolve the terms of every
asset above before redistributing this data or a model trained on it."""

_STATE_DOC = (
    "position (3, m), velocity (3, m/s), attitude quaternion (4, w-first, body to "
    "world) and body rate (3, rad/s), all station-relative"
)


def _noise_tag(cfg: BaseTaskConfig) -> str:
    for preset_name, preset in PRESETS.items():
        if cfg.sensor_noise == preset:
            return _NOISE_TAGS[preset_name]
    return "custom"


def dataset_name(env_cfg: BaseTaskConfig, env: str = "iss") -> str:
    """The published repo name for a run made under `env_cfg`."""
    goal = "goal" if env_cfg.observation.goal_error else "nogoal"
    dt_ms = round(env_cfg.dt * 1000)
    return f"owm-{env}-{_noise_tag(env_cfg)}-{goal}-dt{dt_ms}ms"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _run_env(run_dir: Path) -> tuple[str, BaseTaskConfig]:
    """The environment a run was generated with, and its as-run config.

    The name comes from the run's own `dataset_card.json`, which records it,
    and it decides two things nothing else can supply: which config class the
    as-run `env_config.yaml` is parsed through, and which env the repo is
    named after. Both were hardcoded to iss, so an iss-hcw run could not be
    published at all -- configs forbid extra keys, so parsing one as an
    ISSConfig fails on the reference orbit it carries.

    A run directory from before the card recorded an env, or one naming an
    env this build no longer registers, falls back to iss: every such run is
    an iss run, and reading it that way is better than refusing to publish it.
    """
    card_path = run_dir / "dataset_card.json"
    env = _read_json(card_path).get("env", "iss") if card_path.exists() else "iss"
    if env not in ENV_REGISTRY:
        env = "iss"
    return env, ENV_REGISTRY[env].config_cls.from_yaml(run_dir / "env_config.yaml")


def _split_features(run_dir: Path, split: str) -> dict:
    """The lerobot info.json of `split`, which is the run's own column schema."""
    info = run_dir / split / "meta" / "info.json"
    if not info.exists():
        raise FileNotFoundError(
            f"{info} missing: this run holds no LeRobot split (generated with "
            "--no-lerobot?), so there is no data for the Hub viewer to show behind "
            "the repo's split configs"
        )
    return _read_json(info)


def _shape(feature: dict) -> str:
    dims = feature["shape"]
    return f"({dims[0]},)" if len(dims) == 1 else "(" + ", ".join(map(str, dims)) + ")"


def _observation_doc(env_cfg: BaseTaskConfig) -> str:
    doc = f"the MEASURED state the policy acted on: {_STATE_DOC}"
    if env_cfg.observation.goal_error:
        doc += (
            ", followed by the goal-error block -- position error (3), velocity error "
            "(3), attitude error as an axis-angle rotvec (3) and body-rate error (3) "
            "against the episode's goal"
        )
    return doc


def _schema_table(features: dict, env_cfg: BaseTaskConfig) -> str:
    rows = []
    for name, feature in features.items():
        if name == "observation_vector":
            doc = _observation_doc(env_cfg)
        elif name == "state_vector":
            doc = _STATE_VECTOR_DOC
        else:
            doc = _COLUMN_DOC.get(name, "")
        rows.append(f"| `{name}` | {feature['dtype']} | {_shape(feature)} | {doc} |")
    return "\n".join(rows)


def _splits_table(counts: dict, card_splits: dict) -> str:
    rows = []
    for name, count in counts.items():
        split = card_splits.get(name, {})
        rows.append(
            f"| `{name}` | {count['episodes']} | {count['transitions']} | "
            f"{count['hours']:.2f} | {count['terminated']} | {count['truncated']} | "
            f"{split.get('policy_type', '?')} | {split.get('seed', '?')} |"
        )
    return "\n".join(rows)


def _split_policies(run_dir: Path, splits: list[str]) -> dict[str, PolicyConfig]:
    """The policy each split actually ran, resolving the per-split override."""
    gen_cfg = GenerationConfig.from_yaml(run_dir / "generation_config.yaml")
    run_policy = PolicyConfig.from_yaml(run_dir / "policy_config.yaml")
    return {name: gen_cfg.splits[name].policy or run_policy for name in splits}


def _policy_section(policies: dict[str, PolicyConfig]) -> str:
    lines = []
    for name, policy in policies.items():
        if policy.type == "union":
            weights = " / ".join(f"{w:.2f}" for w in policy.union_weights)
            what = f"one of random / orbit / dock drawn per episode, weighted {weights}"
        else:
            what = _POLICY_PROSE[policy.type]
        ports = [port.name for port in policy.dock.ports]
        targets = (
            f", over {len(ports)} ports: {', '.join(f'`{p}`' for p in ports)}"
            if ports and policy.type in ("dock", "union")
            else ""
        )
        lines.append(f"- `{name}`: **{policy.type}** -- {what}{targets}")

    # Ports approached in some split but never in train are the held-out
    # generalisation axis, so they are worth naming rather than leaving to be
    # diffed off the lists above.
    train_ports = {p.name for p in policies["train"].dock.ports} if "train" in policies else set()
    held_out = [
        port.name
        for name, policy in policies.items()
        for port in policy.dock.ports
        if name != "train" and port.name not in train_ports
    ]
    if train_ports and held_out:
        unique = list(dict.fromkeys(held_out))
        lines.append("")
        lines.append(
            f"Held out of `train`: {', '.join(f'`{p}`' for p in unique)} -- approaches "
            "to those ports appear only in the other splits."
        )
    return "\n".join(lines)


def _provenance_line(provenance: dict) -> str:
    version = provenance.get("owm_envs_version") or "an unreleased build"
    commit = provenance.get("git_commit")
    if commit is None:
        return f"Generated with owm-envs {version}."
    dirty = " (with uncommitted local changes)" if provenance.get("git_dirty") else ""
    return f"Generated with owm-envs {version} at commit `{commit}`{dirty}."


def _dataset_card(name: str, run_dir: Path, env_cfg: BaseTaskConfig) -> str:
    """The README.md published with the run: frontmatter plus the run's own facts."""
    summary = _read_json(run_dir / SUMMARY_FILENAME)
    card = _read_json(run_dir / "dataset_card.json")
    counts = summary["counts"]
    splits = list(counts)
    info = _split_features(run_dir, splits[0])
    env = card.get("env", "iss")

    configs = "\n".join(
        f"  - config_name: {split}\n    data_files: {split}/data/**/*.parquet"
        for split in splits
    )
    render_flag = " --render" if info.get("video_path") else ""

    # No `license:` key: every dataset here is a derived work of assets whose
    # upstream terms were never recorded, so any value would be an assertion
    # nobody can back. _ASSETS_SECTION says so in the body rather than leaving
    # the omission to be noticed.
    return f"""---
pretty_name: {name}
tags:
- world-models
- robotics
- spacecraft
- lerobot
configs:
{configs}
---

# {name}

Docking approaches to the International Space Station: a 12-tonne Dragon-class chaser
manoeuvring from starts between 100 m and 500 m out to a station docking port, under rigid-body
free-flyer dynamics and against the station's 313-box collision hull. Generated with
[owm-envs](https://github.com/sisl/outofthisworldmodel-envs) for world-model training,
at {card["fps"]} Hz (dt = {card["dt"]} s).

{_NOISE_PROSE[_noise_tag(env_cfg)]}

Each split is a self-contained LeRobot dataset in its own directory:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

train = LeRobotDataset("{env}/{splits[0]}", root="<downloaded-repo>/{splits[0]}")
```

## Splits

| split | episodes | transitions | hours | terminated | truncated | policy | seed |
|---|---|---|---|---|---|---|---|
{_splits_table(counts, card.get("splits", {}))}

`terminated` counts episodes that ended on docking success or collision, `truncated`
those that ran out of steps. Normalization statistics (`normalization_stats.json`) are
computed on **train only** and apply to every split.

### Policies

{_policy_section(_split_policies(run_dir, splits))}

## Schema

Every frame of every episode carries:

| column | dtype | shape | meaning |
|---|---|---|---|
{_schema_table(info["features"], env_cfg)}

## Sensor noise

The measurement model the observations were drawn through, as this run recorded it:

```toml
[sensor_noise]
{env_cfg.sensor_noise.to_toml()}```

A scalar sigma is the RMS of the TOTAL error -- the norm of the 3-vector error, or the
total rotation angle for the attitude block -- and is applied isotropically as
sigma/sqrt(3) per axis. `sigma_pos_frac_of_range` follows the same convention against
the chaser's range, and combines with `sigma_pos_m` as independent variances.

## Reproducing

{_provenance_line(card.get("provenance", {}))} The as-run configuration is published
with the data, so the run is reproducible from this repo alone:

```bash
owm-envs generate --out run --env-config env_config.yaml \\
    --gen-config generation_config.yaml{render_flag}
```

{_ASSETS_SECTION}
"""


def _summary(run_dir: Path) -> dict:
    """The run's summary.json, which only a finished run has."""
    path = run_dir / SUMMARY_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- the run did not finish; refusing to push a partial run"
        )
    summary = _read_json(path)
    # `null`, a bare number and a string are all valid JSON, and each of them
    # would raise a TypeError out of the membership test or the lookup below.
    if not isinstance(summary, dict) or "counts" not in summary:
        raise ValueError(f"{path} holds no split counts; it is not a run summary")
    return summary


def push_preview(run_dir: str | Path, name: str | None = None) -> tuple[str, dict]:
    """What a push of `run_dir` would write: its repo name and its split counts.

    Derived the same way `push_run` derives them, and reading nothing but the
    run's own files: the upload mirrors the run onto the repo, so a caller has
    to be able to see which repo that is, and how much data would replace what
    is there, before any of it happens. The namespace is deliberately not
    resolved here -- that is a Hub round-trip, and a caller reporting an
    unreadable run must not have to tell a broken summary.json apart from a
    failed login.
    """
    run_dir = Path(run_dir)
    counts = _summary(run_dir)["counts"]
    env, env_cfg = _run_env(run_dir)
    return name or dataset_name(env_cfg, env=env), counts


def hub_namespace(namespace: str | None = None) -> str:
    """The Hub account a push writes into: `namespace`, or the token's own."""
    if namespace is not None:
        return namespace

    import huggingface_hub

    return huggingface_hub.HfApi().whoami()["name"]


def push_run(
    run_dir: str | Path,
    name: str | None = None,
    namespace: str | None = None,
    private: bool | None = None,
) -> str:
    """Upload `run_dir` as one Hub dataset repo. Returns the repo id.

    The card is written into `run_dir` before the upload starts, so it is part
    of that upload rather than a second commit against a repo that briefly had
    no description at all.

    `private` is three-state. True or False sets the repo's visibility; None
    (the default) makes a NEW repo public and leaves an existing one at
    whatever it already is, so a routine re-push can neither expose a repo
    that was deliberately made private nor hide one people are already using.
    """
    run_dir = Path(run_dir)
    repo_name, _ = push_preview(run_dir, name=name)
    repo_id = f"{hub_namespace(namespace)}/{repo_name}"
    _, env_cfg = _run_env(run_dir)

    import huggingface_hub

    api = huggingface_hub.HfApi()
    (run_dir / "README.md").write_text(_dataset_card(repo_name, run_dir, env_cfg))
    api.create_repo(repo_id, repo_type="dataset", private=bool(private), exist_ok=True)
    if private is not None:
        # create_repo ignores `private` for a repo that already exists, so
        # asking for one there and stopping would report success while
        # uploading into a repo of the other visibility. Applied BEFORE the
        # upload: a run pushed as private must never be publicly readable, not
        # even for the length of the transfer.
        api.update_repo_settings(repo_id, repo_type="dataset", private=private)
    # delete_patterns="*" makes the upload a MIRROR of run_dir, in the same
    # commit: every remote file this run does not have is removed. Without it a
    # regenerated run only overwrites the paths it happens to reuse, and the
    # previous run's leftover parquet still matches the split's
    # `data/**/*.parquet` viewer glob -- one repo serving two runs' frames
    # under one README that describes only the newer.
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(run_dir),
        delete_patterns="*",
    )
    return repo_id
