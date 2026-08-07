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

Which ENVIRONMENT produced it is part of that: `dataset_card.json` records
the name, and it selects the registry entry whose config class parses the
as-run config, whose `StateLayout` generates the state and truth-channel
documentation, and whose `card_summary` names the equations of motion. So
the two things a card would otherwise get wrong for a non-iss run -- the
width and meaning of every state element, and what the chaser was flying
under -- come from the same place the data did.

The whole run directory goes up as a single repo -- every split, plus the
as-run configs and normalization statistics -- so a dataset is one thing to
browse, and one split loads back with
`LeRobotDataset("iss/train", root="<downloaded-repo>/train")`.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml
from pydantic import ValidationError

from ..envs import ENV_REGISTRY, EnvSpec
from ..envs.common.config import BaseTaskConfig
from ..envs.common.layout import StateLayout
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

# One phrase per segment a StateLayout can declare, with its width filled in
# from the layout's own slice. Which of these appear, and in what order, is
# the layout's business -- a layout carrying a segment with no entry here
# fails loudly at card time rather than quietly publishing a state
# description that omits it.
_SEGMENT_DOC = {
    # Everything the epoch entry says beyond the width sits INSIDE the width
    # parenthesis: these phrases are comma-joined into a list of segments, and
    # a trailing clause on the first one reads as another list item.
    "epoch": "epoch as [Julian day, seconds of day] ({width}; UTC, the day "
             "number exact and the float32 seconds-of-day quantized to its own "
             "ulp, which coarsens through the day from well under 0.5 ms to "
             "7.8 ms -- a 50 ms step spans at least 6.4 of those ulps, so "
             "consecutive frames stay distinct)",
    "chief": "chief ECI position and velocity ({width}, m, m/s)",
    "pos": "position ({width}, m)",
    "vel": "velocity ({width}, m/s)",
    "omega": "body rate ({width}, rad/s)",
}

# What `observation_vector` holds under each `observation.mode` an env's
# `make_observe` can select (`envs/iss_numerical/observe.py`). The state
# layout describes `state_vector` and is NOT this row: a mode can narrow it,
# reorder it, and report it in another frame under another attitude
# convention. Keyed by mode name -- a mode with no entry here fails loudly at
# card time rather than publishing another mode's layout as this one's.
_OBS_MODE_DOC = {
    "absolute": "epoch as [Julian day, seconds of day] (2), the chief's ECI position "
                "and velocity (6, m, m/s), the chaser's own ECI position and velocity "
                "(6, m, m/s), the chaser's attitude quaternion (4, w-first, body to "
                "ECI/inertial) and its body rate with respect to ECI (3, rad/s) -- "
                "this mode alone reports the raw state's own columns, so it is the "
                "one mode under which this row and `state_vector` line up",
    "chaser_absolute": "epoch as [Julian day, seconds of day] (2), the CHASER's ECI "
                       "position and velocity (6, m, m/s), the chief measured from the "
                       "chaser -- the chief's ECI position and velocity minus the "
                       "chaser's, still in ECI axes (6, m, m/s) -- the chaser's "
                       "attitude quaternion (4, w-first, body to ECI/inertial) and its "
                       "body rate with respect to ECI (3, rad/s)",
    "chief_absolute": "epoch as [Julian day, seconds of day] (2), the chief's ECI "
                      "position and velocity (6, m, m/s), the chaser's "
                      "station-relative position and velocity in the ROTATING world "
                      "frame (6, m, m/s) -- the frame's own motion at that offset "
                      "already removed, so the velocity is not the inertial difference "
                      "of the two ECI velocities the block beside it would suggest -- "
                      "the chaser's attitude quaternion (4, w-first, body to "
                      "ECI/inertial) and its body rate with respect to ECI (3, rad/s)",
    "relative": "epoch as [Julian day, seconds of day] (2) and the 13-element "
                "canonical relative view: station-relative position (3, m) and "
                "velocity (3, m/s) in the rotating world frame, attitude quaternion "
                "(4, w-first, body to world) and body rate relative to that frame "
                "(3, rad/s). Nothing about the station's own orbit, and no inertial "
                "quantity anywhere in the row -- element for element the layout "
                "`iss-hcw` carries in state",
}

# The quaternion is the one segment whose CONVENTION, not just its width,
# differs across layouts, so it has no single entry in _SEGMENT_DOC: iss and
# iss-hcw's raw slice already is the body-to-world q_bw the task layer
# consumes, while iss-numerical's is q_bi, body to ECI/inertial, which only
# becomes q_bw through `relative_view`. Keyed by the segment's own first
# label (`layout.labels[layout.quat.start]`), which names which convention a
# layout uses -- a layout with a label this doesn't recognise fails loudly at
# card time rather than publishing a guessed-at attitude convention.
_QUAT_DOC_BY_LABEL = {
    "q_w": "attitude quaternion ({width}, w-first, body to world)",
    "q_bi_w": "attitude quaternion ({width}, w-first, body to ECI/inertial)",
}

# Storage order is read off the slices, so this only has to name the fields
# StateLayout can declare, not the order any env puts them in.
_SEGMENT_FIELDS = ("epoch", "chief", "pos", "vel", "quat", "omega")

# The four that make up the canonical relative view, which is what the task
# layer reads and what the sensor model draws over.
_VIEW_FIELDS = ("pos", "vel", "quat", "omega")

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


def _segments(layout: StateLayout) -> list[tuple[str, slice]]:
    """The layout's declared segments, in the order the state stores them."""
    declared = [
        (field, getattr(layout, field))
        for field in _SEGMENT_FIELDS
        if getattr(layout, field) is not None
    ]
    return sorted(declared, key=lambda item: item[1].start)


def _segment_phrase(field: str, sl: slice, layout: StateLayout) -> str:
    """The `_SEGMENT_DOC`/`_QUAT_DOC_BY_LABEL` phrase for one of `layout`'s
    segments, `quat` picked by convention rather than a fixed phrase -- see
    `_QUAT_DOC_BY_LABEL`."""
    if field != "quat":
        return _SEGMENT_DOC[field].format(width=sl.stop - sl.start)
    label = layout.labels[sl.start]
    try:
        return _QUAT_DOC_BY_LABEL[label].format(width=sl.stop - sl.start)
    except KeyError:
        raise KeyError(
            f"no card phrasing registered for quaternion label {label!r}; add one to "
            "_QUAT_DOC_BY_LABEL rather than describing this layout's attitude "
            "convention with another one's phrase"
        ) from None


def _state_doc(layout: StateLayout) -> str:
    """What the state vector holds, element by element, for THIS env.

    Generated from the layout rather than written per env: the widths and the
    order are the ones the backend actually integrates, so a card cannot
    describe a state its dataset does not have. `iss` reproduces the sentence
    this was before it was generated.
    """
    phrases = [_segment_phrase(field, sl, layout) for field, sl in _segments(layout)]
    body = ", ".join(phrases[:-1]) + " and " + phrases[-1]
    # "all station-relative" is only true when the state is the relative view
    # and nothing else. An epoch prefix is an absolute time, so a layout
    # carrying one has to say which elements the qualifier covers -- named
    # rather than positional ("the last four"), since nothing constrains a
    # layout to put its extra segments before the view.
    if len(phrases) == len(_VIEW_FIELDS):
        return body + ", all station-relative"
    if layout.chief is not None:
        # A layout carrying a chief block (iss-numerical) stores the
        # chaser's own ABSOLUTE ECI position, velocity, attitude and body
        # rate -- unlike iss-hcw, whose pos/vel/quat/omega slices already are
        # the relative view, calling these station-relative would be false.
        # The task layer's relative view is derived from this row and the
        # chief block together, not read off either alone.
        return (
            body
            + " -- position, velocity, attitude and body rate are the "
              "chaser's own ECI/inertial state, not station-relative; the "
              "station-relative view the task layer scores is derived from "
              "this row together with the chief block, not read off either "
              "alone"
        )
    return body + " -- position, velocity, attitude and body rate station-relative"


def _state_vector_doc(layout: StateLayout, observe_mode: str | None = None) -> str:
    """The truth column's meaning, and the obs-minus-truth identity that makes
    it worth writing, at THIS env's offsets.

    `observe_mode` is the env's `observation.mode` when its `EnvSpec` declares
    a non-identity `make_observe` (iss-numerical), None for every other env.
    The identity below assumes the recorded `observation_vector` IS this
    layout, column for column -- true only when `observe_mode` is None. A
    non-identity observe hook can narrow, reorder, or otherwise reshape the
    observation (`envs/iss_numerical/observe.py`'s four modes), so there is in
    general no fixed slice of `observation_vector` that lines up with this
    column's own offsets; `drivers/types.py`'s `TrajectoryBatch.true_state`
    docstring makes the same qualification, and this is that qualification
    carried into the card.

    Where `observe_mode` is None, the identity holds over the additive
    channels of the relative view, and the expression names their slices
    rather than the view as a whole. Both halves of that matter. The view is
    the whole state for `iss` but sits behind an epoch prefix for `iss-hcw`,
    so one set of offsets cannot serve both; and the four quaternion columns
    sit in the middle of the view while carrying a COMPOSED rotation
    (`envs/common/sensing.py`), so an expression spanning them would tell a
    reader to subtract four numbers whose difference is not a noise draw at
    all.
    """
    if observe_mode is not None:
        return (
            f"the TRUE {layout.state_dim}-element dynamics state at that frame, before "
            f"the sensor model touched it: {_state_doc(layout)}. That is this column's "
            "own layout and not, in general, the width or the column order of "
            "`observation_vector`, which reports this state reshaped into the "
            f"`{observe_mode}` frame `observation.mode` selects. There is therefore no "
            "fixed slice of `observation_vector` that lines up with this column's own "
            "additive channels for every mode, so the obs-minus-truth noise-draw "
            "recovery an identity-observe env's card describes here does not apply."
        )
    start, stop = layout.pos.start, layout.omega.stop
    # pos and vel are contiguous (StateLayout enforces it); omega sits on the
    # far side of the quaternion, so the additive part of the view is two
    # ranges, never one.
    translation = f"{layout.pos.start}:{layout.vel.stop}"
    rate = f"{layout.omega.start}:{layout.omega.stop}"
    identity = (
        f"`observation_vector[{translation}] - state_vector[{translation}]` and "
        f"`observation_vector[{rate}] - state_vector[{rate}]`"
    )
    outside = [
        _segment_phrase(field, sl, layout)
        for field, sl in _segments(layout)
        if sl.stop <= start or sl.start >= stop
    ]
    untouched = (
        (
            f". The sensor model only ever draws over position, velocity, "
            f"attitude and body rate, so the rest of the row -- "
            f"{', '.join(outside)} -- is identical in the two channels"
        )
        if outside
        else ""
    )
    return (
        "the TRUE dynamics state at that frame, before the sensor model touched it. "
        "Over the ADDITIVE channels -- position and velocity, and body rate -- "
        f"{identity} are exactly the realized noise draws, so the measurement model "
        "can be measured back off the data rather than trusted from the config. "
        "The four quaternion columns between them are not additive: attitude error "
        "is a small rotation composed onto the true attitude and then resolved onto "
        "the same hemisphere, so that draw is recovered as "
        "`quat_multiply(quat_conjugate(state_quat), observation_quat)` rather than "
        "as a difference"
        f"{untouched}"
    )


def _start_doc(env_cfg: BaseTaskConfig) -> str:
    """Where episodes begin, read from wherever this env disperses them."""
    low, high = env_cfg.start_shell()
    if low == high:
        return f"starts at {low:g} m"
    return f"starts between {low:g} m and {high:g} m"


def _noise_tag(cfg: BaseTaskConfig) -> str:
    for preset_name, preset in PRESETS.items():
        if cfg.sensor_noise == preset:
            return _NOISE_TAGS[preset_name]
    return "custom"


def dataset_name(env_cfg: BaseTaskConfig, env: str) -> str:
    """The published repo name for a run of `env` made under `env_cfg`.

    `env` is required rather than defaulting to iss: the default was the last
    remaining path to an `owm-iss-*` name for a run that was not iss at all.
    """
    goal = "goal" if env_cfg.observation.goal_error else "nogoal"
    dt_ms = round(env_cfg.dt * 1000)
    return f"owm-{env}-{_noise_tag(env_cfg)}-{goal}-dt{dt_ms}ms"


def _unknown_env_note(env: str, config_parsed: bool) -> str:
    """A card's own warning that it is describing an env this build lacks.

    Everything below the note -- the state description, the dynamics summary,
    the noise identity -- then comes from the iss fallback rather than from
    the env that generated the data, and a reader has to be told that before
    trusting any of it.

    `config_parsed` is False when even the as-run config would not go through
    the fallback's config class (`_fallback_config`), which widens that
    warning: the measurement model, the start shell and the timestep are then
    partly `iss` defaults too, so the note has to say so rather than let the
    values below read as the run's own.
    """
    if env in ENV_REGISTRY:
        return ""
    note = (
        f"\n\nThis dataset was generated by the `{env}` environment, which the "
        "version of owm-envs that wrote this card does not register, so the state "
        "layout and dynamics described below are `iss`'s and may not be this "
        "dataset's."
    )
    if not config_parsed:
        note += (
            " Its as-run `env_config.yaml` did not parse as an `iss` config either -- "
            "it carries fields this version does not declare -- so only the settings "
            "`iss` shares with it were read, and the measurement model, the start "
            "distances and the timestep quoted below fall back to `iss` defaults "
            "wherever it did not. The repo name was derived from the same partial "
            "reading."
        )
    return note


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _drop_key(payload: dict, loc: tuple) -> None:
    """Remove the (possibly nested) key at a Pydantic error `loc` from `payload`."""
    node: object = payload
    for step in loc[:-1]:
        if not isinstance(node, dict):
            return
        node = node.get(step)
    if isinstance(node, dict):
        node.pop(loc[-1], None)


def _fallback_config(path: Path, config_cls: type[BaseTaskConfig]) -> tuple[BaseTaskConfig, bool]:
    """`path` read through the iss fallback's config class, and whether it
    parsed cleanly.

    Only reached for an env this build cannot place. Configs forbid extra
    keys, so an unregistered env's as-run config generally does NOT parse as
    an `iss` one -- it carries its own sections -- and raising there would
    have crashed the publish path before the card ever got to warn about the
    unknown env. Instead the keys the class rejected are dropped and the rest
    is re-read, which is what keeps the settings iss shares with it (dt, the
    sensor model, the goal-error flag -- everything the repo name is built
    from) coming from the run rather than from a default. Anything that still
    will not validate falls back to defaults outright. Either way the second
    return value is False, and `_unknown_env_note` says on the card which of
    the values below a reader cannot trust.
    """
    try:
        return config_cls.from_yaml(path), True
    except ValidationError as rejected:
        payload = yaml.safe_load(path.read_text()) or {}
        pruned = copy.deepcopy(payload)
        for error in rejected.errors():
            if error["type"] == "extra_forbidden":
                _drop_key(pruned, error["loc"])
        try:
            return config_cls.model_validate(pruned), False
        except ValidationError:
            return config_cls(), False


def _run_env(run_dir: Path) -> tuple[EnvSpec, BaseTaskConfig, bool]:
    """The environment a run was generated with, and its as-run config.

    The name comes from the run's own `dataset_card.json`, which records it,
    and it decides everything the publish path cannot otherwise know: which
    config class the as-run `env_config.yaml` is parsed through, which env the
    repo is named after, and -- through the spec's layout and card summary --
    what the card says the data IS. All of it was hardcoded to iss, so an
    iss-hcw run could not be published at all: configs forbid extra keys, so
    parsing one as an ISSConfig fails on the reference orbit it carries.

    A run directory from before the card recorded an env, or one naming an
    env this build no longer registers, falls back to iss: every such run is
    an iss run, and reading it that way is better than refusing to publish it.

    A run with no card AT ALL is a different case and is refused here, which
    is also what makes push_preview and push_run agree: the card is where the
    fps and dt the README quotes come from, so tolerating its absence here
    only moved the failure to `_dataset_card`, after the preview had already
    told the caller which repo was about to be mirrored over.
    """
    card_path = run_dir / "dataset_card.json"
    if not card_path.exists():
        raise FileNotFoundError(
            f"{card_path} missing: it records the environment the run was generated "
            "with, its frame rate and its timestep, none of which the rest of the "
            "run carries. Regenerate the run with this version of owm-envs, or "
            "hand-write dataset_card.json with those fields."
        )
    env = _read_json(card_path).get("env", "iss")
    config_path = run_dir / "env_config.yaml"
    if env not in ENV_REGISTRY:
        # The repo is named after iss while the README's load example keeps
        # the card's own env name. That divergence is deliberate: the name has
        # to come from an env whose config class actually read the as-run
        # config, and the example has to name the split directory that is
        # really on disk, which the writer built from the card's env. The card
        # says outright that it is describing an env this build cannot place
        # -- including, through `_fallback_config`'s second return value,
        # whether that env's config was readable at all.
        spec = ENV_REGISTRY["iss"]
        cfg, config_parsed = _fallback_config(config_path, spec.config_cls)
        return spec, cfg, config_parsed
    spec = ENV_REGISTRY[env]
    return spec, spec.config_cls.from_yaml(config_path), True


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


def _observation_doc(
    env_cfg: BaseTaskConfig, layout: StateLayout, observe_mode: str | None = None
) -> str:
    """What the recorded observation row holds, column for column.

    For an identity-observe env this row IS the state layout, so the layout's
    own description is also this column's. `observe_mode` (iss-numerical's
    `observation.mode`) means it is not: the row is the state reshaped into
    that mode's frame, at that mode's width and column order, and describing
    it with the layout's phrases would assert a chief block, an inertial
    frame and a body->ECI quaternion that a mode like `relative` does not
    carry at all.
    """
    if observe_mode is None:
        doc = f"the MEASURED state recorded for that frame: {_state_doc(layout)}"
    else:
        try:
            mode_doc = _OBS_MODE_DOC[observe_mode]
        except KeyError:
            raise KeyError(
                f"no card phrasing registered for observation mode {observe_mode!r}; "
                "add one to _OBS_MODE_DOC rather than describing this mode's "
                "observation with another mode's layout"
            ) from None
        doc = (
            "the MEASURED state recorded for that frame, reshaped into the "
            f"`{observe_mode}` frame `observation.mode` selects: {mode_doc}"
        )
    if env_cfg.observation.goal_error:
        doc += (
            ", followed by the goal-error block -- position error (3), velocity error "
            "(3), attitude error as an axis-angle rotvec (3) and body-rate error (3) "
            "against the episode's goal"
        )
    # What the scripted policy actually consumed is a narrower thing than this
    # row, and under observe=state it is not this row at all -- both worth a
    # clause, since a reader modelling the behaviour policy has to know which
    # columns could have entered it. Under a mode-shaped observation it is not
    # this row even under the default observe=measurement: the policy reads the
    # env's own measured state, which the mode reshaped to produce this.
    if observe_mode is None:
        doc += (
            ". The scripted policy read the canonical relative view out of it rather "
            "than the whole row, so any epoch columns were never policy input; and a "
            "run generated with `--observe state` flew the policy on the true state "
            "instead of this one"
        )
    else:
        doc += (
            ". The scripted policy did not read this row at all: it flew on the "
            "canonical relative view derived from the MEASURED raw state the env "
            "publishes alongside it, which is what this row was reshaped from -- so "
            "these columns are what was RECORDED, not what was consumed; and a run "
            "generated with `--observe state` flew it on the true state instead"
        )
    return doc


def _schema_table(features: dict, env_cfg: BaseTaskConfig, env_spec: EnvSpec) -> str:
    layout = env_spec.layout
    # None for an identity-observe env (iss, iss-hcw): the recorded
    # observation_vector IS this layout, column for column. iss-numerical's
    # EnvSpec declares a make_observe, and its mode -- read straight off the
    # as-run config, not guessed -- is what the truth column's doc names as
    # the frame observation_vector was reshaped into.
    observe_mode = getattr(env_cfg.observation, "mode", None) if env_spec.make_observe else None
    rows = []
    for name, feature in features.items():
        if name == "observation_vector":
            doc = _observation_doc(env_cfg, layout, observe_mode)
        elif name == "state_vector":
            doc = _state_vector_doc(layout, observe_mode)
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


def _dataset_card(
    name: str,
    run_dir: Path,
    env_spec: EnvSpec,
    env_cfg: BaseTaskConfig,
    config_parsed: bool = True,
) -> str:
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
manoeuvring from {_start_doc(env_cfg)} out to a station docking port,
under {env_spec.card_summary} and against the station's 313-box collision hull. Generated with
[owm-envs](https://github.com/sisl/outofthisworldmodel-envs) for world-model training,
at {card["fps"]} Hz (dt = {card["dt"]} s).

{_NOISE_PROSE[_noise_tag(env_cfg)]}{_unknown_env_note(env, config_parsed)}

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
{_schema_table(info["features"], env_cfg, env_spec)}

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
    env_spec, env_cfg, _ = _run_env(run_dir)
    return name or dataset_name(env_cfg, env=env_spec.name), counts


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
    env_spec, env_cfg, config_parsed = _run_env(run_dir)

    import huggingface_hub

    api = huggingface_hub.HfApi()
    (run_dir / "README.md").write_text(
        _dataset_card(repo_name, run_dir, env_spec, env_cfg, config_parsed)
    )
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
