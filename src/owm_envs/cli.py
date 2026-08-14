"""Command-line interface for dataset generation.

    owm-envs generate --out logs/run1 --split train:100000t:0 --split val:20000t:1
    owm-envs generate --out logs/run2 --split train:512:0 --noise noncooperative
    owm-envs push logs/run1
    owm-envs list

`--driver auto` selects the fused JAX path when the backend supports it and
falls back to the generic VectorEnv path otherwise, so the same command works
unchanged for a future non-JAX environment.
"""

from __future__ import annotations

import json
import sys
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import typer
import yaml
from pydantic import ValidationError

from .datasets.stats import GenerationConfig, SplitSpec, build_run_metadata
from .datasets.video import (
    COMPOSITE_VIEWS,
    OUTPUT_KEYS,
    VIEW_NAMES,
    keys_for_names,
    parse_view_names,
)
from .drivers.types import RolloutSpec, TrajectoryBatch, pack_episodes
from .envs import ENV_REGISTRY, EnvSpec
from .envs.common.config import BaseTaskConfig
from .envs.common.docking_ports import PORT_NAMES
from .envs.common.outcome import classify_batch
from .envs.common.policies import DockParams, PolicyConfig
from .envs.common.sensing import PRESETS
from .render.asset_hub import (
    EARTH_REPO_ID,
    EARTH_REPO_OWNER,
    download_asset,
    earth_dir,
    upload_assets,
)
from .render.earth import TEXTURE_KINDS, map_relpath, regenerate_map, source_relpath

app = typer.Typer(add_completion=False, help="Generate world-model training datasets.")


@app.command("list")
def list_envs() -> None:
    """List available environments and their observation/action shapes."""
    typer.echo("Available environments:")
    for name, spec in ENV_REGISTRY.items():
        venv = spec.make_vector_env(1, spec.config_cls())
        typer.echo(
            f"  {name}  ({spec.gym_id})  obs={venv.single_observation_space.shape}  "
            f"act={venv.single_action_space.shape}"
        )
        venv.close()


def _policy_config(policy: str, observe: str, ports: str) -> PolicyConfig:
    """Build a PolicyConfig, expanding a comma- or plus-joined port list.

    `ports` accepts "all", which DockParams expands to every entry in
    docking_ports.PORTS at validation time.
    """
    names = tuple(n for n in ports.replace("+", ",").split(",") if n)
    return PolicyConfig(type=policy, observe=observe, dock=DockParams(ports=names))


def _check_dock_ports_agree(
    cfg: BaseTaskConfig, policy_cfg: PolicyConfig, context: str
) -> None:
    """Refuse an env port set the run would not honour.

    Both `generate` and `rollout` resolve an episode's target from
    `policy.dock.ports`, never from `BaseTaskConfig.dock.ports` -- that second
    field is the one `ISSEnv` draws from. They are otherwise the same field,
    resolved by the same code, so a config that sets the env's and leaves the
    policy's empty reads as a multi-port run and would quietly write
    single-target data: the one outcome neither reading of the config asks for.
    The vector adapters draw per-lane ports themselves these days, but the
    drivers still fly the policy's -- `VectorEnvDriver` refuses the combination
    outright -- so the check stands.

    An empty `cfg.dock.ports` -- every config written before that field
    existed -- says nothing about ports and is left alone.
    """
    if not cfg.dock.ports or cfg.dock.ports == policy_cfg.dock.ports:
        return
    env_names = [port.name for port in cfg.dock.ports]
    policy_names = [port.name for port in policy_cfg.dock.ports]
    raise typer.BadParameter(
        f"{context}: dock ports disagree. The env config's dock.ports "
        f"names {env_names}, the policy's dock.ports names {policy_names}, and "
        f"the run flies the policy's -- so it would not record the "
        f"ports the env config asks for. Name the same ports on both (--dock-"
        f"ports or the split's PORTS field for generate, --port for rollout), "
        f"or drop dock.ports from the env config."
    )


def _parse_render_views(spec: str) -> list[str]:
    """`--render-views` -> the view names to record in the generation config.

    The parsing itself belongs to the datasets package, which owns the keys; a
    bad value is a usage error here rather than the ValueError it is there.
    """
    try:
        return list(parse_view_names(spec))
    except ValueError as exc:
        raise typer.BadParameter(f"--render-views: {exc}") from exc


def _parse_split_flags(
    values: list[str], steps: int, observe: str, policy: str, ports: str
) -> dict[str, SplitSpec]:
    """`NAME:COUNT[t]:SEED[:POLICY[:PORTS]]` flags -> split specs, all at `steps` max steps.

    COUNT is an episode count (`64`), or, with a trailing `t`, a minimum
    transition target (`100000t`) -- the split runs whole episodes until it
    has accumulated at least that many (state, action, next-state) pairs.

    A `:POLICY` suffix's PolicyConfig inherits `--observe` rather than the
    PolicyConfig default, so a per-split policy still respects the run-level
    observe flag. A further `:PORTS` suffix is a +-joined list of docking-port
    names, or "all", and overrides `--dock-ports` for that split -- which is
    how a held-out port is arranged: name the training ports on the train
    split and leave validation at the default "all".
    """
    splits: dict[str, SplitSpec] = {}
    for raw in values:
        parts = raw.split(":")
        if len(parts) not in (3, 4, 5) or not parts[0]:
            raise typer.BadParameter(
                f"--split expects NAME:COUNT[t]:SEED[:POLICY[:PORTS]] "
                f"(e.g. train:64:0, train:100000t:0, val:8:1:dock, "
                f"train:64:0:union:harmony_fwd_pma2+poisk_zenith, or val:8:1::all "
                f"to keep --policy and override only the ports), got '{raw}'"
            )
        name, count_text, seed_text = parts[0], parts[1], parts[2]
        transitions_mode = count_text.endswith("t")
        count_value_text = count_text[:-1] if transitions_mode else count_text
        try:
            count, seed = int(count_value_text), int(seed_text)
        except ValueError as exc:
            raise typer.BadParameter(
                f"--split '{raw}': COUNT and SEED must be integers "
                "(COUNT may have a trailing 't' for a transition target)"
            ) from exc
        if count < 1:
            raise typer.BadParameter(f"--split '{raw}': COUNT must be >= 1")
        if name in splits:
            raise typer.BadParameter(f"--split '{raw}': duplicate split name '{name}'")
        # An empty POLICY field only means something when a PORTS field
        # follows it (`train:4:0::all` keeps --policy and overrides the
        # ports). On its own it is a typo, not a way to spell "inherit":
        # dropping the trailing ':' already does that.
        has_ports = len(parts) == 5 and bool(parts[4])
        if len(parts) >= 4 and not parts[3] and not has_ports:
            raise typer.BadParameter(
                f"--split '{raw}': POLICY is empty; drop the trailing ':' to inherit "
                "--policy, or name a port set after it (e.g. train:4:0::all)"
            )
        # None means "inherit the run-level policy", which is what the caller
        # falls back to. Only a split that actually names a policy or a port
        # set gets its own config, so an unqualified split still records no
        # override rather than a copy of the run-level one.
        split_policy = None
        if len(parts) >= 4 and (parts[3] or has_ports):
            try:
                split_policy = _policy_config(
                    parts[3] or policy,
                    observe,
                    parts[4] if len(parts) == 5 else ports,
                )
            except Exception as exc:  # pydantic rejects unknown policies and ports
                raise typer.BadParameter(f"--split '{raw}': {exc}") from exc
        splits[name] = SplitSpec(
            num_episodes=None if transitions_mode else count,
            min_transitions=count if transitions_mode else None,
            max_steps=steps,
            seed=seed,
            policy=split_policy,
        )
    return splits


@app.command()
def generate(
    out: Path = typer.Option(..., help="Run directory to write."),
    env: Optional[str] = typer.Option(
        None,
        help="Environment to generate with: " + ", ".join(ENV_REGISTRY) + ". "
             "Default iss. Exclusive with --gen-config, whose env field governs."),
    policy: str = typer.Option("random", help="random | orbit | dock | union -- "
                               "run-level default; a split's :POLICY suffix overrides it."),
    split: Optional[list[str]] = typer.Option(
        None, "--split",
        help="Repeatable NAME:COUNT[t]:SEED[:POLICY[:PORTS]] "
             "(default: train:64:0 val:8:1::all); COUNT is an episode count, or a "
             "minimum transition target with a trailing 't' (e.g. 100000t); POLICY "
             "overrides --policy for that split and PORTS overrides --dock-ports "
             "(a comma- or +-joined list of port names, or 'all'). The default "
             "leaves validation on every port deliberately: training can hold ports out "
             "while validation still measures the held-out ones.",
    ),
    steps: Optional[int] = typer.Option(None, help="Max steps per episode (default 7200)."),
    num_envs: Optional[int] = typer.Option(None, help="Parallel lanes (default 8)."),
    driver: Optional[str] = typer.Option(None, help="auto | scan | vector (default auto)."),
    fps: Optional[int] = typer.Option(None, help="Frames per second recorded in the dataset. "
                                      "Defaults to the simulation rate, 1/dt."),
    gen_config: Optional[Path] = typer.Option(
        None, help="GenerationConfig YAML; exclusive with "
                   "--env/--split/--steps/--num-envs/--driver/--fps/--render-views. "
                   "configs/iss/gen/default.yaml is the shipped docking recipe: a union-policy "
                   "train split on five ports and a dock-policy val split on all "
                   "eight, so validation measures the held-out approaches (both zenith "
                   "corridors and Unity nadir)."),
    env_config: Optional[Path] = typer.Option(
        None, "--env-config",
        help="Environment config file to load, YAML or TOML by suffix. The shipped "
             "iss environments are under configs/<env>/env/: default.toml plus the six "
             "noise/goal-error variants the public datasets are generated from."),
    noise: Optional[str] = typer.Option(
        None, help="Sensor-noise preset: off | cooperative | noncooperative. "
                   "Overrides the --env-config file's sensor_noise."),
    goal_error: Optional[bool] = typer.Option(
        None, "--goal-error/--no-goal-error",
        help="Append the dock-goal error block to observations. "
             "Overrides the --env-config file's observation.goal_error; "
             "default is whatever the config says."),
    observe: str = typer.Option(
        "measurement", help="What scripted policies consume: state | measurement "
                            "(default: measurement, the noisy value the dataset records)."),
    dock_ports: str = typer.Option(
        "",
        help="Docking ports the dock and union policies may target, drawn uniformly "
             f"per episode: 'all' or a comma-joined subset of {', '.join(PORT_NAMES)}. "
             "Empty (the default) keeps the single pose in BaseTaskConfig.dock. Sets the "
             "default for every split; a :PORTS suffix on --split overrides it, which "
             "is how validation keeps the full set while training holds ports out. The "
             "built-in default splits already put validation on 'all'. Inert for the "
             "random and orbit policies.",
    ),
    lerobot: bool = typer.Option(True, "--lerobot/--no-lerobot", help="Write a LeRobot dataset."),
    render: bool = typer.Option(
        False,
        "--render/--no-render",
        help="Render an egocentric video feed (slow: ~0.1 s/frame; off by default).",
    ),
    render_views: Optional[str] = typer.Option(
        None,
        help="Which video features --render writes; exclusive with --gen-config, which "
             "carries its own. 'all' (the default) is every one of the "
             f"{len(COMPOSITE_VIEWS)} named cameras under its own key, plus the "
             f"composite tiling them into one frame -- {len(OUTPUT_KEYS)} video streams "
             f"per split. Or a comma-joined list of {', '.join(VIEW_NAMES)} to write "
             "fewer. The default costs six draws per frame rather than one, and stores "
             f"{len(OUTPUT_KEYS)} encoded streams rather than one, so pass "
             "'--render-views fpv' for a lean training run: that writes the egocentric "
             f"view alone, under {OUTPUT_KEYS[0]}, which is the key training reads. "
             "Any run that includes fpv also gets a per-episode copy of that clip "
             "under media/fpv/<split>/.",
    ),
    render_workers: int = typer.Option(
        1,
        help="Parallel render worker processes (episodes fan out across them). "
             "Each worker owns a renderer costing ~1.9 GiB of VRAM on the "
             "--gpu-index card.",
    ),
    gpu_index: Optional[int] = typer.Option(
        None,
        help="GPU this run uses, for the rollout as well as the renderer "
             "(default: wgpu's own choice, and JAX's own choice). Counts "
             "discrete GPUs only. Also settable via OWM_ENVS_GPU_INDEX.",
    ),
) -> None:
    """Roll out trajectories for every split and write a dataset run directory."""
    if render and not lerobot:
        raise typer.BadParameter(
            "--render has no effect with --no-lerobot: there is no writer to consume the "
            "rendered frames, so rendering would be pure wasted cost. Drop --render, or "
            "drop --no-lerobot so the frames are written."
        )

    if render_workers < 1:
        raise typer.BadParameter(f"--render-workers must be >= 1, got {render_workers}")

    # Checked before the GPU is touched, as is everything from here down to
    # the `if render:` block: a usage error must cost a usage error, not an
    # adapter probe -- which pins a device for the process -- followed by one.
    if render_views is not None:
        _parse_render_views(render_views)
    if gen_config is not None and any(
        v is not None for v in (env, split, steps, num_envs, driver, fps, render_views)
    ):
        raise typer.BadParameter(
            "--gen-config is exclusive with "
            "--env/--split/--steps/--num-envs/--driver/--fps/--render-views"
        )
    if env is not None and env not in ENV_REGISTRY:
        raise typer.BadParameter(
            f"unknown environment '{env}'; available: {', '.join(ENV_REGISTRY)}")

    if gen_config is not None:
        try:
            gen = GenerationConfig.from_yaml(gen_config)
        except ValidationError as exc:
            raise typer.BadParameter(str(exc)) from exc
        except (OSError, yaml.YAMLError) as exc:
            raise typer.BadParameter(
                f"cannot read --gen-config {gen_config}: {exc}"
            ) from exc
    else:
        resolved_steps = steps if steps is not None else 7200
        try:
            gen = GenerationConfig(
                env=env or "iss",
                splits=_parse_split_flags(
                    split or ["train:64:0", "val:8:1::all"], resolved_steps, observe, policy, dock_ports
                ),
                num_envs=num_envs if num_envs is not None else 8,
                fps=fps,
                driver=driver if driver is not None else "auto",
                render_views=(
                    _parse_render_views(render_views)
                    if render_views is not None
                    else list(VIEW_NAMES)
                ),
            )
        except ValidationError as exc:
            raise typer.BadParameter(str(exc)) from exc

    if gen.num_envs < 1:
        raise typer.BadParameter(f"num_envs must be >= 1, got {gen.num_envs}")

    # From the recipe, not the flag: --gen-config carries its own selection,
    # and the as-run copy of that recipe is what records which views a dataset
    # was built with.
    view_keys = keys_for_names(gen.render_views)

    env_name = gen.env
    if env_name not in ENV_REGISTRY:
        raise typer.BadParameter(
            f"unknown environment '{env_name}'; available: {', '.join(ENV_REGISTRY)}"
        )
    env_spec = ENV_REGISTRY[env_name]

    # Resolved from the recipe rather than the flag, so it also catches a
    # --gen-config naming a non-renderable env, and checked here rather than at
    # the flag checks above because that is the first point either source has
    # been read. Still before the GPU is touched and before any rollout.
    if render and not env_spec.renderable:
        renderable = ", ".join(name for name, spec in ENV_REGISTRY.items() if spec.renderable)
        raise typer.BadParameter(
            f"--render does not support the '{env_name}' environment yet, only "
            f"{renderable}. Every frame is posed through a render adapter, which "
            "is what reads an env's rows in that env's own element order, and "
            f"'{env_name}' has none registered (EnvSpec.make_render_adapter). "
            "Rendering it arrives with its adapter, which is what flips "
            "EnvSpec.renderable; until then, drop --render to generate the "
            "vector dataset."
        )

    # Everything from here to the `if render:` block reads the caller's own
    # arguments, or checks a dependency the install either has or does not, and
    # all of it comes before the GPU probe: neither a usage error nor a missing
    # extra may cost an adapter probe -- which pins a device for the whole
    # process -- on the way to being reported. Usage errors come first within
    # that, so a run that was never valid says so rather than blaming an
    # uninstalled extra.
    try:
        cfg = env_spec.config_cls.load(env_config) if env_config is not None else env_spec.config_cls()
    except (OSError, ValueError, yaml.YAMLError) as exc:
        # A missing file, a suffix load() does not dispatch on, unparseable
        # text, or a field the schema rejects -- all of them are the caller
        # naming the wrong file, not a bug to show a traceback for.
        raise typer.BadParameter(
            f"cannot read --env-config {env_config}: {exc}", param_hint="--env-config"
        ) from exc
    if noise is not None:
        if noise not in PRESETS:
            raise typer.BadParameter(
                f"unknown --noise preset '{noise}'; use one of: {', '.join(PRESETS)}"
            )
        cfg = cfg.model_copy(update={"sensor_noise": PRESETS[noise]})
    if goal_error is not None:
        # Copies the existing observation model rather than replacing it with
        # a bare `ObservationConfig` -- iss-numerical's is a subtype carrying
        # `mode`, which `_observation_space()` and `make_observe()` both read,
        # and a plain `ObservationConfig` has no such field.
        cfg = cfg.model_copy(
            update={"observation": cfg.observation.model_copy(update={"goal_error": goal_error})}
        )
    resolved_fps = _resolve_fps(gen.fps, cfg.dt)
    try:
        policy_cfg = _policy_config(policy, observe, dock_ports)
    except Exception as exc:  # pydantic rejects unknown policies and ports
        raise typer.BadParameter(f"invalid policy '{policy}': {exc}") from exc

    if lerobot:
        # An environment missing lerobot otherwise hits ModuleNotFoundError deep
        # inside write_lerobot_split. Check up front, before spending time on a
        # rollout, and say what a user who asked for a dataset needs to do to
        # actually get one, rather than quietly writing metadata only.
        try:
            import lerobot  # noqa: F401
        except ModuleNotFoundError as exc:
            raise typer.BadParameter(
                "lerobot is not installed; the datasets stack ships in the base "
                "install, so rebuild the environment with 'uv sync' -- or pass "
                "--no-lerobot"
            ) from exc

    if render:
        from .render.device import check_gpu_index, select_gpu

        try:
            if render_workers == 1:
                # pygfx pins one shared wgpu device per process the first time
                # a scene is built, so the adapter has to be chosen up front.
                # Only for the single-worker path, which renders in THIS
                # process.
                select_gpu(gpu_index)
            else:
                # With a pool every renderer lives in a worker, and this
                # process must not take a device the workers need -- so the
                # index is only bounds-checked here. That check still belongs
                # before the rollout: the workers do not start until an hour
                # of rollout is already spent.
                check_gpu_index(gpu_index)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--gpu-index") from exc

    # Drivers bake their policy in at construction, so a split with its own
    # policy needs its own driver instance.
    # Every split's ports are checked before any split generates. Checking a
    # split just before it runs is not enough: the splits are generated in
    # order, so a disagreement in the last one would be found only after every
    # earlier split had already paid for its full rollout.
    for name, spec in gen.splits.items():
        _check_dock_ports_agree(cfg, spec.policy or policy_cfg, f"split '{name}'")

    batches = {}
    for name, spec in gen.splits.items():
        split_policy = spec.policy or policy_cfg
        chosen = _resolve_driver(gen.driver, cfg, split_policy, gen.num_envs, env_spec)
        batch = chosen.driver.generate(
            RolloutSpec(
                num_episodes=spec.num_episodes,
                max_steps=spec.max_steps,
                seed=spec.seed,
                min_transitions=spec.min_transitions,
            )
        )
        target = (
            f">={spec.min_transitions} transitions"
            if spec.min_transitions is not None
            else f"{spec.num_episodes} episodes"
        )
        ports_note = (
            f" ports={len(split_policy.dock.ports)}" if split_policy.dock.ports else ""
        )
        typer.echo(f"[generate] {name}: driver={chosen.name} policy={split_policy.type}"
                   f"{ports_note} "
                   f"target={target}, got {batch.num_episodes} episodes, "
                   f"{batch.total_transitions} transitions")
        batches[name] = batch

    # Built now so a batch that cannot produce statistics fails here, before
    # any dataset is written, but flushed last: these files are what marks the
    # run complete, so a failure downstream must not leave them behind.
    metadata = build_run_metadata(
        cfg=cfg, policy_cfg=policy_cfg, gen_cfg=gen, batches=batches, fps=resolved_fps
    )

    if render:
        # Every renderer resolves these three itself, and a miss downloads and
        # bakes a full map from a multi-gigabyte source. Done once here, the
        # render workers each find a finished file: N concurrent bakes cannot
        # exhaust memory, and no two workers can settle on different tiers and
        # mix Earth resolutions within one dataset. Resolution touches no wgpu
        # device, so this process still takes none.
        from .render.earth import earth_texture_path

        for kind in ("color", "clouds", "bump"):
            earth_texture_path(kind, allow_download=True)

    for name, batch in batches.items():
        frames = None
        if render:
            from .datasets.video import iter_batch_frames, tee_episode_clips
            from .render.iss_scene import RenderConfig

            render_cfg = RenderConfig(**cfg.render) if cfg.render else RenderConfig()
            total = int(batch.lengths.sum())
            # Frame and worker counts only: per-frame cost moves with
            # resolution and scene, and extra workers scale sub-linearly, so
            # any duration printed here would be a prediction this code
            # cannot make. How many workers to spend is the operator's call.
            typer.echo(
                f"[render] {name}: {total} frames, "
                f"{len(view_keys)} video feature(s) "
                f"({', '.join(key.rsplit('.', 1)[-1] for key in view_keys)}), "
                f"{render_workers} worker(s)"
            )
            # Lazy: the writer pulls one episode's clips at a time. Rendering
            # a whole split first would need ~98 GB of RAM per feature at 500k
            # frames.
            frames = iter_batch_frames(
                batch, render_cfg, keys=view_keys,
                workers=render_workers, gpu_index=gpu_index,
                # The environment that produced the batch, not a built
                # adapter: a render worker is a spawned process and rebuilds
                # its own from these two.
                env_name=env_name, env_cfg=cfg,
            )
            # Tapped on the way past rather than rendered again: every view's
            # clips are also written per episode under media/<view>/<split>/,
            # which is the one-video-per-rollout shape that reviewers and the
            # training side's tooling expect. Auxiliary files, not dataset
            # features; see tee_episode_clips for what they cost.
            frames = tee_episode_clips(frames, out / "media", name, resolved_fps)
        if lerobot:
            from .datasets.lerobot_writer import write_lerobot_split

            write_lerobot_split(out / name, f"{env_name}/{name}", batch,
                                fps=resolved_fps, frames=frames)
            typer.echo(f"[generate] wrote LeRobot split to {out / name}")

    metadata.write(out)
    typer.echo(f"[done] {out}")


def _episode_row(batch: TrajectoryBatch, index: int) -> dict:
    """One episode of `batch`, sliced to its own length, for `pack_episodes`.

    Every array is cut at `batch.lengths[index]` rather than handed over whole:
    the rows kept by a retrying rollout come from several batches, each padded
    to ITS OWN longest episode, and a row carrying another batch's padding
    would be repacked with that padding inside its length -- zeros that read
    back as real timesteps rather than as pad.

    The four optional channels are included only when the source batch has
    them, which is exactly when `pack_episodes` is told to read them.
    """
    length = int(batch.lengths[index])
    row = {
        "obs": batch.observations[index, :length],
        "act": batch.actions[index, :length],
        "rew": batch.rewards[index, :length],
        "terminated": bool(batch.terminated[index]),
        "truncated": bool(batch.truncated[index]),
    }
    if batch.policy_ids is not None:
        row["policy_id"] = int(batch.policy_ids[index])
    if batch.dock_targets is not None:
        row["dock_target"] = batch.dock_targets[index]
    if batch.true_state is not None:
        row["true_state"] = batch.true_state[index, :length]
    if batch.terminal_events is not None:
        row["terminal_events"] = batch.terminal_events[index]
    return row


@app.command()
def rollout(
    out: Path = typer.Option(..., help="Directory to write the rollout into."),
    env: str = typer.Option("iss-numerical",
                            help="Environment: " + ", ".join(ENV_REGISTRY) + "."),
    env_config: Optional[Path] = typer.Option(
        None, "--env-config",
        help="Environment config file, YAML or TOML by suffix. Defaults to the "
             "environment's own defaults."),
    policy: str = typer.Option("dock", help="dock | orbit | random. 'union' is a "
                               "training mixture, not a behaviour worth filming."),
    port: str = typer.Option("", help="Docking port to fly to. Dock policy only; "
                             f"one of {', '.join(PORT_NAMES)}. Empty keeps the "
                             "single pose in the env config's dock section."),
    episodes: int = typer.Option(1, help="Episodes to keep."),
    seed: int = typer.Option(0, help="Seed for the first attempt; retries advance it."),
    steps: int = typer.Option(7200, help="Max steps per episode."),
    require_dock: bool = typer.Option(
        False, "--require-dock/--no-require-dock",
        help="Keep only episodes that ended docked, retrying seeds until "
             "--episodes of them exist. Off by default: a rollout of whatever "
             "happened is the more common thing to want."),
    max_attempts: Optional[int] = typer.Option(
        None, help="Cap on episodes rolled while retrying (default: 20x --episodes). "
                   "A port that cannot be reached under a given noise preset is a "
                   "finding, not something to loop on."),
    render: bool = typer.Option(True, "--render/--no-render",
                                help="Render video. On by default -- video is the point."),
    render_views: str = typer.Option(
        "fpv,dragon_iso",
        help=f"Views to record: 'all' or a comma-joined list of {', '.join(VIEW_NAMES)}."),
    frame_stride: int = typer.Option(
        0, help="Also write every Nth frame as a PNG, for figures. 0 writes none."),
    render_workers: int = typer.Option(1, help="Parallel render worker processes."),
    gpu_index: Optional[int] = typer.Option(
        None, help="GPU this run uses, for the rollout as well as the renderer."
    ),
) -> None:
    """Roll out a few episodes and write their video, for review or figures.

    Unlike `generate`, which fills a transition budget and writes a LeRobot
    dataset, this produces a handful of clips and no dataset. It is also the
    only path that can insist on a particular OUTCOME: `--require-dock` keeps
    drawing seeds until it has the requested number of successful docks, which
    is what a port-by-port sweep of docking footage needs.
    """
    # Every argument check comes first, before the env config is read, before
    # the GPU is probed -- which pins a device for the whole process -- and
    # long before any rollout: `--require-dock` runs can spend an hour before
    # the first frame, and a usage error must cost a usage error rather than
    # that. Same discipline generate documents at its own flag checks.
    if env not in ENV_REGISTRY:
        raise typer.BadParameter(
            f"unknown environment '{env}'; available: {', '.join(ENV_REGISTRY)}"
        )
    if policy == "union":
        raise typer.BadParameter(
            "'union' draws a different sub-policy per episode, so a handful of "
            "clips filmed under it shows no one behaviour -- it is a training "
            "mixture, and `generate --policy union` is where it belongs. Film "
            "dock, orbit or random."
        )
    if policy not in ("dock", "orbit", "random"):
        raise typer.BadParameter(
            f"unknown policy '{policy}'; use dock, orbit or random"
        )
    if episodes < 1:
        raise typer.BadParameter(f"--episodes must be >= 1, got {episodes}")
    if steps < 1:
        raise typer.BadParameter(f"--steps must be >= 1, got {steps}")
    if render_workers < 1:
        raise typer.BadParameter(f"--render-workers must be >= 1, got {render_workers}")
    if port and port not in PORT_NAMES:
        raise typer.BadParameter(
            f"unknown port '{port}'; one of: {', '.join(PORT_NAMES)}"
        )
    if port and policy != "dock":
        # Refused rather than ignored: the '{policy}' policy never regulates to
        # a port, so accepting --port here would let a port-by-port sweep
        # report eight port-targeted rollouts when it produced eight identical
        # ones that flew nowhere near a port.
        raise typer.BadParameter(
            f"--port is inert for the '{policy}' policy, which does not fly to a "
            f"docking port at all, so this run would not be the port-targeted "
            f"rollout it reads as. Drop --port, or pass --policy dock."
        )
    if frame_stride < 0:
        raise typer.BadParameter(f"--frame-stride must be >= 0, got {frame_stride}")
    if frame_stride and not render:
        raise typer.BadParameter(
            "--frame-stride taps the rendered frames on their way past, and "
            "--no-render produces none for it to tap. Drop --frame-stride, or "
            "drop --no-render."
        )
    view_keys = keys_for_names(_parse_render_views(render_views))
    attempts_cap = max_attempts if max_attempts is not None else 20 * episodes
    if attempts_cap < episodes:
        raise typer.BadParameter(
            f"--max-attempts {attempts_cap} is below --episodes {episodes}, so "
            f"the requested number of episodes cannot be reached however they "
            f"turn out; raise it to at least {episodes}."
        )
    # Created up front rather than at write time: an unwritable path, or one
    # that is already a file, is the caller naming the wrong directory, and
    # under --require-dock the first write comes an hour of retries later --
    # taking every episode rolled in the meantime with it.
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise typer.BadParameter(
            f"cannot write --out {out}: {exc}", param_hint="--out"
        ) from exc

    env_spec = ENV_REGISTRY[env]
    try:
        cfg = env_spec.config_cls.load(env_config) if env_config is not None else env_spec.config_cls()
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise typer.BadParameter(
            f"cannot read --env-config {env_config}: {exc}", param_hint="--env-config"
        ) from exc

    if render and not env_spec.renderable:
        renderable = ", ".join(name for name, spec in ENV_REGISTRY.items() if spec.renderable)
        raise typer.BadParameter(
            f"--render does not support the '{env}' environment yet, only "
            f"{renderable}. Every frame is posed through a render adapter, which "
            "is what reads an env's rows in that env's own element order, and "
            f"'{env}' has none registered (EnvSpec.make_render_adapter). "
            "Rendering it arrives with its adapter, which is what flips "
            "EnvSpec.renderable; until then, drop --render to roll out without "
            "video."
        )

    try:
        policy_cfg = _policy_config(policy, "measurement", port)
    except Exception as exc:  # pydantic rejects unknown policies and ports
        raise typer.BadParameter(f"invalid policy '{policy}': {exc}") from exc

    _check_dock_ports_agree(cfg, policy_cfg, "rollout")

    if render:
        from .render.device import check_gpu_index, select_gpu

        # Both of these resolve before the rollout, exactly as generate does
        # them and for the same reason: with --require-dock the clips are not
        # reached until an hour of retries is already spent, and neither an
        # unusable --gpu-index nor a dt with no whole frame rate to write the
        # clips at may be discovered there.
        resolved_fps = _resolve_fps(None, cfg.dt)
        try:
            if render_workers == 1:
                select_gpu(gpu_index)
            else:
                check_gpu_index(gpu_index)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--gpu-index") from exc

    # Seeds are retried whole batches at a time: an attempt rolls as many
    # episodes as are still wanted, keeps whichever qualify, and the next
    # attempt asks for the shortfall under the next seed.
    driver_seed, attempted, kept = seed, 0, []
    while len(kept) < episodes and attempted < attempts_cap:
        wanted = min(episodes - len(kept), attempts_cap - attempted)
        chosen = _resolve_driver("auto", cfg, policy_cfg, wanted, env_spec)
        batch = chosen.driver.generate(
            RolloutSpec(num_episodes=wanted, max_steps=steps, seed=driver_seed)
        )
        outcomes = classify_batch(batch, cfg, env_spec)
        for index, outcome in enumerate(outcomes):
            if len(kept) == episodes:
                break
            if require_dock and not outcome.docked:
                continue
            kept.append((driver_seed, wanted, index, batch, outcome))
        attempted += batch.num_episodes
        typer.echo(
            f"[rollout] seed={driver_seed}: {batch.num_episodes} episode(s), "
            f"{sum(o.docked for o in outcomes)} docked, "
            f"{len(kept)}/{episodes} kept ({attempted}/{attempts_cap} rolled)"
        )
        driver_seed += 1

    if len(kept) < episodes:
        raise typer.BadParameter(
            f"only {len(kept)} of {attempted} episodes docked, short of the "
            f"{episodes} requested. Raise --max-attempts, or check whether "
            f"{'port ' + port if port else 'this dock pose'} is reachable under "
            f"this config's sensor noise."
        )

    source = kept[0][3]
    kept_batch = pack_episodes(
        [_episode_row(batch, index) for _, _, index, batch, _ in kept],
        obs_dim=source.observations.shape[2],
        act_dim=source.actions.shape[2],
        records_policy_ids=source.policy_ids is not None,
        records_dock_targets=source.dock_targets is not None,
        records_true_state=source.true_state is not None,
        state_dim=env_spec.layout.state_dim,
        records_terminal_events=source.terminal_events is not None,
    )

    # The config as run, not the file that was passed: --env-config is
    # optional and the defaults it falls back to move with the package, so a
    # clip is only reproducible from the config it was actually flown under.
    cfg.to_toml(out / "env_config.toml")
    (out / "rollout.json").write_text(
        json.dumps(
            {
                "env": env,
                "policy": policy,
                "port": port,
                "require_dock": require_dock,
                # The horizon flown, which the env config cannot supply: an
                # episode ends at min(--steps, cfg.max_steps), so a lowered
                # --steps is only recoverable from here.
                "steps": steps,
                # Enough to re-roll any one clip on its own:
                #   rollout --seed SEED --episodes WANTED --steps STEPS \
                #           --no-require-dock --env-config env_config.toml
                # and take BATCH_INDEX out of it. `wanted` is part of that and
                # not decoration -- the driver is built with num_envs=wanted,
                # so the attempt's size sets lane count, lane assignment and
                # how much of the seed's stream each lane consumes. `episode`
                # is the position in this rollout, which is the number its
                # clip and its stills directory carry; `batch_index` is the
                # position within the attempt that rolled it, and the two
                # differ whenever an attempt's earlier episodes were dropped.
                "episodes": [
                    {
                        "episode": position,
                        "seed": episode_seed,
                        "wanted": wanted,
                        "batch_index": index,
                        **asdict(outcome),
                    }
                    for position, (episode_seed, wanted, index, _, outcome)
                    in enumerate(kept)
                ],
            },
            indent=2,
        )
        + "\n"
    )

    if render:
        from .datasets.video import (
            iter_batch_frames,
            tee_episode_clips,
            tee_episode_stills,
        )
        from .render.earth import earth_texture_path
        from .render.iss_scene import RenderConfig

        # Resolved once here rather than inside each render worker, which is
        # what generate does and for the same reasons: a miss bakes a full map
        # from a multi-gigabyte source, and N concurrent bakes could exhaust
        # memory or settle on different tiers.
        for kind in ("color", "clouds", "bump"):
            earth_texture_path(kind, allow_download=True)

        render_cfg = RenderConfig(**cfg.render) if cfg.render else RenderConfig()
        typer.echo(
            f"[render] {int(kept_batch.lengths.sum())} frames, "
            f"{len(view_keys)} view(s) "
            f"({', '.join(key.rsplit('.', 1)[-1] for key in view_keys)}), "
            f"{render_workers} worker(s)"
        )
        frames = iter_batch_frames(
            kept_batch, render_cfg, keys=view_keys, workers=render_workers,
            gpu_index=gpu_index, env_name=env, env_cfg=cfg,
        )
        frames = tee_episode_clips(frames, out / "media", "rollout", resolved_fps)
        if frame_stride:
            frames = tee_episode_stills(frames, out / "frames", frame_stride)
        # Both tees are pass-through generators, and unlike generate there is
        # no dataset writer downstream to pull them, so nothing renders unless
        # this drains the stream itself. Drained through a zero-length deque
        # rather than a `for` loop: a loop variable is bound until the NEXT
        # next() returns, holding the episode just written while the pool
        # renders the one after it -- two episodes of every view at once,
        # which is the one-episode bound both tees are built to keep.
        deque(frames, maxlen=0)

    typer.echo(f"[done] {out}")


def _stdin_is_interactive() -> bool:
    """Whether a confirmation prompt could actually be answered."""
    return sys.stdin.isatty()


@app.command()
def push(
    run_dir: Path = typer.Argument(..., help="Finished run directory (must contain summary.json)."),
    name: Optional[str] = typer.Option(
        None, help="Repo name (default: derived "
                   "owm-{env}-{version}-{noise}-{goal}-dt{ms}ms-{size}, "
                   "editable at the prompt)."),
    namespace: Optional[str] = typer.Option(
        None, help="Hub namespace (default: the HF_TOKEN account, editable at "
                   "the prompt)."),
    private: Optional[bool] = typer.Option(
        None, "--private/--public",
        help="Repo visibility. Given neither, a new repo is public and one that "
             "already exists keeps the visibility it has."),
    yes: bool = typer.Option(
        False, "--yes", "-y",
        help="Confirm the mirror up front, skipping the prompt. Required when "
             "stdin is not a terminal."),
) -> None:
    """Upload a run directory to the HuggingFace Hub as a dataset repo.

    The upload MIRRORS the run onto the repo: everything already there and not
    in this run is deleted. The repo name is derived from the run's own env
    config and target size, so a regenerated run of the same recipe targets
    the same repo -- which is why the target and the sizes are printed and
    confirmed before anything is uploaded.
    """
    from .datasets.hub import hub_namespace, push_preview, push_run

    try:
        repo_name, counts = push_preview(run_dir, name=name)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        # An unfinished run, one with no LeRobot split, or one whose own
        # summary or env config cannot be read: the caller named the wrong
        # directory or generated it wrong, so say so as a usage error rather
        # than as a traceback. Only the run's own files are read under this
        # handler -- a Hub failure is not the run's fault and must not be
        # reported as one.
        raise typer.BadParameter(
            f"cannot read the run in {run_dir}: {exc}", param_hint="RUN_DIR"
        ) from exc
    # Prompted, not just defaulted: a push usually goes to the token's own
    # account under the derived name, but redirecting one to an org or
    # renaming it should not require memorizing flags. A given flag, --yes,
    # or a non-interactive stdin each keep the defaults silently.
    prompting = _stdin_is_interactive() and not yes
    if namespace is None and prompting:
        namespace = typer.prompt("Push to owner", default=hub_namespace())
    else:
        namespace = hub_namespace(namespace)
    if name is None and prompting:
        repo_name = typer.prompt("Dataset name", default=repo_name)
    repo_id = f"{namespace}/{repo_name}"

    typer.echo(f"[push] {run_dir} -> {repo_id}")
    for split, count in counts.items():
        typer.echo(f"[push]   {split}: {count['episodes']} episodes, "
                   f"{count['transitions']} transitions")
    typer.echo("[push] this MIRRORS the run onto that repo: whatever is there now "
               "is replaced, and any file this run does not have is deleted")
    if not yes:
        if not _stdin_is_interactive():
            raise typer.BadParameter(
                f"refusing to mirror over {repo_id} unconfirmed: stdin is not a "
                "terminal, so the prompt above cannot be answered. Check the repo "
                "and the sizes, then pass --yes.",
                param_hint="--yes",
            )
        typer.confirm(f"Replace {repo_id} with this run?", abort=True)

    # The confirmed name and namespace are handed back rather than left to be
    # derived a second time, so the repo that is written is the one that was
    # named above.
    repo_id = push_run(run_dir, name=repo_name, namespace=namespace, private=private)
    typer.echo(f"[push] https://huggingface.co/datasets/{repo_id}")


earth_app = typer.Typer(add_completion=False, help="Publish and retrieve the Earth texture assets.")
app.add_typer(earth_app, name="earth")


@earth_app.command("push")
def earth_push(
    namespace: Optional[str] = typer.Option(
        None, help=f"Hub namespace (default: the {EARTH_REPO_ID} owner)."),
    private: Optional[bool] = typer.Option(
        None, "--private/--public",
        help="Repo visibility. Given neither, a new repo is public and one that "
             "already exists keeps the visibility it has."),
    yes: bool = typer.Option(
        False, "--yes", "-y",
        help="Confirm the upload up front, skipping the prompt. Required when "
             "stdin is not a terminal."),
) -> None:
    """Upload the Earth maps and sources to the asset dataset repo.

    The upload is additive: it replaces the files it names and leaves every
    other file in the repo alone.
    """
    relpaths = [map_relpath(k) for k in TEXTURE_KINDS]
    relpaths += [source_relpath(k) for k in TEXTURE_KINDS]
    present = [p for p in relpaths if (earth_dir() / p).exists()]
    if not present:
        raise typer.BadParameter(
            f"none of the Earth assets are present under {earth_dir()}; there is "
            "nothing to publish"
        )

    typer.echo(f"[earth] publishing to {namespace or EARTH_REPO_OWNER}")
    for relpath in present:
        size = (earth_dir() / relpath).stat().st_size / 1e6
        typer.echo(f"[earth]   {relpath} ({size:.0f} MB)")
    if not yes:
        if not _stdin_is_interactive():
            raise typer.BadParameter(
                "refusing to publish unconfirmed: stdin is not a terminal, so the "
                "prompt cannot be answered. Check the file list above, then pass --yes.",
                param_hint="--yes",
            )
        typer.confirm("Publish these files?", abort=True)

    repo_id = upload_assets(present, namespace=namespace, private=private)
    typer.echo(f"[earth] https://huggingface.co/datasets/{repo_id}")


@earth_app.command("pull-sources")
def earth_pull_sources() -> None:
    """Fetch the high-resolution sources needed to regenerate the maps.

    Several gigabytes. The renderer never needs these -- it reads the maps.
    """
    for kind in TEXTURE_KINDS:
        relpath = source_relpath(kind)
        path = download_asset(relpath)
        if path is None:
            raise typer.BadParameter(f"could not fetch {relpath} from {EARTH_REPO_ID}")
        typer.echo(f"[earth] {path}")


@earth_app.command("regenerate")
def earth_regenerate(
    kind: str = typer.Option(
        ",".join(TEXTURE_KINDS), "--kind",
        help="Comma-separated texture kinds to downsample."),
) -> None:
    """Downsample the full maps from the local sources, replacing existing maps.

    This is how a source that has just been replaced reaches the renderer:
    resolution otherwise prefers the hosted map over downsampling.
    """
    kinds = [k.strip() for k in kind.split(",") if k.strip()]
    unknown = [k for k in kinds if k not in TEXTURE_KINDS]
    if unknown:
        raise typer.BadParameter(
            f"unknown texture kind(s): {', '.join(unknown)}. "
            f"Choose from {', '.join(TEXTURE_KINDS)}.",
            param_hint="--kind",
        )
    for k in kinds:
        typer.echo(f"[earth] downsampling {k}...")
        try:
            typer.echo(f"[earth] {regenerate_map(k)}")
        except FileNotFoundError as exc:
            raise typer.BadParameter(
                f"{exc}. Fetch the sources with `owm-envs earth pull-sources`.",
                param_hint="--kind",
            ) from exc


def _simulation_fps(dt: float) -> int | None:
    """Whole frames per second implied by `dt`, or None if 1/dt is not whole.

    One frame is recorded per simulation step, so 1/dt is the rate at which
    the recorded frames actually occur.
    """
    rate = 1.0 / dt
    nearest = round(rate)
    if nearest < 1 or abs(rate - nearest) > 1e-9:
        return None
    return int(nearest)


def _resolve_fps(requested: int | None, dt: float) -> int:
    """Frame rate to stamp on the dataset.

    fps is metadata: it does not change the rollout, but it is what a consumer
    reads back to recover the interval between frames. When it disagrees with
    1/dt, every recovered interval is wrong and the video plays at the wrong
    speed, so the default tracks dt rather than any fixed rate.

    A dt whose reciprocal is not whole has no exact integer fps at all. The
    default is refused there rather than rounded, and an explicit value is
    warned about, since it cannot be right -- only chosen deliberately.
    """
    simulation_fps = _simulation_fps(dt)
    if requested is None:
        if simulation_fps is None:
            raise typer.BadParameter(
                f"dt={dt} gives a simulation rate of {1.0 / dt:.6g} frames per "
                "second, which is not whole, so there is no exact integer fps "
                "to default to; pass --fps explicitly"
            )
        return simulation_fps
    if requested < 1:
        raise typer.BadParameter(f"--fps must be >= 1, got {requested}")
    mismatch = "inter-frame interval that the physics did not use"
    if simulation_fps is None:
        # No integer can match a non-whole rate, so an explicit --fps is
        # necessarily wrong here rather than merely disagreeing. Warn on its
        # own terms: comparing it to a rounded rate would imply some other
        # integer would have been right.
        typer.echo(
            f"[warn] --fps {requested} cannot match the simulation rate "
            f"{1.0 / dt:.6g} (dt={dt}), which is not whole; no integer fps "
            f"can, so the dataset will report an {mismatch}"
        )
    elif requested != simulation_fps:
        typer.echo(
            f"[warn] --fps {requested} does not match the simulation rate "
            f"{simulation_fps} (dt={dt}); the dataset will report an {mismatch}"
        )
    return requested


class _Chosen:
    def __init__(self, name: str, driver: object):
        self.name = name
        self.driver = driver


def _resolve_driver(
    requested: str, cfg: BaseTaskConfig, policy_cfg: PolicyConfig, num_envs: int, env_spec: EnvSpec
) -> _Chosen:
    from .drivers.scan_driver import ScanDriver, supports_fused_rollout
    from .drivers.vector_env_driver import VectorEnvDriver
    from .envs.common.policy_source import TaskPolicySource

    def build_vector() -> _Chosen:
        # The vector env reads its dock pose straight off cfg and has no
        # channel for a per-episode target, so a policy given any port set at
        # all regulates to the assigned port but is scored against DockConfig.
        # One port is no safer than several: even PMA-2's derived pose sits
        # 0.84 m off the shipped DockConfig one, well outside the success
        # gate. The scan driver, which --driver auto selects, carries the
        # target per lane and does not have this gap.
        ports = policy_cfg.dock.ports
        if ports:
            targeted = f"the {len(ports)} ports" if len(ports) > 1 else "the port"
            typer.echo(
                f"[warn] --driver vector scores dock success against DockConfig, not "
                f"{targeted} this policy targets ({', '.join(p.name for p in ports)}), "
                f"so 'success' is "
                f"scored at the wrong pose for every episode whose assigned port is not "
                f"DockConfig's; --driver scan does not have this limitation"
            )
        # An env whose layout carries a chief block (iss-numerical) holds
        # ABSOLUTE ECI columns, and its relative view is the difference of two
        # of them. This driver crosses a numpy boundary between the env and
        # the policy source, where the state has already been narrowed to
        # float32, so that difference is taken at f32's ~0.5 m grain on a
        # ~6.8e6 m radius rather than at the state's own width -- see
        # `envs/common/policy_source.py`. The scan driver derives the same
        # view inside the f64 rollout and carries none of it.
        if env_spec.layout.chief is not None:
            typer.echo(
                f"[warn] --driver vector hands {env_spec.name}'s policy inputs -- and "
                "the goal-error block it appends to every recorded observation -- a "
                "relative view differenced from float32 ECI columns, leaving up to "
                "~1 m of quantization on the station-relative position they read; the "
                "observation columns themselves still come from the env's own f64 "
                "derivation, and --driver scan derives the view at f64 throughout"
            )
        # TaskPolicySource applies its own policy-aware goal-error block (see
        # augment_observation) from the ORIGINAL cfg; the env it drives must
        # therefore stay at the raw 13-dim observation, or the block would be
        # appended twice -- once by the env, once by the policy source.
        env_cfg = cfg.model_copy(update={"observation": cfg.observation.model_copy(update={"goal_error": False})})
        return _Chosen(
            "vector",
            VectorEnvDriver(
                env_factory=lambda: env_spec.make_vector_env(num_envs, env_cfg),
                policy_source=TaskPolicySource(cfg, policy_cfg, view=env_spec.view),
            ),
        )

    def build_scan() -> _Chosen:
        return _Chosen(
            "scan", ScanDriver(cfg=cfg, policy_cfg=policy_cfg, num_envs=num_envs, env_spec=env_spec)
        )

    if requested == "vector":
        return build_vector()
    if requested == "scan":
        return build_scan()
    if requested == "auto":
        # The capability check that lets a future non-JAX backend work unchanged.
        return build_scan() if supports_fused_rollout(env_spec.make_dynamics(cfg)) else build_vector()
    raise typer.BadParameter(f"unknown driver '{requested}'; use auto, scan or vector")
