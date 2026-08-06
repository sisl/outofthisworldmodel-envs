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

import sys
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
from .drivers.types import RolloutSpec
from .envs import ENV_REGISTRY, EnvSpec
from .envs.common.config import BaseTaskConfig, ObservationConfig
from .envs.common.docking_ports import PORT_NAMES
from .envs.common.policies import DockParams, PolicyConfig
from .envs.common.sensing import PRESETS

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
    cfg: ISSConfig, policy_cfg: PolicyConfig, split: str
) -> None:
    """Refuse an env port set that generation would not honour.

    Generation resolves an episode's target from `policy.dock.ports`, never
    from `ISSConfig.dock.ports` -- that second field is the one `ISSEnv` draws
    from. They are otherwise the same field, resolved by the same code, so a
    config that sets the env's and leaves the policy's empty reads as a
    multi-port run and would quietly write single-target data: the one outcome
    neither reading of the config asks for. `ISSVectorEnv` refuses the same
    disagreement rather than ignoring it, and so does this.

    An empty `cfg.dock.ports` -- every config written before that field
    existed -- says nothing about ports and is left alone.
    """
    if not cfg.dock.ports or cfg.dock.ports == policy_cfg.dock.ports:
        return
    env_names = [port.name for port in cfg.dock.ports]
    policy_names = [port.name for port in policy_cfg.dock.ports]
    raise typer.BadParameter(
        f"split '{split}': dock ports disagree. The env config's dock.ports "
        f"names {env_names}, the policy's dock.ports names {policy_names}, and "
        f"generation flies the policy's -- so this run would not record the "
        f"ports the env config asks for. Name the same ports on both (--dock-"
        f"ports, or the split's PORTS field), or drop dock.ports from the env "
        f"config."
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
        None, help="GenerationConfig YAML; exclusive with --split/--steps/--num-envs/--driver/--fps. "
                   "configs/generation_default.yaml is the shipped docking recipe: a union-policy "
                   "train split on five ports and a dock-policy val split on all "
                   "eight, so validation measures the held-out approaches (both zenith "
                   "corridors and Unity nadir)."),
    env_config: Optional[Path] = typer.Option(
        None, "--env-config",
        help="Environment config file to load, YAML or TOML by suffix. The shipped "
             "iss environments are configs/iss_*.toml: iss_default.toml plus the six "
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
             "Empty (the default) keeps the single pose in ISSConfig.dock. Sets the "
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
        help="GPU adapter index for rendering (default: wgpu's own choice). "
             "Counts discrete GPUs only. Also settable via OWM_ENVS_GPU_INDEX.",
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

    # Both checked before the GPU is touched: a usage error must cost a usage
    # error, not an adapter probe -- which pins a device for the process --
    # followed by one.
    if render_views is not None:
        _parse_render_views(render_views)
    if gen_config is not None and any(
        v is not None for v in (env, split, steps, num_envs, driver, fps, render_views)
    ):
        raise typer.BadParameter(
            "--gen-config is exclusive with "
            "--env/--split/--steps/--num-envs/--driver/--fps/--render-views"
        )

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

    if lerobot:
        # lerobot is declared only in the optional 'datasets' extra, so a
        # base install hits ModuleNotFoundError deep inside write_lerobot_split.
        # Check up front, before spending time on a rollout, and say what a
        # user who asked for a dataset needs to do to actually get one,
        # rather than quietly writing metadata only.
        try:
            import lerobot  # noqa: F401
        except ModuleNotFoundError as exc:
            raise typer.BadParameter(
                "lerobot is not installed; install the 'datasets' extra "
                "(pip install 'owm-envs[datasets]') or pass --no-lerobot"
            ) from exc

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

    env_name = gen.env if gen_config is not None else (env or "iss")
    if env_name not in ENV_REGISTRY:
        raise typer.BadParameter(
            f"unknown environment '{env_name}'; available: {', '.join(ENV_REGISTRY)}"
        )
    env_spec = ENV_REGISTRY[env_name]

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
        cfg = cfg.model_copy(update={"observation": ObservationConfig(goal_error=goal_error)})
    resolved_fps = _resolve_fps(gen.fps, cfg.dt)
    try:
        policy_cfg = _policy_config(policy, observe, dock_ports)
    except Exception as exc:  # pydantic rejects unknown policies and ports
        raise typer.BadParameter(f"invalid policy '{policy}': {exc}") from exc

    # Drivers bake their policy in at construction, so a split with its own
    # policy needs its own driver instance.
    # Every split's ports are checked before any split generates. Checking a
    # split just before it runs is not enough: the splits are generated in
    # order, so a disagreement in the last one would be found only after every
    # earlier split had already paid for its full rollout.
    for name, spec in gen.splits.items():
        _check_dock_ports_agree(cfg, spec.policy or policy_cfg, name)

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


def _stdin_is_interactive() -> bool:
    """Whether a confirmation prompt could actually be answered."""
    return sys.stdin.isatty()


@app.command()
def push(
    run_dir: Path = typer.Argument(..., help="Finished run directory (must contain summary.json)."),
    name: Optional[str] = typer.Option(
        None, help="Repo name (default: derived owm-{env}-{noise}-{goal}-dt{ms}ms)."),
    namespace: Optional[str] = typer.Option(
        None, help="Hub namespace (default: the HF_TOKEN account)."),
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
    in this run is deleted. The repo name is derived from the run's env config
    alone, so a trial run and the production run generated from the same config
    target the same repo -- which is why the target and the sizes are printed
    and confirmed before anything is uploaded.
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
    namespace = hub_namespace(namespace)
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
    requested: str, cfg: BaseTaskConfig, policy_cfg: PolicyConfig, num_envs: int, spec: EnvSpec
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
        # TaskPolicySource applies its own policy-aware goal-error block (see
        # augment_observation) from the ORIGINAL cfg; the env it drives must
        # therefore stay at the raw 13-dim observation, or the block would be
        # appended twice -- once by the env, once by the policy source.
        env_cfg = cfg.model_copy(update={"observation": cfg.observation.model_copy(update={"goal_error": False})})
        return _Chosen(
            "vector",
            VectorEnvDriver(
                env_factory=lambda: spec.make_vector_env(num_envs, env_cfg),
                policy_source=TaskPolicySource(cfg, policy_cfg, view=spec.view),
            ),
        )

    def build_scan() -> _Chosen:
        return _Chosen(
            "scan", ScanDriver(cfg=cfg, policy_cfg=policy_cfg, num_envs=num_envs, env_spec=spec)
        )

    if requested == "vector":
        return build_vector()
    if requested == "scan":
        return build_scan()
    if requested == "auto":
        # The capability check that lets a future non-JAX backend work unchanged.
        return build_scan() if supports_fused_rollout(spec.make_dynamics(cfg)) else build_vector()
    raise typer.BadParameter(f"unknown driver '{requested}'; use auto, scan or vector")
