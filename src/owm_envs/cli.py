"""Command-line interface for dataset generation.

    owm-envs generate --out logs/run1 --episodes 512 --policy union
    owm-envs list

`--driver auto` selects the fused JAX path when the backend supports it and
falls back to the generic VectorEnv path otherwise, so the same command works
unchanged for a future non-JAX environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .datasets.stats import build_run_metadata
from .drivers.types import RolloutSpec
from .envs.iss.config import ISSConfig
from .envs.iss.policies import PolicyConfig

app = typer.Typer(add_completion=False, help="Generate world-model training datasets.")


@app.command("list")
def list_envs() -> None:
    """List available environments and their observation/action shapes."""
    from .envs.iss.env import ISSEnv

    env = ISSEnv()
    typer.echo("Available environments:")
    typer.echo(
        f"  iss  (ISS-Docking-v0)  obs={env.observation_space.shape}  "
        f"act={env.action_space.shape}"
    )


@app.command()
def generate(
    out: Path = typer.Option(..., help="Run directory to write."),
    env: str = typer.Option("iss", help="Environment name."),
    episodes: int = typer.Option(64, help="Episodes to generate."),
    steps: int = typer.Option(2000, help="Max steps per episode."),
    policy: str = typer.Option("random", help="random | orbit | dock | union"),
    seed: int = typer.Option(0, help="Base seed."),
    driver: str = typer.Option("auto", help="auto | scan | vector"),
    num_envs: int = typer.Option(8, help="Parallel lanes."),
    fps: Optional[int] = typer.Option(
        None,
        help="Frames per second recorded in the dataset. Defaults to the "
        "simulation rate, 1/dt, from the environment config.",
    ),
    config: Optional[Path] = typer.Option(None, help="ISSConfig YAML to load."),
    split: str = typer.Option("train", help="Split name for the output."),
    lerobot: bool = typer.Option(True, "--lerobot/--no-lerobot", help="Write a LeRobot dataset."),
    render: bool = typer.Option(
        False,
        "--render/--no-render",
        help="Render an egocentric video feed (slow: ~0.1 s/frame; off by default).",
    ),
    render_view: str = typer.Option("DRAGON_FPV", help="Camera view to render, when --render is set."),
) -> None:
    """Roll out trajectories and write a dataset run directory."""
    if env != "iss":
        raise typer.BadParameter(f"unknown environment '{env}'; only 'iss' exists")
    if render and not lerobot:
        raise typer.BadParameter(
            "--render has no effect with --no-lerobot: there is no writer to consume the "
            "rendered frames, so rendering would be pure wasted cost. Drop --render, or "
            "drop --no-lerobot so the frames are written."
        )

    if num_envs < 1:
        # Unguarded, this reaches the scan driver's horizon calculation
        # (division by num_envs -> ZeroDivisionError) or leaves the vector
        # driver spinning forever, since no lane ever produces an episode.
        raise typer.BadParameter(f"--num-envs must be >= 1, got {num_envs}")

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

    cfg = ISSConfig.from_yaml(config) if config is not None else ISSConfig()
    fps = _resolve_fps(fps, cfg.dt)
    try:
        policy_cfg = PolicyConfig(type=policy)
    except Exception as exc:  # pydantic rejects unknown policy types
        raise typer.BadParameter(f"invalid policy '{policy}': {exc}") from exc

    chosen = _resolve_driver(driver, cfg, policy_cfg, num_envs)
    typer.echo(f"[generate] driver={chosen.name} policy={policy} episodes={episodes}")

    spec = RolloutSpec(num_episodes=episodes, max_steps=steps, seed=seed)
    batch = chosen.driver.generate(spec)
    typer.echo(
        f"[generate] {batch.num_episodes} episodes, {batch.total_transitions} transitions"
    )

    # Built now so a batch that cannot produce statistics fails here, before
    # any dataset is written, but flushed last: these files are what marks the
    # run complete, so a failure downstream must not leave them behind.
    metadata = build_run_metadata(
        cfg=cfg, policy_cfg=policy_cfg, batches={split: batch}, fps=fps, seed=seed
    )

    frames = None
    if render:
        from .datasets.video import render_batch_frames
        from .render.iss_scene import RenderConfig

        render_cfg = RenderConfig(**cfg.render) if cfg.render else RenderConfig()
        total = int(batch.lengths.sum())
        typer.echo(
            f"[render] {total} frames at ~0.1 s/frame -> roughly {total * 0.1 / 60:.1f} min"
        )
        frames = render_batch_frames(batch, render_cfg, view=render_view)

    if lerobot:
        from .datasets.lerobot_writer import write_lerobot_split

        write_lerobot_split(out / split, f"{env}/{split}", batch, fps=fps, frames=frames)
        typer.echo(f"[generate] wrote LeRobot split to {out / split}")

    metadata.write(out)
    typer.echo(f"[done] {out}")


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


def _resolve_driver(requested: str, cfg: ISSConfig, policy_cfg: PolicyConfig, num_envs: int) -> _Chosen:
    from .drivers.scan_driver import ScanDriver, supports_fused_rollout
    from .drivers.vector_env_driver import VectorEnvDriver
    from .envs.iss.dynamics import ISSDynamics
    from .envs.iss.vector_env import ISSVectorEnv

    from .envs.iss.policy_source import ISSPolicySource

    def build_vector() -> _Chosen:
        return _Chosen(
            "vector",
            VectorEnvDriver(
                env_factory=lambda: ISSVectorEnv(num_envs=num_envs, cfg=cfg),
                policy_source=ISSPolicySource(cfg, policy_cfg),
            ),
        )

    def build_scan() -> _Chosen:
        return _Chosen("scan", ScanDriver(cfg=cfg, policy_cfg=policy_cfg, num_envs=num_envs))

    if requested == "vector":
        return build_vector()
    if requested == "scan":
        return build_scan()
    if requested == "auto":
        # The capability check that lets a future non-JAX backend work unchanged.
        return build_scan() if supports_fused_rollout(ISSDynamics(cfg)) else build_vector()
    raise typer.BadParameter(f"unknown driver '{requested}'; use auto, scan or vector")
