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

from .datasets.stats import write_run_metadata
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
    fps: int = typer.Option(24, help="Frames per second recorded in the dataset."),
    config: Optional[Path] = typer.Option(None, help="ISSConfig YAML to load."),
    split: str = typer.Option("train", help="Split name for the output."),
    lerobot: bool = typer.Option(True, "--lerobot/--no-lerobot", help="Write a LeRobot dataset."),
) -> None:
    """Roll out trajectories and write a dataset run directory."""
    if env != "iss":
        raise typer.BadParameter(f"unknown environment '{env}'; only 'iss' exists")

    cfg = ISSConfig.from_yaml(config) if config is not None else ISSConfig()
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

    out.mkdir(parents=True, exist_ok=True)
    write_run_metadata(out, cfg=cfg, policy_cfg=policy_cfg, batches={split: batch}, fps=fps, seed=seed)

    if lerobot:
        from .datasets.lerobot_writer import write_lerobot_split

        write_lerobot_split(out / split, f"{env}/{split}", batch, fps=fps)
        typer.echo(f"[generate] wrote LeRobot split to {out / split}")

    typer.echo(f"[done] {out}")


class _Chosen:
    def __init__(self, name: str, driver: object):
        self.name = name
        self.driver = driver


def _resolve_driver(requested: str, cfg: ISSConfig, policy_cfg: PolicyConfig, num_envs: int) -> _Chosen:
    from .drivers.scan_driver import ScanDriver, supports_fused_rollout
    from .drivers.vector_env_driver import VectorEnvDriver
    from .envs.iss.dynamics import ISSDynamics
    from .envs.iss.vector_env import ISSVectorEnv

    from .envs.iss.policy_source import IssPolicySource

    def build_vector() -> _Chosen:
        return _Chosen(
            "vector",
            VectorEnvDriver(
                env_factory=lambda: ISSVectorEnv(num_envs=num_envs, cfg=cfg),
                policy_source=IssPolicySource(cfg, policy_cfg),
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
