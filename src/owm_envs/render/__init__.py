"""Offscreen rendering for the ISS environment.

Importing this package pulls in pygfx and wgpu, which are an OPTIONAL extra.
Nothing in `owm_envs` imports it at top level -- the environment and the dataset
pipeline both work without a GPU or a render install.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["resources_dir", "asset_path"]


def resources_dir() -> Path:
    """Root of the packaged 3D assets."""
    return Path(__file__).resolve().parent / "resources"


def asset_path(*parts: str) -> Path:
    """Resolve a packaged asset, failing loudly and legibly when it is absent."""
    path = resources_dir().joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(
            f"render asset not found: {path}. If this is a fresh clone, the "
            f"assets are git-lfs tracked -- run `git lfs pull`."
        )
    return path
