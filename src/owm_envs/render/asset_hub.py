"""The HuggingFace dataset repo mirroring the Earth texture assets.

The repo's tree is identical to the local tree under `resources/earth/`:
`maps/` holds the finished full-resolution maps the renderer reads, and
`sources/` the multi-gigabyte imagery they are downsampled from. Because the
two trees have the same shape, `hf_hub_download` replicating repo structure
beneath `local_dir` places every file exactly where the resolver in
`owm_envs.render.earth` already looks for it.

This module is the only place that knows the repo id.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from huggingface_hub import hf_hub_download

from owm_envs.render import resources_dir

EARTH_REPO_OWNER = "sislaboratory"
EARTH_REPO_NAME = "owm-earth-textures"
EARTH_REPO_ID = f"{EARTH_REPO_OWNER}/{EARTH_REPO_NAME}"


def earth_dir() -> Path:
    """Local root that the dataset repo's tree is replicated into."""
    return resources_dir() / "earth"


def download_asset(relpath: str) -> Path | None:
    """Fetch one repo-relative asset. Returns None when it is unavailable.

    Never raises: callers resolve against a committed fallback that is
    present in every clone, so an unreachable Hub is a quality downgrade
    rather than a failure.
    """
    try:
        return Path(
            hf_hub_download(
                repo_id=EARTH_REPO_ID,
                repo_type="dataset",
                filename=relpath,
                local_dir=str(earth_dir()),
            )
        )
    except Exception as exc:  # HfHubHTTPError, OSError, and anything else
        warnings.warn(f"could not fetch {relpath} ({exc})")
        return None
