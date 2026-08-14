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
from collections.abc import Sequence
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

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


EARTH_CARD = """\
---
license: other
license_name: unrecorded
pretty_name: OWM Earth Textures
tags:
  - earth-observation
  - texture
  - rendering
---

# OWM Earth Textures

Full-globe equirectangular Earth imagery for the renderer in
[`sisl/outofthisworldmodel-envs`](https://github.com/sisl/outofthisworldmodel-envs).

- `maps/` -- the finished maps the renderer reads. `earth_color_full.jpg` and
  `earth_clouds_full.jpg` are 16384x8192 (~2.4 km/texel at the equator);
  `earth_bump_full.png` is 8192x4096 and stays PNG because the normal map is
  computed from height gradients, which JPEG block artifacts corrupt.
- `sources/` -- the high-resolution imagery the maps are downsampled from.

The renderer fetches `maps/` automatically. `sources/` is needed only to
regenerate the maps at a different width.

## Provenance and licence

**The upstream licence terms for this imagery were not recorded when it was
first collected.** The list below is what is actually documented; it is not an
assertion of terms. Anyone assessing terms should check all four origins.

- https://sketchfab.com/3d-models/earth-41fc80d85dfd480281f21b74b2de2faa
- https://science.nasa.gov/resource/earth-3d-model/
- https://www.cgtrader.com/items/6013252/download-page
- https://maps.drsys.eu/

Which specific origin produced each file was not recorded per file.
"""


def upload_assets(
    relpaths: Sequence[str],
    *,
    namespace: str | None = None,
    private: bool | None = None,
) -> str:
    """Publish the named repo-relative assets. Returns the repo id.

    The upload is ADDITIVE: `allow_patterns` names exactly the files being
    published and no `delete_patterns` is passed, so publishing regenerated
    maps cannot delete the sources, which are gigabytes and expensive to
    restore.

    The card is uploaded BEFORE the assets. The repo is public and carries
    imagery whose licence terms were not recorded, so there must be no window
    in which the files are readable and the disclosure is not.
    """
    api = HfApi()
    repo_id = EARTH_REPO_ID if namespace is None else f"{namespace}/{EARTH_REPO_NAME}"
    api.create_repo(repo_id, repo_type="dataset", private=bool(private), exist_ok=True)
    if private is not None:
        # create_repo ignores `private` for a repo that already exists.
        api.update_repo_settings(repo_id, repo_type="dataset", private=private)
    api.upload_file(
        path_or_fileobj=EARTH_CARD.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(earth_dir()),
        allow_patterns=list(relpaths),
    )
    return repo_id
