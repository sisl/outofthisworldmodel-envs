"""The Hub-backed Earth asset mirror: where files land, and how failure behaves."""

import warnings
from pathlib import Path

import pytest

from owm_envs.render import asset_hub
from owm_envs.render.asset_hub import EARTH_REPO_ID, download_asset, earth_dir


@pytest.fixture
def fake_resources(tmp_path, monkeypatch):
    monkeypatch.setattr(asset_hub, "resources_dir", lambda: tmp_path)
    return tmp_path


def test_earth_dir_is_the_local_root_the_repo_tree_replicates_into(fake_resources):
    assert earth_dir() == fake_resources / "earth"


def test_download_replicates_the_repo_path_under_earth_dir(fake_resources, monkeypatch):
    # The repo tree and the local tree are the same shape, so a downloaded
    # `maps/x.jpg` must land where the resolver's tier 1 already looks.
    seen = {}

    def fake_download(*, repo_id, repo_type, filename, local_dir):
        seen.update(repo_id=repo_id, repo_type=repo_type, filename=filename)
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"jpeg")
        return str(dest)

    monkeypatch.setattr(asset_hub, "hf_hub_download", fake_download)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        path = download_asset("maps/earth_color_full.jpg")

    assert path == fake_resources / "earth" / "maps" / "earth_color_full.jpg"
    assert path.read_bytes() == b"jpeg"
    assert seen == {
        "repo_id": EARTH_REPO_ID,
        "repo_type": "dataset",
        "filename": "maps/earth_color_full.jpg",
    }


def test_download_returns_none_on_failure_rather_than_raising(fake_resources, monkeypatch):
    # Every caller has a committed fallback, so a Hub outage is a quality
    # downgrade and must never surface as an exception mid-render.
    def boom(**kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr(asset_hub, "hf_hub_download", boom)
    with pytest.warns(UserWarning, match="could not fetch"):
        assert download_asset("maps/earth_color_full.jpg") is None


def test_the_published_repo_id_is_pinned():
    # Repointing the mirror silently changes what every clone downloads.
    assert asset_hub.EARTH_REPO_OWNER == "sislaboratory"
    assert asset_hub.EARTH_REPO_NAME == "owm-earth-textures"
    assert EARTH_REPO_ID == "sislaboratory/owm-earth-textures"
