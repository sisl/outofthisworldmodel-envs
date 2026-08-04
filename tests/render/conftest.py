"""Keep the render tests off the network.

Scene construction resolves Earth textures with `allow_download=True`, so on
a machine without the high-resolution sources on disk an unguarded test run
would pull gigabytes from the asset mirror. Every test here must be
satisfiable by the committed fallback maps.
"""

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(url, *args, **kwargs):
        raise OSError(f"tests must not download: {url}")

    monkeypatch.setattr("urllib.request.urlretrieve", refuse)
