"""Keep the whole test suite off the network.

Scene construction resolves Earth textures with `allow_download=True`, so on
a machine without the high-resolution sources on disk an unguarded run would
pull gigabytes from the asset mirror. Any test that builds a scene reaches
that path -- `tests/envs/iss` constructs render-mode environments too -- so
the guard lives at the root rather than beside the render tests. Every test
must be satisfiable by the committed fallback maps.
"""

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(url, *args, **kwargs):
        raise OSError(f"tests must not download: {url}")

    monkeypatch.setattr("urllib.request.urlretrieve", refuse)
