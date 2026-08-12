"""Keep the whole test suite off the network, and off the GPU.

Scene construction resolves Earth textures with `allow_download=True`, so on
a machine without the high-resolution sources on disk an unguarded run would
pull gigabytes from the asset mirror. Any test that builds a scene reaches
that path -- `tests/envs/iss` constructs render-mode environments too -- so
the guard lives at the root rather than beside the render tests. Every test
must be satisfiable by the committed fallback maps.

The suite computes on the CPU backend for two reasons. The golden rollouts
assert bit identity, and a backend is free to lower the same arithmetic
differently, so a GPU host would compare this package's results against
another backend's rounding rather than against a change in this package --
which is the only thing those tests exist to catch. And a test process that
reaches a GPU takes 75% of it, which on a shared host is a card taken from
whoever else is on it for the length of a run.

Set before anything imports jax, because the choice is read when the backend
initialises, which happens during import -- see `owm_envs._entry` for the same
constraint on the CLI side. `setdefault`, so a run that means to exercise the
GPU path can still say `JAX_PLATFORMS=` and be believed.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(url, *args, **kwargs):
        raise OSError(f"tests must not download: {url}")

    monkeypatch.setattr("urllib.request.urlretrieve", refuse)
