"""Which GPU a rollout runs on, and how much of that card XLA may claim.

XLA reads both settings from the environment when its backend first
initialises, so what a unit test can hold is the environment `pin_gpu` leaves
behind, plus its refusal to pretend once the backend is already up. That those
settings really do bound a rollout is a hardware property no fake can
establish: on the dual-A100 host the trial rollout peaks at 1.5 GiB, against
the 59.44 GiB the default allocator asks for on an 80 GB card.
"""

import os

import pytest

from owm_envs._jax_config import pin_gpu

PREALLOCATE = "XLA_PYTHON_CLIENT_PREALLOCATE"
VISIBLE = "CUDA_VISIBLE_DEVICES"
ORDER = "CUDA_DEVICE_ORDER"


@pytest.fixture(autouse=True)
def _unpinned(monkeypatch):
    """A fresh process's starting point: no backend yet, and no settings."""
    monkeypatch.setattr(
        "owm_envs._jax_config.xla_bridge.backends_are_initialized", lambda: False
    )
    for name in (PREALLOCATE, VISIBLE, ORDER):
        monkeypatch.delenv(name, raising=False)


def test_an_index_confines_cuda_and_stops_the_full_card_grab(monkeypatch):
    pin_gpu(1)
    assert os.environ[VISIBLE] == "1"
    assert os.environ[PREALLOCATE] == "false"
    # Without this CUDA counts by capability while the operator and nvidia-smi
    # count along the bus, so "1" would name two different cards on the two
    # halves of the run.
    assert os.environ[ORDER] == "PCI_BUS_ID"


def test_no_index_still_stops_the_full_card_grab():
    """The 75% claim is wrong on whichever card JAX picks for itself."""
    pin_gpu(None)
    assert os.environ[PREALLOCATE] == "false"
    assert VISIBLE not in os.environ


def test_an_explicit_index_overrides_an_inherited_visible_devices(monkeypatch):
    """An ambient value must not quietly win over an operator's request.

    Deferring to it would put the rollout back on the card the operator asked
    it to leave, and say nothing about having done so.
    """
    monkeypatch.setenv(VISIBLE, "0")
    pin_gpu(1)
    assert os.environ[VISIBLE] == "1"


def test_an_operator_set_allocation_policy_is_left_alone(monkeypatch):
    """Growing on demand is the default, not a rule imposed on the operator."""
    monkeypatch.setenv(PREALLOCATE, "true")
    pin_gpu(1)
    assert os.environ[PREALLOCATE] == "true"


def test_pinning_after_the_backend_is_up_is_refused(monkeypatch):
    """Too late fails closed.

    The settings cannot reach a backend that has already read them, so the
    alternative to refusing is a run that holds most of a shared card and
    computes on the wrong GPU without ever saying so. This is the only point
    that can still see the ordering, so it is where it is enforced.
    """
    monkeypatch.setattr(
        "owm_envs._jax_config.xla_bridge.backends_are_initialized", lambda: True
    )
    with pytest.raises(RuntimeError, match="too late"):
        pin_gpu(1)
    assert VISIBLE not in os.environ
    assert PREALLOCATE not in os.environ
