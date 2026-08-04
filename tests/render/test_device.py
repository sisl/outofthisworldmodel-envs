"""GPU adapter selection for multi-GPU hosts.

Adapter selection itself needs a GPU and mutates process-global pygfx state,
so only the pure picking logic is unit tested here; the selection call is
covered by a smoke check against real hardware.
"""

import pytest

from owm_envs.render.device import _pick_adapter, select_gpu


class _Fake:
    def __init__(self, summary, adapter_type="DiscreteGPU"):
        self.summary = summary
        self.info = {"adapter_type": adapter_type}


def _dual_a100():
    return [
        _Fake("NVIDIA A100 #0"),
        _Fake("NVIDIA A100 #1"),
        _Fake("llvmpipe (CPU)", adapter_type="CPU"),
        _Fake("NVIDIA A100 via OpenGL", adapter_type="Unknown"),
    ]


def test_pick_by_index_prefers_discrete_gpus():
    assert _pick_adapter(_dual_a100(), 1).summary == "NVIDIA A100 #1"


def test_index_counts_only_discrete_adapters():
    assert _pick_adapter(_dual_a100(), 0).summary == "NVIDIA A100 #0"
    with pytest.raises(ValueError):
        _pick_adapter(_dual_a100(), 2)


def test_hyphenated_adapter_type_is_also_discrete():
    adapters = [_Fake("llvmpipe (CPU)", adapter_type="CPU"),
                _Fake("A100", adapter_type="discrete-gpu")]
    assert _pick_adapter(adapters, 0).summary == "A100"


def test_falls_back_to_all_adapters_when_none_are_discrete():
    adapters = [_Fake("llvmpipe (CPU)", adapter_type="CPU")]
    assert _pick_adapter(adapters, 0).summary == "llvmpipe (CPU)"


def test_out_of_range_names_adapters():
    with pytest.raises(ValueError, match="A100 #0"):
        _pick_adapter([_Fake("A100 #0")], 3)


def test_negative_index_rejected():
    with pytest.raises(ValueError, match="A100 #0"):
        _pick_adapter([_Fake("A100 #0")], -1)


def test_select_gpu_without_index_or_env_is_a_noop(monkeypatch):
    monkeypatch.delenv("OWM_ENVS_GPU_INDEX", raising=False)
    monkeypatch.setattr(
        "owm_envs.render.device._select", lambda index: pytest.fail("must not select")
    )
    select_gpu(None)


def test_select_gpu_reads_env_var(monkeypatch):
    monkeypatch.setenv("OWM_ENVS_GPU_INDEX", "1")
    seen = []
    monkeypatch.setattr("owm_envs.render.device._select", seen.append)
    select_gpu(None)
    assert seen == [1]


def test_explicit_index_overrides_env_var(monkeypatch):
    monkeypatch.setenv("OWM_ENVS_GPU_INDEX", "1")
    seen = []
    monkeypatch.setattr("owm_envs.render.device._select", seen.append)
    select_gpu(0)
    assert seen == [0]


def test_unparseable_env_var_names_the_variable(monkeypatch):
    monkeypatch.setenv("OWM_ENVS_GPU_INDEX", "first")
    with pytest.raises(ValueError, match="OWM_ENVS_GPU_INDEX"):
        select_gpu(None)
