"""GPU adapter selection for multi-GPU hosts.

Adapter selection itself needs a GPU and mutates process-global pygfx state,
so only the pure picking logic is unit tested here. That the index maps to the
physical card is a hardware property no fake can establish; it was verified on
the dual-A100 host with nvidia-smi and is recorded in the module docstring of
`owm_envs.render.device`.

The fakes mirror a real enumeration on that host: Vulkan adapters first, then
the OpenGL views of the same cards.
"""

import pytest

from owm_envs.render.device import _pick_adapter, select_gpu


@pytest.fixture(autouse=True)
def _forget_selection(monkeypatch):
    """Which GPU is pinned is process-global; each test starts unpinned."""
    monkeypatch.setattr("owm_envs.render.device._SELECTED", None)


class _Fake:
    def __init__(self, summary, adapter_type="DiscreteGPU", backend_type="Vulkan"):
        self.summary = summary
        self.info = {"adapter_type": adapter_type, "backend_type": backend_type}


def _dual_a100():
    return [
        _Fake("NVIDIA A100 #0"),
        _Fake("NVIDIA A100 #1"),
        _Fake("llvmpipe (CPU)", adapter_type="CPU"),
        _Fake("NVIDIA A100 OpenGL", adapter_type="Unknown", backend_type="OpenGL"),
    ]


def test_pick_by_index_prefers_discrete_gpus():
    assert _pick_adapter(_dual_a100(), 1).summary == "NVIDIA A100 #1"


def test_index_counts_only_discrete_adapters():
    assert _pick_adapter(_dual_a100(), 0).summary == "NVIDIA A100 #0"
    with pytest.raises(ValueError):
        _pick_adapter(_dual_a100(), 2)


def test_duplicate_opengl_views_do_not_shift_the_index():
    """A driver reporting the OpenGL view as discrete must not add a slot."""
    adapters = [
        _Fake("A100 #0"),
        _Fake("A100 #1"),
        _Fake("A100 #0 OpenGL", backend_type="OpenGL"),
        _Fake("A100 #1 OpenGL", backend_type="OpenGL"),
    ]
    assert _pick_adapter(adapters, 1).summary == "A100 #1"
    with pytest.raises(ValueError, match="out of range"):
        _pick_adapter(adapters, 2)


def test_hyphenated_adapter_type_is_also_discrete():
    adapters = [_Fake("llvmpipe (CPU)", adapter_type="CPU"),
                _Fake("A100", adapter_type="discrete-gpu")]
    assert _pick_adapter(adapters, 0).summary == "A100"


def test_no_discrete_gpu_is_an_error_not_a_cpu_fallback():
    """Silently rendering a dataset on llvmpipe would cost hours, not minutes."""
    adapters = [_Fake("llvmpipe (CPU)", adapter_type="CPU")]
    with pytest.raises(ValueError, match="no discrete GPU adapter"):
        _pick_adapter(adapters, 0)


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


def test_reselecting_the_same_gpu_is_a_noop(monkeypatch):
    """A run rendering several splits asks once per split, and by the second
    one a renderer exists -- which is exactly when pygfx refuses to select."""
    seen = []
    monkeypatch.setattr("owm_envs.render.device._select", seen.append)
    select_gpu(1)
    select_gpu(1)
    assert seen == [1]


def test_selecting_a_different_gpu_is_not_suppressed(monkeypatch):
    # Switching cards mid-process must reach pygfx and raise there, not be
    # silently swallowed into rendering on the first card.
    seen = []
    monkeypatch.setattr("owm_envs.render.device._select", seen.append)
    select_gpu(0)
    select_gpu(1)
    assert seen == [0, 1]


def test_unparseable_env_var_names_the_variable(monkeypatch):
    monkeypatch.setenv("OWM_ENVS_GPU_INDEX", "first")
    with pytest.raises(ValueError, match="OWM_ENVS_GPU_INDEX"):
        select_gpu(None)
