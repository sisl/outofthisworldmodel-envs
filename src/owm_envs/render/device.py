"""Select which GPU adapter pygfx renders on.

Both A100s on a dual-GPU host expose the identical adapter name ("NVIDIA A100
80GB PCIe (DiscreteGPU) via Vulkan"), so wgpu's name-based selection cannot
tell them apart; selection has to go by enumeration index. Vulkan also ignores
CUDA_VISIBLE_DEVICES, so that is not an escape hatch either.

Must run before the first renderer or scene is built: pygfx creates one shared
wgpu device per process on first use and refuses to rebuild it, so
`select_adapter` raises RuntimeError once that device exists.

Against wgpu-py 0.27.0 / pygfx 0.15.2 the API is
`wgpu.gpu.enumerate_adapters_sync()` and
`pygfx.renderers.wgpu.select_adapter(adapter)`, and `AdapterInfo["adapter_type"]`
reads "DiscreteGPU" / "CPU" / "Unknown" -- note the CamelCase, not the
hyphenated WebGPU spelling.
"""

from __future__ import annotations

import os

ENV_VAR = "OWM_ENVS_GPU_INDEX"


def _is_discrete(adapter) -> bool:
    """Whether an adapter is a real GPU, spelling-insensitively.

    wgpu reports "DiscreteGPU" but the WebGPU spec spells the same value
    "discrete-gpu", so normalise rather than trust one of them.
    """
    kind = str(adapter.info.get("adapter_type", "")).lower().replace("-", "")
    return kind == "discretegpu"


def _pick_adapter(adapters: list, index: int):
    """Return the index-th discrete GPU, counting only discrete adapters.

    Enumeration interleaves the CPU rasteriser and duplicate OpenGL views of
    the same cards, so indexing raw enumeration order would not line up with
    the physical GPUs a caller means.
    """
    discrete = [a for a in adapters if _is_discrete(a)]
    pool = discrete or list(adapters)
    if not 0 <= index < len(pool):
        names = ", ".join(f"{i}: {a.summary}" for i, a in enumerate(pool))
        raise ValueError(f"gpu index {index} out of range; available adapters: {names}")
    return pool[index]


def _select(index: int) -> None:
    import wgpu
    from pygfx.renderers import wgpu as pygfx_wgpu

    pygfx_wgpu.select_adapter(_pick_adapter(wgpu.gpu.enumerate_adapters_sync(), index))


def select_gpu(index: int | None = None) -> None:
    """Pin rendering to one GPU, from `index` or `OWM_ENVS_GPU_INDEX`.

    A no-op when neither is set, which leaves wgpu its own default choice.
    """
    if index is None:
        raw = os.environ.get(ENV_VAR)
        if raw is None:
            return
        try:
            index = int(raw)
        except ValueError:
            raise ValueError(f"{ENV_VAR} must be an integer, got {raw!r}") from None
    _select(index)
