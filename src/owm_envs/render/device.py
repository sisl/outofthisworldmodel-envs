"""Select which GPU adapter pygfx renders on.

Identical GPUs in one host expose the identical adapter name, so wgpu's
name-based selection cannot tell them apart; selection has to go by
enumeration index. Vulkan also ignores CUDA_VISIBLE_DEVICES, so that is not
an escape hatch either.

Must run before the first renderer or scene is built: pygfx creates one shared
wgpu device per process on first use and refuses to rebuild it, so
`select_adapter` raises RuntimeError once that device exists.

pygfx does not promise that enumeration order matches nvidia-smi order, and
wgpu exposes no PCI/bus id to check it against -- identical cards report the
same vendor_id and device_id, which is why the index is the only
discriminator. Verify the mapping on any new host before trusting it: render
at one index at a time and confirm the allocation lands on the card
nvidia-smi numbers the same way, leaving the others idle.

Against wgpu-py 0.27.0 / pygfx 0.15.2 the API is
`wgpu.gpu.enumerate_adapters_sync()` and
`pygfx.renderers.wgpu.select_adapter(adapter)`, and `AdapterInfo["adapter_type"]`
reads "DiscreteGPU" / "CPU" / "Unknown" -- note the CamelCase, not the
hyphenated WebGPU spelling.
"""

from __future__ import annotations

import os

ENV_VAR = "OWM_ENVS_GPU_INDEX"

# Which GPU this process is already pinned to, so a repeat request for the
# same one can be skipped -- pygfx refuses `select_adapter` once a renderer
# exists, and a run that renders several splits asks once per split.
_SELECTED: int | None = None


def _is_discrete(adapter) -> bool:
    """Whether an adapter is a real GPU, spelling-insensitively.

    wgpu reports "DiscreteGPU" but the WebGPU spec spells the same value
    "discrete-gpu", so normalise rather than trust one of them.
    """
    kind = str(adapter.info.get("adapter_type", "")).lower().replace("-", "")
    return kind == "discretegpu"


def _pick_adapter(adapters: list, index: int):
    """Return the index-th discrete GPU, counting only discrete adapters.

    Enumeration lists the CPU rasteriser and a second, OpenGL view of each
    card alongside the Vulkan ones, so raw enumeration order would not line up
    with the physical GPUs a caller means. Keeping a single backend is what
    makes the count physical: wgpu emits Vulkan first, then Metal, D3D12 and
    OpenGL, so the first discrete adapter's backend is the primary one, and
    restricting to it drops the duplicate views of cards already counted.
    """
    discrete = [a for a in adapters if _is_discrete(a)]
    if not discrete:
        seen = ", ".join(a.summary for a in adapters) or "none"
        raise ValueError(
            f"gpu index {index} requested but no discrete GPU adapter is available; "
            f"enumerated adapters: {seen}"
        )
    backend = discrete[0].info.get("backend_type")
    pool = [a for a in discrete if a.info.get("backend_type") == backend]
    if not 0 <= index < len(pool):
        names = ", ".join(f"{i}: {a.summary}" for i, a in enumerate(pool))
        raise ValueError(f"gpu index {index} out of range; available adapters: {names}")
    return pool[index]


def _enumerate() -> list:
    import wgpu

    return wgpu.gpu.enumerate_adapters_sync()


def _select(index: int) -> None:
    from pygfx.renderers import wgpu as pygfx_wgpu

    pygfx_wgpu.select_adapter(_pick_adapter(_enumerate(), index))


def resolve_gpu_index(index: int | None) -> int | None:
    """The requested GPU: the argument, else the env var, else none at all.

    Public because the renderer is no longer the only consumer: the rollout's
    JAX device is chosen from the same request, and it has to be chosen from
    the same rule, or a run that says which GPU to use through the environment
    rather than the flag would pin only half of itself.
    """
    if index is not None:
        return index
    raw = os.environ.get(ENV_VAR)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{ENV_VAR} must be an integer, got {raw!r}") from None


def select_gpu(index: int | None = None) -> None:
    """Pin rendering to one GPU, from `index` or `OWM_ENVS_GPU_INDEX`.

    A no-op when neither is set, which leaves wgpu its own default choice,
    and a no-op when this process is already pinned to that same GPU: a
    caller rendering several splits in a row asks once per split, and by the
    second one a renderer exists, which is exactly what pygfx refuses to
    re-select under. Asking for a DIFFERENT GPU still goes through, so
    switching cards mid-process raises pygfx's error rather than silently
    rendering on the first one.
    """
    global _SELECTED
    index = resolve_gpu_index(index)
    if index is None or index == _SELECTED:
        return
    _select(index)
    _SELECTED = index


def check_gpu_index(index: int | None = None) -> None:
    """Raise if `index` names no available GPU, without pinning this process.

    For the caller that renders in worker processes: those select their own
    adapter, so the parent learns nothing about a mistyped index until the
    pool starts -- which is after the rollout, an hour it should not have to
    spend to find out. Enumeration answers the question on its own; only
    `select_adapter` pins the process, and this deliberately does not call it,
    so the workers still get their pick of the card.
    """
    index = resolve_gpu_index(index)
    if index is None:
        return
    _pick_adapter(_enumerate(), index)
