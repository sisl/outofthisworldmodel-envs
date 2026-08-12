"""JAX settings this package fixes for itself: precision, and which GPU.

Precision is settled on import, because every jax-consuming path needs it and
none of them may disagree. The GPU is settled by a call, because it is the
operator's choice per run -- see `pin_gpu`, and `_entry.py` for why it has to
be made before this package's own imports reach astrojax.

float64 must be real: the [jd, seconds-of-day] epoch prefix is carried in
f64 because an f32 seconds-of-day biases the advance by hundreds of
seconds per orbit at small dt (see `envs/common/epoch_state.py`),
`envs/common/zonal_gravity.py` evaluates the zonal harmonics in f64, and
the iss-numerical orbit propagator integrates in f64. All f32 paths pin
their dtypes explicitly, so flipping this changes nothing for them --
enforced by the golden-rollout tests.

Imported by the two jax chokepoints (`core`, `envs.common`) rather than
the package root, which must stay importable without jax so the driver
seam can host non-JAX backends.
"""

import os

import jax
from jax._src import xla_bridge

jax.config.update("jax_enable_x64", True)


def pin_gpu(index: int | None) -> None:
    """Confine JAX to one GPU, and leave the rest of that card to the renderer.

    XLA's allocator claims 75% of every card it initialises, whatever the
    rollout needs -- tens of gigabytes for a workload that peaks near 1.5 GiB.
    Where the GPUs are shared, that grab fails outright once a neighbour holds
    enough of the card, and it fails as a backoff loop whose outcome depends on
    how much the neighbour happens to hold that second, so the same command
    survives one minute and dies the next. Growing on demand costs a little
    allocator work and removes the failure.

    `index` is the same GPU `--gpu-index` hands the renderer, so one flag names
    one card for the whole run. Only the CUDA half is steered here: wgpu
    reaches the GPU through Vulkan, which ignores CUDA_VISIBLE_DEVICES
    entirely and is pointed at its adapter by `render/device.py` instead.
    CUDA_DEVICE_ORDER makes CUDA count the way the operator does -- along the
    bus, as nvidia-smi numbers them, rather than by capability. That puts the
    CUDA half on the same numbering `render/device.py` has operators verify the
    Vulkan half against; it does not on its own establish that the two
    enumerations agree, which is a hardware property and that module documents
    how to check it. Both are set from the request rather than deferring to an
    inherited value, because an ambient CUDA_VISIBLE_DEVICES would otherwise
    quietly overrule an explicit one and put the rollout back on the card the
    operator asked it to leave. The cost of that choice is that a scheduler
    which allocates GPUs by presetting CUDA_VISIBLE_DEVICES is overruled too,
    so on such a host the flag should be left unset and the allocation left to
    speak for itself.

    XLA reads all of this from the environment when the backend first
    initialises, so this has to run before that -- which here means before the
    imports that reach astrojax, because astrojax initialises a backend as it
    is imported. `_entry.py` is what holds that ordering, and the check below
    is what keeps it honest: too late is fatal, not quiet. Settings cannot be
    applied to a backend that is already up, so a run that got here late would
    hold most of a shared card and compute on a GPU it was not sent to, and
    would do both without saying so. Refusing turns a silent reordering into
    an error at the one point that can still see it.
    """
    if xla_bridge.backends_are_initialized():
        raise RuntimeError(
            "GPU pinning came too late to take effect: JAX has already "
            "initialised its backend, which is when XLA reads the device and "
            "allocation settings. Something imported a jax-consuming module "
            "before this ran, so the process would keep whatever device it "
            "started on and hold up to 75% of that card. Pin before the first "
            "import that reaches astrojax; see owm_envs._entry."
        )
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if index is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
