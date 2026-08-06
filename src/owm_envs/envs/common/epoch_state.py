"""Absolute time as two state elements, aligned with astrojax Epoch internals.

prefix[0] = Julian Day number (integer-valued; exact in f32 -- JD ~2.46e6 is
far below 2**24). prefix[1] = seconds of day in [0, 86400) (~8 ms f32 grain).
A single float32 MJD element was rejected in the design: ~674 s resolution
at current dates, coarser than an eighth of an orbit; a bare f32 JD is worse
still (ulp ~0.25 day) and cannot advance at all at sub-daily dt. The
precision story lives in this two-element split, not in the advance step.

`advance_epoch_state` upcasts `sec` to f64, adds `dt`, and casts back to the
prefix dtype: a single f32 add per step is already correctly rounded, so
this is bit-identical to adding directly in f32 for this add-then-cast-back
shape -- the per-step f32 quantization of seconds-of-day is a benign ~8 ms
random walk either way. The f64 upcast is kept because it is free and
future-proofs any multi-op epoch arithmetic added later.
"""

from __future__ import annotations

import jax.numpy as jnp
from astrojax import config as astrojax_config
from astrojax.epoch import Epoch

EPOCH_DIM = 2
EPOCH_LABELS: tuple[str, ...] = ("epoch_jd_day", "epoch_sec_of_day_s")

_SECONDS_PER_DAY = 86400.0


def epoch_prefix(epoch: Epoch, dtype=jnp.float32) -> jnp.ndarray:
    return jnp.asarray(
        [jnp.asarray(epoch._jd, jnp.float64), jnp.asarray(epoch._seconds, jnp.float64)],
        dtype=dtype,
    )


def advance_epoch_state(prefix: jnp.ndarray, dt: float) -> jnp.ndarray:
    jd = prefix[..., 0].astype(jnp.float64)
    sec = prefix[..., 1].astype(jnp.float64) + jnp.float64(dt)
    days, sec = jnp.floor_divide(sec, _SECONDS_PER_DAY), jnp.mod(sec, _SECONDS_PER_DAY)
    return jnp.stack([jd + days, sec], axis=-1).astype(prefix.dtype)


def epoch_from_prefix(prefix: jnp.ndarray) -> Epoch:
    jd = prefix[..., 0].astype(jnp.int32)
    # astrojax's own float dtype (astrojax.config.get_dtype(), default f32)
    # is independent of this package's jax_enable_x64 flag. _seconds and
    # _kahan_c must match it: Epoch._from_internal performs no casting, and
    # a float64 leaf here silently violates that invariant -- it breaks
    # outright under jax_numpy_dtype_promotion="strict" (TypePromotionError
    # in Epoch's arithmetic/equality/jd()/gmst()) and otherwise mixes
    # dtypes without gaining precision, since the value was already
    # rounded to astrojax's dtype on the way in.
    seconds = prefix[..., 1].astype(astrojax_config.get_dtype())
    # Epoch._from_internal is the pytree-unflatten constructor path: private,
    # but the intended way to build an Epoch from raw split components. A
    # candidate to become a public astrojax constructor upstream.
    return Epoch._from_internal(jd, seconds, jnp.zeros_like(seconds))
