"""Absolute time as two state elements, aligned with astrojax Epoch internals.

prefix[0] = Julian Day number (integer-valued; exact even in f32 -- JD ~2.46e6
is far below 2**24). prefix[1] = seconds of day in [0, 86400).

The two-element split is what buys the precision, and it beats any single
float: an f32 MJD element resolves only ~674 s at current dates, coarser than
an eighth of an orbit, and a bare f32 JD is worse still (ulp ~0.25 day) --
it cannot advance at all at sub-daily dt.

The split alone is not sufficient, though, which is why the prefix defaults to
f64. Carrying `sec` in f32 is safe only while dt stays large relative to the
f32 ulp of seconds-of-day -- 3.9 ms below 65536 s, 7.8 ms above. When dt is
not an exact multiple of that ulp, every add rounds the same way within a
binade, so the error is a systematic bias rather than the benign random walk
a rounding argument would suggest. At dt = 0.05 s from 2026-08-01T06:00:00Z
the f32 prefix loses 290 s over one ISS orbit: dt is 12.8 ulps wide below the
65536 s binade (rounds up, +0.78 ms/step) and 6.4 ulps above it (rounds down,
-3.1 ms/step), and the second regime covers most of the orbit. f64 has ~1e-11
grain at these magnitudes and no such regime. Consumers cast to f32 only at
observation and record boundaries, where a single rounding is harmless.

`advance_epoch_state` is deliberately dtype-preserving: it upcasts `sec` to
f64, adds, and casts back to the prefix dtype, so an f32 prefix still behaves
exactly like f32 arithmetic (a single f32 add is already correctly rounded,
making this bit-identical to adding in f32 for this add-then-cast-back shape).
That keeps the f32 counterfactual measurable in tests, and the f64 upcast is
free and future-proofs any multi-op epoch arithmetic added later.
"""

from __future__ import annotations

import jax.numpy as jnp
from astrojax import config as astrojax_config
from astrojax.epoch import Epoch

EPOCH_DIM = 2
EPOCH_LABELS: tuple[str, ...] = ("epoch_jd_day", "epoch_sec_of_day_s")

_SECONDS_PER_DAY = 86400.0


def epoch_prefix(epoch: Epoch, dtype=jnp.float64) -> jnp.ndarray:
    """Split `epoch` into [jd_day, sec_of_day]. Scalar `Epoch` only; returns
    (2,). Defaults to f64 -- see the module docstring for why f32 carriage
    biases the advance."""
    return jnp.asarray(
        [jnp.asarray(epoch._jd, jnp.float64), jnp.asarray(epoch._seconds, jnp.float64)],
        dtype=dtype,
    )


def advance_epoch_state(prefix: jnp.ndarray, dt: float | jnp.ndarray) -> jnp.ndarray:
    jd = prefix[..., 0].astype(jnp.float64)
    sec = prefix[..., 1].astype(jnp.float64) + jnp.float64(dt)
    days, sec = jnp.floor_divide(sec, _SECONDS_PER_DAY), jnp.mod(sec, _SECONDS_PER_DAY)
    return jnp.stack([jd + days, sec], axis=-1).astype(prefix.dtype)


def seconds_between(prefix_a: jnp.ndarray, prefix_b: jnp.ndarray) -> jnp.ndarray:
    """Signed seconds from `prefix_b` to `prefix_a` (a minus b), always in
    f64 regardless of the input dtypes. The day and seconds-of-day terms are
    differenced separately so the large, exactly-representable JD number
    cancels before it can swamp the seconds term."""
    a = prefix_a.astype(jnp.float64)
    b = prefix_b.astype(jnp.float64)
    days = a[..., 0] - b[..., 0]
    return days * _SECONDS_PER_DAY + (a[..., 1] - b[..., 1])


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
