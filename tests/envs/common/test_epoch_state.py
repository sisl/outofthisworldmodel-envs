import jax
import jax.numpy as jnp
import numpy as np
from astrojax.epoch import Epoch

from owm_envs.envs.common.epoch_state import (
    EPOCH_DIM, advance_epoch_state, epoch_from_prefix, epoch_prefix,
    seconds_between,
)

# One ISS orbit at the numerical env's integrator step. The start epoch is
# deliberately not noon UTC: astrojax's internal day boundary is at noon, so
# 06:00Z puts seconds-of-day at 64800, high in the f32 binade where the
# quantization bias bites (see epoch_state's module docstring).
_DT = 0.05
_STEPS = 110_800
_START = "2026-08-01T06:00:00Z"


def _scan_elapsed(dtype) -> float:
    """Total seconds actually accumulated by `_STEPS` advances, measured in
    f64 from the prefix itself so the measurement cannot mask the drift."""
    p0 = epoch_prefix(Epoch(_START), dtype=dtype)
    end, _ = jax.lax.scan(
        lambda q, _: (advance_epoch_state(q, _DT), None), p0, None, length=_STEPS
    )
    return float(seconds_between(end, p0))


def test_prefix_roundtrip():
    e = Epoch("2026-08-01T12:34:56Z")
    p = epoch_prefix(e)
    assert p.shape == (EPOCH_DIM,)
    e2 = epoch_from_prefix(p)
    # The prefix is f64, but epoch_from_prefix narrows seconds to astrojax's
    # own dtype (f32 by default), whose sec-of-day grain is ~8 ms.
    assert abs(float(e2 - e)) < 0.01


def test_advance_rolls_over_day_boundary():
    # astrojax's internal Epoch day boundary follows the Julian Day
    # convention and falls at noon UTC, not calendar midnight: seconds-of-day
    # is ~0 at 12:00:00Z and ~86400 just before it. 11:59:59Z is one second
    # from that internal rollover.
    e = Epoch("2026-08-01T11:59:59Z")
    p = epoch_prefix(e)
    p2 = advance_epoch_state(p, 2.0)
    assert float(p2[0]) == float(p[0]) + 1.0
    assert 0.0 <= float(p2[1]) < 2.0


def test_advance_drift_bounded_over_one_orbit():
    """One ISS orbit of dt=0.05 steps on the f64 default prefix accumulates
    the elapsed time essentially exactly. The measured residual is 3e-7 s --
    consistent with plain f64 accumulation rounding at a ~6.5e4 s magnitude
    (ulp ~1.5e-11) -- so the bound is set just loose enough to absorb
    platform variation while staying six orders below the f32 failure."""
    assert abs(_scan_elapsed(jnp.float64) - _STEPS * _DT) < 1e-4


def test_f32_prefix_really_drifts():
    """Counterfactual pinning why the prefix defaults to f64: the same scan
    on an f32 prefix LOSES ~290 s over the orbit. dt=0.05 is 12.8 f32 ulps
    below the 65536 s binade and 6.4 above it, so every add rounds the same
    direction within a binade -- the error is a systematic bias, not a random
    walk, which is exactly why it is signed and reproducible rather than
    merely large. Both sign and magnitude are pinned: if a future change
    makes f32 exact, or flips the drift to a gain, the f64 default and the
    module docstring's explanation both need revisiting."""
    error = _scan_elapsed(jnp.float32) - _STEPS * _DT
    assert -350.0 < error < -250.0  # measured: -289.63 s, deterministic


def test_advance_is_traceable_and_dtype_preserving():
    p = epoch_prefix(Epoch("2026-08-01T00:00:00Z"))
    assert p.dtype == jnp.float64  # the default
    step = jax.jit(lambda q: advance_epoch_state(q, 0.1))
    assert step(p).dtype == jnp.float64
    p32 = epoch_prefix(Epoch("2026-08-01T00:00:00Z"), dtype=jnp.float32)
    assert step(p32).dtype == jnp.float32


def test_seconds_between_same_day():
    e = Epoch("2026-08-01T00:00:00Z")
    a = advance_epoch_state(epoch_prefix(e), 137.25)
    assert np.isclose(float(seconds_between(a, epoch_prefix(e))), 137.25, atol=1e-9)


def test_seconds_between_crosses_day_boundary():
    # 11:59:59Z is one second from astrojax's internal (noon) rollover, so
    # this difference spans a jd increment and must not be confused by it.
    b = epoch_prefix(Epoch("2026-08-01T11:59:59Z"))
    a = advance_epoch_state(b, 5.0)
    assert float(a[0]) == float(b[0]) + 1.0
    assert np.isclose(float(seconds_between(a, b)), 5.0, atol=1e-9)


def test_seconds_between_is_signed():
    b = epoch_prefix(Epoch("2026-08-01T11:59:59Z"))
    a = advance_epoch_state(b, 5.0)
    assert np.isclose(float(seconds_between(b, a)), -5.0, atol=1e-9)


def test_seconds_between_is_f64_even_for_f32_inputs():
    e = Epoch("2026-08-01T06:00:00Z")
    p = epoch_prefix(e, dtype=jnp.float32)
    assert seconds_between(advance_epoch_state(p, 1.0), p).dtype == jnp.float64
