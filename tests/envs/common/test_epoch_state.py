import jax
import jax.numpy as jnp
import numpy as np
from astrojax.epoch import Epoch

from owm_envs.envs.common.epoch_state import (
    EPOCH_DIM, advance_epoch_state, epoch_from_prefix, epoch_prefix,
)


def test_prefix_roundtrip():
    e = Epoch("2026-08-01T12:34:56Z")
    p = epoch_prefix(e)
    assert p.shape == (EPOCH_DIM,)
    e2 = epoch_from_prefix(p)
    assert abs(float(e2 - e)) < 0.01  # seconds; f32 sec-of-day is ~8 ms grained


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
    """5540 steps of 1.0 s from mid-day: pure-f32 accumulation would bias
    ~10 s; the f64 add keeps total error within the +/-4 ms quantization
    random walk (~0.3 s at 3 sigma)."""
    e = Epoch("2026-08-01T12:00:00Z")
    p = epoch_prefix(e)
    step = jax.jit(lambda q: advance_epoch_state(q, 1.0))
    for _ in range(5540):
        p = step(p)
    end = epoch_from_prefix(p)
    assert abs(float(end - e) - 5540.0) < 0.5


def test_advance_is_traceable_and_f32():
    p = epoch_prefix(Epoch("2026-08-01T00:00:00Z"))
    out = jax.jit(lambda q: advance_epoch_state(q, 0.1))(p)
    assert out.dtype == jnp.float32
