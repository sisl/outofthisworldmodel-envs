import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from owm_envs.envs.common.epoch_state import advance_epoch_state, epoch_prefix
from owm_envs.envs.common.orbit import ReferenceOrbit
from owm_envs.envs.iss_numerical.config import NumericalConfig
from owm_envs.envs.iss_numerical.dynamics import chaser_state_from_view
from owm_envs.envs.iss_numerical.render_adapter import make_render_adapter

# The default epoch (2026-08-01T00:00:00Z) with the default ISS-like orbit
# already crosses both edges of the eclipse within one orbital period
# (checked directly against ReferenceOrbit/illumination in iss_hcw's own
# suite), so no separate probe epoch is needed for the sweep test below.
CFG = NumericalConfig(max_range_m=None, dock={"enabled": False}, physics={"collision_boxes_path": []})

# A chaser colocated with the chief and co-rotating with the world frame --
# the fixed point `test_view.py` pins -- so `relative_view` reports zeros and
# the identity quaternion regardless of where the chief itself is. Only the
# lighting fields below depend on the chief/epoch this gets paired with.
_IDENTITY_VIEW = jnp.zeros(13, jnp.float64).at[6].set(1.0)


def _state_at(prefix: np.ndarray, chief: np.ndarray) -> np.ndarray:
    chief = jnp.asarray(chief, jnp.float64)
    chaser = chaser_state_from_view(chief, _IDENTITY_VIEW)
    return np.asarray(jnp.concatenate([jnp.asarray(prefix, jnp.float64), chief, chaser]))


def _epoch0_state() -> tuple[np.ndarray, ReferenceOrbit]:
    ref = ReferenceOrbit(CFG.orbit)
    prefix = epoch_prefix(ref.epoch0)
    chief = ref.chief_state_eci(0.0)
    return _state_at(prefix, chief), ref


def test_illumination_is_in_unit_range():
    adapter = make_render_adapter(CFG)
    state, _ = _epoch0_state()
    inputs = adapter(state, None)
    assert 0.0 <= inputs.lighting.illumination <= 1.0


def test_sun_direction_is_unit():
    adapter = make_render_adapter(CFG)
    state, _ = _epoch0_state()
    inputs = adapter(state, None)
    assert np.isclose(np.linalg.norm(inputs.lighting.sun_direction_world), 1.0, atol=1e-5)


def test_moon_distance_is_realistic():
    adapter = make_render_adapter(CFG)
    state, _ = _epoch0_state()
    inputs = adapter(state, None)
    distance = np.linalg.norm(inputs.lighting.moon_vector_world)
    assert 3.4e8 < distance < 4.2e8


def test_chief_distance_matches_the_state_columns_exactly():
    # Unlike iss_hcw, the chief here is not reconstructed from an analytic
    # orbit -- it is read straight out of the state -- so this can assert
    # exact agreement with an ARBITRARY chief position rather than a physical
    # bound on a real orbit's radius.
    adapter = make_render_adapter(CFG)
    ref = ReferenceOrbit(CFG.orbit)
    prefix = epoch_prefix(ref.epoch0)
    chief = jnp.array([1.0e7, 2.0e6, -3.0e6, 10.0, -20.0, 30.0], dtype=jnp.float64)
    state = _state_at(prefix, chief)

    inputs = adapter(state, None)
    expected = float(np.linalg.norm(np.asarray(chief[0:3])))
    assert inputs.lighting.chief_distance_m == pytest.approx(expected, rel=1e-6)


def test_position_and_quaternion_come_from_the_derived_relative_view():
    # The chaser is colocated with and co-rotating with the chief, so the
    # relative view -- and so the posed frame -- is zero/identity regardless
    # of the chief's own (arbitrary) position.
    adapter = make_render_adapter(CFG)
    ref = ReferenceOrbit(CFG.orbit)
    prefix = epoch_prefix(ref.epoch0)
    chief = jnp.array([1.0e7, 2.0e6, -3.0e6, 10.0, -20.0, 30.0], dtype=jnp.float64)
    state = _state_at(prefix, chief)

    inputs = adapter(state, None)
    np.testing.assert_allclose(inputs.position_world, [0.0, 0.0, 0.0], atol=1e-4)
    np.testing.assert_allclose(inputs.quaternion_bw, [1.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_eclipse_sweep_visits_both_extremes_and_stays_geometrically_consistent():
    """Sweep one orbital period at 60 s spacing: an ISS-inclination LEO orbit
    passes through both full sunlight and umbra, and whenever the adapter
    reports near-total eclipse the sun direction must point away from the
    chief's radial (+z world, "up" away from Earth) -- the sun sitting
    geometrically behind the Earth, agreeing with the photometric story."""
    adapter = make_render_adapter(CFG)
    ref = ReferenceOrbit(CFG.orbit)
    epoch0_prefix = epoch_prefix(ref.epoch0)
    period = 2.0 * np.pi / ref.mean_motion

    illuminations = []
    eclipse_dots = []
    t = 0.0
    while t < period:
        prefix = advance_epoch_state(epoch0_prefix, t)
        chief = ref.chief_state_eci(t)
        inputs = adapter(_state_at(prefix, chief), None)
        illuminations.append(inputs.lighting.illumination)
        if inputs.lighting.illumination < 0.01:
            eclipse_dots.append(float(np.dot(inputs.lighting.sun_direction_world, [0.0, 0.0, 1.0])))
        t += 60.0

    illuminations = np.array(illuminations)
    assert (illuminations > 0.99).any(), "expected at least one fully-lit sample"
    assert (illuminations < 0.01).any(), "expected at least one eclipsed sample"
    assert eclipse_dots, "expected at least one sample below the eclipse threshold"
    assert all(d < 0.0 for d in eclipse_dots)


def test_the_per_frame_ephemeris_is_compiled():
    """Called op by op, the sun/moon/eclipse math below is uncompiled JAX
    dispatch over a handful of scalars -- see the module docstring. Nothing
    here times anything -- a timing assertion is flaky on a shared machine --
    only that the kernel every frame goes through is a jitted one."""
    adapter = make_render_adapter(CFG)
    assert isinstance(adapter._kernel, jax.stages.Wrapped)


def test_adapter_pickle_round_trip():
    adapter = make_render_adapter(CFG)
    restored = pickle.loads(pickle.dumps(adapter))

    # The jitted callable itself never crosses the boundary: the worker jits
    # its own from what was pickled, which is what keeps the adapter sendable.
    assert isinstance(restored._kernel, jax.stages.Wrapped)
    assert restored._kernel is not adapter._kernel

    state, _ = _epoch0_state()
    original = adapter(state, None)
    round_tripped = restored(state, None)

    np.testing.assert_array_equal(original.position_world, round_tripped.position_world)
    np.testing.assert_array_equal(original.quaternion_bw, round_tripped.quaternion_bw)
    assert original.lighting.illumination == pytest.approx(round_tripped.lighting.illumination)
    np.testing.assert_array_equal(
        original.lighting.sun_direction_world, round_tripped.lighting.sun_direction_world
    )


def test_adapter_output_is_plain_numpy():
    adapter = make_render_adapter(CFG)
    state, _ = _epoch0_state()
    inputs = adapter(state, None)

    assert isinstance(inputs.position_world, np.ndarray)
    assert isinstance(inputs.quaternion_bw, np.ndarray)
    assert isinstance(inputs.lighting.sun_direction_world, np.ndarray)
    assert isinstance(inputs.lighting.moon_vector_world, np.ndarray)
    assert isinstance(inputs.lighting.illumination, float)
    assert isinstance(inputs.lighting.chief_distance_m, float)
    for arr in (
        inputs.position_world,
        inputs.quaternion_bw,
        inputs.lighting.sun_direction_world,
        inputs.lighting.moon_vector_world,
    ):
        assert not isinstance(arr, jax.Array)


def test_adapter_accepts_a_plain_numpy_input():
    adapter = make_render_adapter(CFG)
    state, _ = _epoch0_state()
    assert isinstance(state, np.ndarray)
    inputs = adapter(state, None)
    assert inputs.lighting is not None
