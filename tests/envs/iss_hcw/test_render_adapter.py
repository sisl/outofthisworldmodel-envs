import pickle

import jax
import numpy as np
import pytest

from owm_envs.envs.common.epoch_state import advance_epoch_state, epoch_prefix
from owm_envs.envs.common.orbit import ReferenceOrbit
from owm_envs.envs.iss_hcw.config import HCWConfig
from owm_envs.envs.iss_hcw.render_adapter import make_render_adapter

# The default epoch (2026-08-01T00:00:00Z) with the default ISS-like orbit
# already crosses both edges of the eclipse within one orbital period
# (checked directly against ReferenceOrbit/illumination), so no separate
# probe epoch is needed for the sweep test below.
CFG = HCWConfig()


def _state_at(prefix: np.ndarray) -> np.ndarray:
    """A 15D state row carrying the given epoch prefix. The adapter's
    lighting fields depend only on the prefix, so the rest of the view is
    arbitrary (zero) here."""
    return np.concatenate([np.asarray(prefix, dtype=np.float64), np.zeros(13, dtype=np.float64)])


def test_illumination_is_in_unit_range():
    adapter = make_render_adapter(CFG)
    ref = ReferenceOrbit(CFG.orbit)
    inputs = adapter(_state_at(epoch_prefix(ref.epoch0)), None)
    assert 0.0 <= inputs.lighting.illumination <= 1.0


def test_sun_direction_is_unit():
    adapter = make_render_adapter(CFG)
    ref = ReferenceOrbit(CFG.orbit)
    inputs = adapter(_state_at(epoch_prefix(ref.epoch0)), None)
    assert np.isclose(np.linalg.norm(inputs.lighting.sun_direction_world), 1.0, atol=1e-5)


def test_moon_distance_is_realistic():
    adapter = make_render_adapter(CFG)
    ref = ReferenceOrbit(CFG.orbit)
    inputs = adapter(_state_at(epoch_prefix(ref.epoch0)), None)
    distance = np.linalg.norm(inputs.lighting.moon_vector_world)
    assert 3.4e8 < distance < 4.2e8


def test_chief_distance_within_orbit_radius_bounds():
    adapter = make_render_adapter(CFG)
    ref = ReferenceOrbit(CFG.orbit)
    inputs = adapter(_state_at(epoch_prefix(ref.epoch0)), None)
    # Matches the module's documented ~4 m max reconstruction error, not
    # machine precision.
    margin = 10.0
    lo = CFG.orbit.sma_m * (1.0 - CFG.orbit.ecc) - margin
    hi = CFG.orbit.sma_m * (1.0 + CFG.orbit.ecc) + margin
    assert lo <= inputs.lighting.chief_distance_m <= hi


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
        inputs = adapter(_state_at(prefix), None)
        illuminations.append(inputs.lighting.illumination)
        if inputs.lighting.illumination < 0.01:
            eclipse_dots.append(float(np.dot(inputs.lighting.sun_direction_world, [0.0, 0.0, 1.0])))
        t += 60.0

    illuminations = np.array(illuminations)
    assert (illuminations > 0.99).any(), "expected at least one fully-lit sample"
    assert (illuminations < 0.01).any(), "expected at least one eclipsed sample"
    assert eclipse_dots, "expected at least one sample below the eclipse threshold"
    assert all(d < 0.0 for d in eclipse_dots)


def test_adapter_pickle_round_trip():
    adapter = make_render_adapter(CFG)
    restored = pickle.loads(pickle.dumps(adapter))

    ref = ReferenceOrbit(CFG.orbit)
    state = _state_at(epoch_prefix(ref.epoch0))
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
    ref = ReferenceOrbit(CFG.orbit)
    inputs = adapter(_state_at(epoch_prefix(ref.epoch0)), None)

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
