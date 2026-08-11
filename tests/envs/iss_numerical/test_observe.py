"""The four observation modes.

`make_observe` is where this env decides what frame a recorded observation
reports the pair in; the state behind it is always the full 21D one. These
tests pin each mode's block layout against the state slices it is built from,
that every mode is derived at the state's own f64 width and narrowed to f32
only at the end, and that the emitted widths are the ones `OBS_MODE_DIM`
advertises.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from owm_envs.envs.iss_numerical.config import NUM_LAYOUT, OBS_MODE_DIM, NumericalConfig
from owm_envs.envs.iss_numerical.dynamics import NumericalDynamics, relative_view
from owm_envs.envs.iss_numerical.observe import make_observe

MODES = tuple(OBS_MODE_DIM)

# Non-degenerate on every channel: a standoff well inside the domain, real
# start motion, and an attitude error large enough that q_bw is nowhere near
# the identity.
WIDE_DISPERSION = {
    "start_radius_range_m": (50.0, 200.0),
    "start_speed_max_m_s": 1.0,
    "start_rate_max_rad_s": 0.05,
    "start_attitude_error_max_deg": 60.0,
    "epoch_offset_range_s": (0.0, 3000.0),
}


def _cfg(mode="relative", **overrides) -> NumericalConfig:
    physics = {"collision_boxes_path": [], "linear_damping": 0.0, "angular_damping": 0.0}
    physics.update(overrides.pop("physics", {}))
    return NumericalConfig(
        max_range_m=None,
        dock={"enabled": False},
        physics=physics,
        orbit=WIDE_DISPERSION,
        observation={"mode": mode},
        **overrides,
    )


def _state(seed=0) -> jnp.ndarray:
    return NumericalDynamics(_cfg()).reset(jax.random.PRNGKey(seed))


def _f32(x) -> np.ndarray:
    return np.asarray(x, np.float32)


def test_the_modes_index_literals_are_the_layouts_own_slices():
    """`observe.py` slices the state with bare literals (`state[2:8]`,
    `state[8:14]`, ...) and every test below checks a mode against those same
    literals. That pair agrees with itself no matter where `NUM_LAYOUT` puts
    a segment, so this is the assertion that ties both to the declaration the
    rest of the env reads -- a layout change that leaves the literals behind
    fails here instead of silently emitting mislabelled observations.
    """
    assert NUM_LAYOUT.state_dim == 21
    assert NUM_LAYOUT.epoch == slice(0, 2)
    assert NUM_LAYOUT.chief == slice(2, 8)
    # The chaser's own ECI block, which `_chaser_absolute` moves to the front
    # as `state[8:14]` -- contiguous by StateLayout's own invariant.
    assert NUM_LAYOUT.pos == slice(8, 11)
    assert NUM_LAYOUT.vel == slice(11, 14)
    # `state[14:21]`, the attitude and rate tail every absolute mode copies
    # through unchanged.
    assert NUM_LAYOUT.quat == slice(14, 18)
    assert NUM_LAYOUT.omega == slice(18, 21)


def test_every_declared_mode_emits_its_advertised_width():
    """`OBS_MODE_DIM` is the public width table and the dispatch is the code
    behind it; a mode in one and not the other is a config value that loads
    and then fails at the first observation."""
    observed = {mode: np.asarray(make_observe(_cfg(mode))(_state())).shape[0] for mode in MODES}
    assert observed == OBS_MODE_DIM


def test_each_mode_narrows_to_f32_at_the_boundary():
    state = _state()
    for mode in MODES:
        assert make_observe(_cfg(mode))(state).dtype == jnp.float32, mode


def test_absolute_mode_is_the_state_itself():
    state = _state()
    obs = make_observe(_cfg("absolute"))(state)
    np.testing.assert_array_equal(np.asarray(obs), _f32(state))


def test_relative_mode_is_the_epoch_prefix_and_the_canonical_view():
    """The 15D layout `iss_hcw` carries in-state: [epoch(2) | view(13)],
    q_bw included, so a consumer reading either env's observation reads the
    same elements in the same order."""
    state = _state()
    obs = np.asarray(make_observe(_cfg("relative"))(state))
    np.testing.assert_array_equal(obs[0:2], _f32(state[0:2]))
    np.testing.assert_array_equal(obs[2:15], _f32(relative_view(state)))


def test_chaser_absolute_mode_reports_the_chief_measured_from_the_chaser():
    state = _state()
    obs = np.asarray(make_observe(_cfg("chaser_absolute"))(state))
    s = np.asarray(state, np.float64)
    np.testing.assert_array_equal(obs[0:2], _f32(s[0:2]))
    np.testing.assert_array_equal(obs[2:8], _f32(s[8:14]))
    np.testing.assert_array_equal(obs[8:14], _f32(s[2:8] - s[8:14]))
    np.testing.assert_array_equal(obs[14:21], _f32(s[14:21]))


def test_chief_absolute_mode_reports_the_chaser_as_the_world_frame_offset():
    state = _state()
    obs = np.asarray(make_observe(_cfg("chief_absolute"))(state))
    s = np.asarray(state, np.float64)
    np.testing.assert_array_equal(obs[0:2], _f32(s[0:2]))
    np.testing.assert_array_equal(obs[2:8], _f32(s[2:8]))
    np.testing.assert_array_equal(obs[8:14], _f32(relative_view(state)[0:6]))
    np.testing.assert_array_equal(obs[14:21], _f32(s[14:21]))


@pytest.mark.parametrize(
    "mode, block", [("relative", slice(2, 5)), ("chaser_absolute", slice(8, 11))]
)
def test_absolute_positions_are_differenced_before_the_f32_narrowing(mode, block):
    """Both modes resolve a ~100 m separation out of two ~6.8e6 m ECI
    positions, where f32's grain is ~0.5 m. Narrowing first and differencing
    after would put a third of a metre of quantization on a channel the dock
    gate reads at 0.1 m, so the order of the two operations is the whole
    accuracy of the mode, not a rounding detail.
    """
    state = _state()
    obs = np.asarray(make_observe(_cfg(mode))(state))
    narrowed = jnp.asarray(_f32(state), jnp.float64)
    narrow_first = np.asarray(make_observe(_cfg(mode))(narrowed))
    assert np.linalg.norm(obs[block] - narrow_first[block]) > 0.01


def test_modes_are_jit_traceable():
    """The scan driver evaluates the observation inside `jax.lax.scan`.

    Agreement is to one f32 ulp, not bit-for-bit: XLA reassociates and fuses
    the f64 rotation the relative channels go through, so the traced and
    eager paths can land on adjacent f32 values (1.8e-7 relative, measured).
    The bound is stated rather than left to assert_allclose's default, which
    at 1e-7 sits below that and would fail intermittently.
    """
    state = _state()
    for mode in MODES:
        observe = make_observe(_cfg(mode))
        np.testing.assert_allclose(
            np.asarray(jax.jit(observe)(state)), np.asarray(observe(state)),
            rtol=1e-6, atol=1e-9, err_msg=mode,
        )
