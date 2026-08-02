import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astrojax.attitude_dynamics import torque_gravity_gradient
from astrojax.relative_motion import hcw_stm

from owm_envs.core.quaternion import quat_conjugate, quat_to_rotmat
from owm_envs.envs.iss.config import (
    DockConfig,
    ISSConfig,
    PhysicsConfig,
    default_collision_boxes_path,
)
from owm_envs.envs.iss.dynamics import STATE_LABELS, ISSDynamics
from owm_envs.envs.iss.orbit import RTN_FROM_WORLD, OrbitConfig, ReferenceOrbit

ZERO_ACTION = jnp.zeros((6,), dtype=jnp.float32)


def make_state(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0),
               omega=(0.0, 0.0, 0.0)) -> jnp.ndarray:
    return jnp.asarray([*pos, *vel, *quat, *omega], dtype=jnp.float32)


def test_state_is_13_dimensional():
    dyn = ISSDynamics(ISSConfig())
    assert dyn.state_dim == 13
    assert dyn.action_dim == 6
    assert len(STATE_LABELS) == 13


def test_zero_action_from_rest_stays_at_rest():
    dyn = ISSDynamics(ISSConfig(
        physics=PhysicsConfig(collision_boxes_path=None), dock=DockConfig(enabled=False)
    ))
    s = make_state(pos=(50.0, 0.0, 0.0))
    s_next, _ = dyn.step(s, ZERO_ACTION)
    np.testing.assert_allclose(np.asarray(s_next), np.asarray(s), atol=1e-5)


def test_constant_velocity_translates():
    dyn = ISSDynamics(ISSConfig(
        physics=PhysicsConfig(collision_boxes_path=None), dock=DockConfig(enabled=False)
    ))
    s = make_state(pos=(50.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0))
    s_next, _ = dyn.step(s, ZERO_ACTION)
    # linear_damping defaults to 0.0 (vacuum, no medium to damp against), so
    # constant velocity over dt=0.05 covers exactly 0.05 m.
    assert np.isclose(float(s_next[0] - s[0]), 0.05, atol=1e-6)


def test_body_force_accelerates_along_body_axis():
    dyn = ISSDynamics(ISSConfig(
        physics=PhysicsConfig(collision_boxes_path=None), dock=DockConfig(enabled=False)
    ))
    s = make_state()
    action = jnp.array([12000.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    s_next, _ = dyn.step(s, action)
    # a = F/m = 12000/12000 = 1 m/s^2; after dt=0.05 => v ~ 0.05 m/s
    assert np.isclose(float(s_next[3]), 0.05, atol=1e-3)
    assert np.isclose(float(s_next[4]), 0.0, atol=1e-6)


def test_quaternion_stays_normalized_over_long_rollout():
    dyn = ISSDynamics(ISSConfig(
        physics=PhysicsConfig(collision_boxes_path=None), dock=DockConfig(enabled=False)
    ))
    s = make_state(omega=(0.5, -0.3, 0.2))
    for _ in range(500):
        s, _ = dyn.step(s, ZERO_ACTION)
    assert np.isclose(float(jnp.linalg.norm(s[6:10])), 1.0, atol=1e-4)


def test_angular_damping_decays_spin():
    # At angular_damping=0.02 with inertia_diag[0]=80000, the per-step change
    # is domega = -angular_damping/inertia * omega * dt = -0.02/80000 * 1.0 *
    # 0.05 = -1.25e-8, a relative change about 10x below float32's ULP at 1.0
    # (~1.19e-7): it rounds away to exactly 1.0 every step and never
    # accumulates, since each step re-quantizes to float32 (see
    # test_angular_damping_is_inert_at_default_config below). Override
    # angular_damping to 200.0 here so the relative change (~1.25e-4) is well
    # above float32 resolution, letting this test actually check that the
    # damping term is wired into the EOM.
    dyn = ISSDynamics(ISSConfig(
        physics=PhysicsConfig(collision_boxes_path=None, angular_damping=200.0),
        dock=DockConfig(enabled=False),
    ))
    s = make_state(omega=(1.0, 0.0, 0.0))
    s_next, _ = dyn.step(s, ZERO_ACTION)
    assert float(s_next[10]) < 1.0


def test_angular_damping_is_inert_at_default_config():
    # angular_damping is set explicitly to 0.02 here rather than relying on
    # the config default (0.0, the physical value for vacuum), to exercise
    # the float32 precision limit documented in test_angular_damping_decays_
    # spin above: at angular_damping=0.02 with inertia_diag[0]=80000, the
    # per-step decay is far below float32 resolution, so omega is
    # bit-for-bit unchanged both after one step and after a longer rollout,
    # since the decrement underflows on every single step rather than
    # accumulating. This assertion would only fail if this simulator's
    # dtype precision changed enough to make angular damping numerically
    # observable here.
    dyn = ISSDynamics(ISSConfig(
        physics=PhysicsConfig(collision_boxes_path=None, angular_damping=0.02),
        dock=DockConfig(enabled=False),
    ))
    s = make_state(omega=(1.0, 0.0, 0.0))
    s_next, _ = dyn.step(s, ZERO_ACTION)
    assert float(s_next[10]) == 1.0

    for _ in range(100):
        s, _ = dyn.step(s, ZERO_ACTION)
    assert float(s[10]) == 1.0


def test_collision_fires_inside_a_box():
    dyn = ISSDynamics(ISSConfig(
        physics=PhysicsConfig(
            collision_boxes_path=[{"center": [10.0, 0.0, 0.0], "size": [4.0, 4.0, 4.0]}]
        ),
        dock=DockConfig(enabled=False),
    ))
    _, events = dyn.step(make_state(pos=(10.0, 0.0, 0.0)), ZERO_ACTION)
    assert bool(events.collision) is True


def test_collision_does_not_fire_far_away():
    dyn = ISSDynamics(ISSConfig(
        physics=PhysicsConfig(
            collision_boxes_path=[{"center": [10.0, 0.0, 0.0], "size": [4.0, 4.0, 4.0]}]
        ),
        dock=DockConfig(enabled=False),
    ))
    _, events = dyn.step(make_state(pos=(500.0, 0.0, 0.0)), ZERO_ACTION)
    assert bool(events.collision) is False


def test_collision_accounts_for_chaser_radius():
    # Box half-extent 2.0 at origin, chaser radius 2.25 => contact out to 4.25 m.
    physics = PhysicsConfig(
        collision_boxes_path=[{"center": [0.0, 0.0, 0.0], "size": [4.0, 4.0, 4.0]}],
        dragon_collision_radius_m=2.25,
    )
    dyn = ISSDynamics(ISSConfig(physics=physics, dock=DockConfig(enabled=False)))
    _, near = dyn.step(make_state(pos=(4.0, 0.0, 0.0)), ZERO_ACTION)
    _, far = dyn.step(make_state(pos=(6.0, 0.0, 0.0)), ZERO_ACTION)
    assert bool(near.collision) is True
    assert bool(far.collision) is False


def test_collision_detects_tunnelling_through_a_thin_box():
    # Box half-extent 1.0 (2 m thick) at the origin, chaser radius 2.25 =>
    # capture window [-3.25, 3.25] along x, 6.5 m wide. A single dt=0.05 step
    # at 160 m/s covers ~8 m -- more than the capture window -- starting and
    # ending clear of the box on opposite sides, so the endpoint alone would
    # never see it.
    physics = PhysicsConfig(
        collision_boxes_path=[{"center": [0.0, 0.0, 0.0], "size": [2.0, 2.0, 2.0]}],
        dragon_collision_radius_m=2.25,
    )
    dyn = ISSDynamics(ISSConfig(physics=physics, dock=DockConfig(enabled=False)))
    s = make_state(pos=(-4.0, 0.0, 0.0), vel=(160.0, 0.0, 0.0))
    s_next, events = dyn.step(s, ZERO_ACTION)
    # Confirm the setup actually tunnels: both endpoints land clear of the
    # expanded box, yet the straight-line path crosses it.
    assert float(s[0]) < -3.25
    assert float(s_next[0]) > 3.25
    assert bool(events.collision) is True


def test_collision_sweep_has_no_false_positive_when_passing_nearby():
    # Same fast pass along x, but offset 10 m in y -- well clear of the
    # expanded box's 3.25 m half-width -- so the segment never comes near it.
    physics = PhysicsConfig(
        collision_boxes_path=[{"center": [0.0, 0.0, 0.0], "size": [2.0, 2.0, 2.0]}],
        dragon_collision_radius_m=2.25,
    )
    dyn = ISSDynamics(ISSConfig(physics=physics, dock=DockConfig(enabled=False)))
    s = make_state(pos=(-4.0, 10.0, 0.0), vel=(160.0, 0.0, 0.0))
    _, events = dyn.step(s, ZERO_ACTION)
    assert bool(events.collision) is False


def test_collision_zero_length_segment_inside_box_still_collides():
    # A stationary chaser (pos_prev == pos_next) must reduce to a plain
    # point-in-box test, not divide by a zero segment length and produce NaN.
    physics = PhysicsConfig(
        collision_boxes_path=[{"center": [10.0, 0.0, 0.0], "size": [4.0, 4.0, 4.0]}],
    )
    dyn = ISSDynamics(ISSConfig(physics=physics, dock=DockConfig(enabled=False)))
    _, events = dyn.step(make_state(pos=(10.0, 0.0, 0.0)), ZERO_ACTION)
    assert bool(events.collision) is True


def test_collision_zero_length_segment_far_away_does_not_collide():
    physics = PhysicsConfig(
        collision_boxes_path=[{"center": [10.0, 0.0, 0.0], "size": [4.0, 4.0, 4.0]}],
    )
    dyn = ISSDynamics(ISSConfig(physics=physics, dock=DockConfig(enabled=False)))
    _, events = dyn.step(make_state(pos=(500.0, 0.0, 0.0)), ZERO_ACTION)
    assert bool(events.collision) is False


def test_collision_sweep_is_jit_and_vmap_compatible():
    physics = PhysicsConfig(
        collision_boxes_path=[{"center": [0.0, 0.0, 0.0], "size": [2.0, 2.0, 2.0]}],
        dragon_collision_radius_m=2.25,
    )
    dyn = ISSDynamics(ISSConfig(physics=physics, dock=DockConfig(enabled=False)))
    s = make_state(pos=(-4.0, 0.0, 0.0), vel=(160.0, 0.0, 0.0))

    s_next, events = jax.jit(dyn.step)(s, ZERO_ACTION)
    assert bool(events.collision) is True

    states = jnp.stack([s, s.at[1].set(10.0)])
    actions = jnp.stack([ZERO_ACTION, ZERO_ACTION])
    _, vmapped = jax.vmap(jax.jit(dyn.step))(states, actions)
    assert vmapped.collision.shape == (2,)
    assert bool(vmapped.collision[0]) is True
    assert bool(vmapped.collision[1]) is False


def test_dock_requires_both_distance_and_speed():
    cfg = ISSConfig(
        physics=PhysicsConfig(collision_boxes_path=None),
        # Attitude gate off: this test isolates distance/speed, and the
        # states below use the identity quaternion, not dock's default
        # docking-port orientation.
        dock=DockConfig(enabled=True, position=(0.0, 0.0, 0.0),
                        max_distance_m=0.1, max_velocity_m_s=0.5,
                        max_attitude_error_deg=None),
    )
    dyn = ISSDynamics(cfg)

    _, slow_and_close = dyn.step(make_state(pos=(0.0, 0.0, 0.0)), ZERO_ACTION)
    assert bool(slow_and_close.docked) is True

    _, fast_and_close = dyn.step(make_state(pos=(0.0, 0.0, 0.0), vel=(5.0, 0.0, 0.0)), ZERO_ACTION)
    assert bool(fast_and_close.docked) is False

    _, slow_and_far = dyn.step(make_state(pos=(50.0, 0.0, 0.0)), ZERO_ACTION)
    assert bool(slow_and_far.docked) is False


def test_dock_disabled_never_fires():
    dyn = ISSDynamics(ISSConfig(
        physics=PhysicsConfig(collision_boxes_path=None),
        dock=DockConfig(enabled=False, position=(0.0, 0.0, 0.0)),
    ))
    _, events = dyn.step(make_state(), ZERO_ACTION)
    assert bool(events.docked) is False


def test_dock_requires_attitude_when_the_gate_is_set():
    cfg = ISSConfig(dock=DockConfig(
        enabled=True, position=(0.0, 0.0, 0.0), quaternion=(1.0, 0.0, 0.0, 0.0),
        max_distance_m=1.0, max_velocity_m_s=1.0, max_attitude_error_deg=10.0,
    ), physics=PhysicsConfig(collision_boxes_path=None))
    dyn = ISSDynamics(cfg)

    aligned = make_state(pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0))
    assert bool(dyn.step(aligned, ZERO_ACTION)[1].docked) is True

    # 180 deg about x -- in position, but pointing the wrong way entirely.
    flipped = make_state(pos=(0.0, 0.0, 0.0), quat=(0.0, 1.0, 0.0, 0.0))
    assert bool(dyn.step(flipped, ZERO_ACTION)[1].docked) is False


def test_attitude_gate_accepts_the_negated_quaternion():
    """q and -q are the SAME rotation. Without abs() on w, one of them fails."""
    cfg = ISSConfig(dock=DockConfig(
        enabled=True, position=(0.0, 0.0, 0.0), quaternion=(1.0, 0.0, 0.0, 0.0),
        max_distance_m=1.0, max_velocity_m_s=1.0, max_attitude_error_deg=10.0,
    ), physics=PhysicsConfig(collision_boxes_path=None))
    dyn = ISSDynamics(cfg)
    negated = make_state(pos=(0.0, 0.0, 0.0), quat=(-1.0, 0.0, 0.0, 0.0))
    assert bool(dyn.step(negated, ZERO_ACTION)[1].docked) is True


def test_dock_requires_body_rates_when_the_gate_is_set():
    cfg = ISSConfig(dock=DockConfig(
        enabled=True, position=(0.0, 0.0, 0.0), max_distance_m=1.0,
        max_velocity_m_s=1.0, max_body_rate_rad_s=0.01,
        # Attitude gate off: this test isolates body rate, and the states
        # below use the identity quaternion, not dock's default docking-port
        # orientation.
        max_attitude_error_deg=None,
    ), physics=PhysicsConfig(collision_boxes_path=None))
    dyn = ISSDynamics(cfg)
    still = make_state(pos=(0.0, 0.0, 0.0), omega=(0.0, 0.0, 0.0))
    assert bool(dyn.step(still, ZERO_ACTION)[1].docked) is True
    tumbling = make_state(pos=(0.0, 0.0, 0.0), omega=(5.0, 0.0, 0.0))
    assert bool(dyn.step(tumbling, ZERO_ACTION)[1].docked) is False


def test_gates_can_be_explicitly_disabled_to_admit_any_attitude_and_rate():
    # max_attitude_error_deg / max_body_rate_rad_s default to 5 deg / 0.5
    # deg/s, not off -- but explicit None still admits any attitude/rate,
    # preserving the position-and-velocity-only success criteria for callers
    # who opt out of the gates.
    cfg = ISSConfig(dock=DockConfig(
        enabled=True, position=(0.0, 0.0, 0.0), max_distance_m=1.0, max_velocity_m_s=1.0,
        max_attitude_error_deg=None, max_body_rate_rad_s=None,
    ), physics=PhysicsConfig(collision_boxes_path=None))
    dyn = ISSDynamics(cfg)
    wild = make_state(pos=(0.0, 0.0, 0.0), quat=(0.0, 1.0, 0.0, 0.0), omega=(9.0, 9.0, 9.0))
    assert bool(dyn.step(wild, ZERO_ACTION)[1].docked) is True


def test_attitude_gate_is_jit_and_vmap_compatible():
    cfg = ISSConfig(dock=DockConfig(
        enabled=True, position=(0.0, 0.0, 0.0), max_distance_m=1.0,
        max_velocity_m_s=1.0, max_attitude_error_deg=10.0, max_body_rate_rad_s=0.1,
    ), physics=PhysicsConfig(collision_boxes_path=None))
    dyn = ISSDynamics(cfg)
    states = jnp.stack([make_state(pos=(0.0, 0.0, 0.0))] * 3)
    _, events = jax.vmap(jax.jit(dyn.step))(states, jnp.zeros((3, 6), jnp.float32))
    assert events.docked.shape == (3,)


def test_reset_places_chaser_on_the_start_sphere():
    dyn = ISSDynamics(ISSConfig(physics=PhysicsConfig(start_radius_m=100.0)))
    s = dyn.reset(jax.random.PRNGKey(0))
    assert s.shape == (13,)
    assert np.isclose(float(jnp.linalg.norm(s[0:3])), 100.0, atol=1e-3)
    np.testing.assert_allclose(np.asarray(s[3:6]), np.zeros(3), atol=1e-6)
    np.testing.assert_allclose(np.asarray(s[10:13]), np.zeros(3), atol=1e-6)
    assert np.isclose(float(jnp.linalg.norm(s[6:10])), 1.0, atol=1e-5)


def test_reset_points_body_z_at_the_iss():
    from owm_envs.core.quaternion import rotate_body_to_world

    dyn = ISSDynamics(ISSConfig(physics=PhysicsConfig(start_radius_m=100.0)))
    s = dyn.reset(jax.random.PRNGKey(3))
    nose_world = rotate_body_to_world(s[6:10], jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32))
    to_iss = -s[0:3] / jnp.linalg.norm(s[0:3])
    np.testing.assert_allclose(np.asarray(nose_world), np.asarray(to_iss), atol=1e-4)


def test_reset_is_deterministic_per_key():
    dyn = ISSDynamics(ISSConfig())
    a = dyn.reset(jax.random.PRNGKey(7))
    b = dyn.reset(jax.random.PRNGKey(7))
    c = dyn.reset(jax.random.PRNGKey(8))
    np.testing.assert_allclose(np.asarray(a), np.asarray(b))
    assert not np.allclose(np.asarray(a), np.asarray(c))


def test_step_is_jit_compatible():
    dyn = ISSDynamics(ISSConfig(
        physics=PhysicsConfig(collision_boxes_path=default_collision_boxes_path())
    ))
    jitted = jax.jit(dyn.step)
    s_next, events = jitted(make_state(pos=(100.0, 0.0, 0.0)), ZERO_ACTION)
    assert s_next.shape == (13,)
    assert bool(events.collision) is False


def test_step_is_vmap_compatible():
    dyn = ISSDynamics(ISSConfig(
        physics=PhysicsConfig(collision_boxes_path=None), dock=DockConfig(enabled=False)
    ))
    states = jnp.stack([make_state(pos=(float(i), 0.0, 0.0)) for i in range(4)])
    actions = jnp.zeros((4, 6), dtype=jnp.float32)
    s_next, events = jax.vmap(dyn.step)(states, actions)
    assert s_next.shape == (4, 13)
    assert events.collision.shape == (4,)


# --- Reference-orbit dynamics (CW acceleration + gravity-gradient torque) ---
#
# astrojax API conventions confirmed by reading source
# (.venv/.../astrojax/attitude_dynamics/gravity_gradient.py,
# .venv/.../astrojax/relative_motion/hcw_dynamics.py):
#
# - `hcw_derivative(state_rtn, n)`: state is `[x, y, z, xdot, ydot, zdot]` in
#   the chief's RTN frame; returns `[xdot, ydot, zdot, xddot, yddot, zddot]`,
#   i.e. the acceleration half is `[3:6]`.
# - `torque_gravity_gradient(q, r_eci, I, mu)`: despite its docstring
#   claiming `q` is the body->inertial rotation, it is actually the
#   INERTIAL->BODY (world->body) quaternion -- i.e. `quat_conjugate(q_bw)`,
#   not `q_bw` -- confirmed empirically below: `q_bw` passed directly gives
#   a torque rotated away from the closed-form value by the same angle as
#   the attitude itself, while its conjugate matches to float32 precision.
#   (`astrojax`'s `quaternion_to_rotation_matrix(q)` produces the identical
#   matrix as this codebase's `quat_to_rotmat(q).T`, i.e. world->body for a
#   `q_bw`-convention input -- the reverse of what the docstring states.)
#   `r_eci` is the spacecraft's position vector (primary center ->
#   spacecraft) in the inertial frame, shape (3,). `I` is the FULL 3x3
#   inertia tensor (not a 3-vector diagonal) -- `dynamics.py` passes
#   `jnp.diag(self._inertia_diag)`. Returns torque in the BODY frame.
#   Internally it computes `r_hat_body x (I r_hat_body)` -- since that
#   expression is invariant under negating the input vector, it doesn't
#   matter whether `r_hat` is nadir or zenith; the cross-check below uses
#   nadir (`R_bw^T @ [0,0,-1]`) and still agrees.


def _orbit_cfg(**overrides) -> ISSConfig:
    return ISSConfig(
        physics=PhysicsConfig(collision_boxes_path=None),
        dock=DockConfig(enabled=False),
        orbit=OrbitConfig(enabled=True, **overrides),
    )


def test_orbit_disabled_step_matches_default_config_step():
    # Explicit `orbit=OrbitConfig(enabled=False)` vs relying on the default
    # (also disabled) must be bit-identical: the disabled branch in `_eom`
    # is a static Python `if`, resolved at trace time, so it adds no ops.
    cfg_default = ISSConfig(
        physics=PhysicsConfig(collision_boxes_path=None), dock=DockConfig(enabled=False)
    )
    cfg_explicit = ISSConfig(
        physics=PhysicsConfig(collision_boxes_path=None),
        dock=DockConfig(enabled=False),
        orbit=OrbitConfig(enabled=False),
    )
    dyn_default = ISSDynamics(cfg_default)
    dyn_explicit = ISSDynamics(cfg_explicit)

    s = make_state(pos=(50.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0), omega=(0.1, -0.2, 0.3))
    action = jnp.array([100.0, 20.0, -5.0, 0.01, -0.02, 0.03], dtype=jnp.float32)

    s_next_default, _ = dyn_default.step(s, action)
    s_next_explicit, _ = dyn_explicit.step(s, action)
    np.testing.assert_array_equal(np.asarray(s_next_default), np.asarray(s_next_explicit))


def test_cw_enabled_origin_at_rest_stays_at_rest():
    dyn = ISSDynamics(_orbit_cfg())
    s = make_state()
    s_next, _ = dyn.step(s, ZERO_ACTION)
    np.testing.assert_allclose(np.asarray(s_next), np.asarray(s), atol=1e-6)


def test_cw_radial_offset_produces_expected_radial_acceleration():
    # R = +z_world (RTN_FROM_WORLD row 0). A purely radial, at-rest offset
    # has zero along-track/normal RTN components, so the HCW equations give
    # x_ddot = 3n^2 x, y_ddot = -2n*xdot = 0, z_ddot = -n^2 z = 0: the whole
    # acceleration should land back on +z_world.
    cfg = _orbit_cfg()
    dyn = ISSDynamics(cfg)
    n = ReferenceOrbit(cfg.orbit).mean_motion

    s = make_state(pos=(0.0, 0.0, 100.0))
    deriv = dyn._eom(s, ZERO_ACTION)
    accel_w = np.asarray(deriv[3:6])

    expected_a_r = 3.0 * n**2 * 100.0
    np.testing.assert_allclose(accel_w, np.array([0.0, 0.0, expected_a_r]), rtol=1e-3, atol=1e-7)


def test_cw_free_drift_matches_analytic_hcw_solution_over_60s():
    cfg = _orbit_cfg()
    dyn = ISSDynamics(cfg)
    n = ReferenceOrbit(cfg.orbit).mean_motion

    s = make_state(pos=(0.0, 0.0, 100.0))  # x_R = 100 m, at rest
    total_t = 60.0
    n_steps = int(round(total_t / cfg.dt))
    for _ in range(n_steps):
        s, _ = dyn.step(s, ZERO_ACTION)

    pos_rtn = RTN_FROM_WORLD @ np.asarray(s[0:3])
    vel_rtn = RTN_FROM_WORLD @ np.asarray(s[3:6])
    state_rtn = np.concatenate([pos_rtn, vel_rtn])

    state0_rtn = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    expected_rtn = np.asarray(hcw_stm(n_steps * cfg.dt, n)) @ state0_rtn

    np.testing.assert_allclose(state_rtn, expected_rtn, rtol=1e-3, atol=1e-3)


def test_gravity_gradient_torque_matches_closed_form_for_random_attitudes():
    cfg = _orbit_cfg()
    dyn = ISSDynamics(cfg)
    n = ReferenceOrbit(cfg.orbit).mean_motion
    I = np.diag(np.asarray(cfg.physics.inertia_diag, dtype=np.float64))  # noqa: E741

    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q = q / np.linalg.norm(q)
        q_bw = jnp.asarray(q, dtype=jnp.float32)

        tau_astrojax = np.asarray(
            torque_gravity_gradient(quat_conjugate(q_bw), dyn._r_world_gg, dyn._inertia_matrix)
        )

        R_bw = np.asarray(quat_to_rotmat(q_bw), dtype=np.float64)
        u_nadir = R_bw.T @ np.array([0.0, 0.0, -1.0])
        tau_closed_form = 3.0 * n**2 * np.cross(u_nadir, I @ u_nadir)

        np.testing.assert_allclose(tau_astrojax, tau_closed_form, rtol=1e-4, atol=1e-8)


