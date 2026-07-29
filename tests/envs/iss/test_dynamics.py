import jax
import jax.numpy as jnp
import numpy as np
import pytest

from owm_envs.envs.iss.config import (
    DockConfig,
    ISSConfig,
    PhysicsConfig,
    default_collision_boxes_path,
)
from owm_envs.envs.iss.dynamics import STATE_LABELS, ISSDynamics

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
    # dt=0.05 with a small linear damping term, so slightly under 0.05 m.
    assert 0.045 < float(s_next[0] - s[0]) <= 0.05


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
    # At the default angular_damping=0.02 and inertia_diag[0]=80000, the true
    # per-step change is domega = -angular_damping/inertia * omega * dt
    # = -0.02/80000 * 1.0 * 0.05 = -1.25e-8 -- a relative change of 1.25e-8,
    # about 10x below float32's ULP at 1.0 (~1.19e-7). It rounds away to
    # exactly 1.0 every step and never accumulates (each step re-quantizes to
    # float32), so the default config cannot exercise this path at all -- see
    # test_angular_damping_is_inert_at_default_config below. Override
    # angular_damping here so the effect is well above float32 resolution
    # (relative change ~1.25e-4) and this test actually checks that the
    # damping term is wired into the EOM.
    dyn = ISSDynamics(ISSConfig(
        physics=PhysicsConfig(collision_boxes_path=None, angular_damping=200.0),
        dock=DockConfig(enabled=False),
    ))
    s = make_state(omega=(1.0, 0.0, 0.0))
    s_next, _ = dyn.step(s, ZERO_ACTION)
    assert float(s_next[10]) < 1.0


def test_angular_damping_is_inert_at_default_config():
    # Pins the float32 precision gotcha documented above: with the shipped
    # defaults (angular_damping=0.02, inertia_diag[0]=80000), the per-step
    # decay is far below float32 resolution, so omega is bit-for-bit
    # unchanged -- both after one step and after a longer rollout, since the
    # decrement underflows on every single step rather than accumulating.
    # If this test starts failing, the defaults or the float32 dtype mandate
    # changed and angular damping is now actually observable in this sim.
    dyn = ISSDynamics(ISSConfig(
        physics=PhysicsConfig(collision_boxes_path=None), dock=DockConfig(enabled=False)
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


def test_dock_requires_both_distance_and_speed():
    cfg = ISSConfig(
        physics=PhysicsConfig(collision_boxes_path=None),
        dock=DockConfig(enabled=True, position=(0.0, 0.0, 0.0),
                        max_distance_m=0.1, max_velocity_m_s=0.5),
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


def test_quat_from_body_z_to_handles_exact_antiparallel_target():
    # Degenerate case: target_dir == -body_z, where the `w = 1 + dot` form
    # goes to zero. The fallback (180-degree rotation about x) must trigger
    # and still produce a unit quaternion that correctly maps +z -> -z.
    from owm_envs.core.quaternion import rotate_body_to_world
    from owm_envs.envs.iss.dynamics import BODY_Z, _quat_from_body_z_to

    target = jnp.array([0.0, 0.0, -1.0], dtype=jnp.float32)
    q = _quat_from_body_z_to(target)

    assert np.isclose(float(jnp.linalg.norm(q)), 1.0, atol=1e-6)
    mapped = rotate_body_to_world(q, BODY_Z)
    np.testing.assert_allclose(np.asarray(mapped), np.asarray(target), atol=1e-5)
