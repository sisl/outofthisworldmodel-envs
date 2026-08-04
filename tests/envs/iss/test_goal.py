import jax.numpy as jnp
import numpy as np

from owm_envs.core.quaternion import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_from_body_z_to,
    quat_multiply,
)
from owm_envs.envs.iss.config import ISSConfig, dock_target
from owm_envs.envs.iss.goal import (
    GOAL_ERROR_DIM,
    _orbit_goal_error,
    dock_goal_error,
    goal_error,
    make_augment,
)
from owm_envs.envs.iss.policies import PolicyConfig


def _state(pos, vel=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0), rate=(0.0, 0.0, 0.0)):
    return jnp.asarray(np.concatenate([np.asarray(p, np.float32) for p in (pos, vel, quat, rate)]))


def test_goal_error_is_zero_at_the_goal():
    cfg = ISSConfig()
    at_dock = _state(cfg.dock.position, quat=cfg.dock.quaternion)
    err = dock_goal_error(at_dock, jnp.asarray(dock_target(cfg)))
    np.testing.assert_allclose(np.asarray(err), np.zeros(GOAL_ERROR_DIM), atol=1e-5)


def test_goal_error_components_and_ordering():
    cfg = ISSConfig()
    s = _state(np.asarray(cfg.dock.position) + np.array([1.0, -2.0, 3.0]),
               vel=(0.5, 0.0, 0.0), quat=cfg.dock.quaternion, rate=(0.0, 0.1, 0.0))
    err = np.asarray(dock_goal_error(s, jnp.asarray(dock_target(cfg))))
    np.testing.assert_allclose(err[0:3], [1.0, -2.0, 3.0], atol=1e-5)   # pos_err = measured - target
    np.testing.assert_allclose(err[3:6], [0.5, 0.0, 0.0], atol=1e-5)    # vel_err (target vel = 0)
    np.testing.assert_allclose(err[6:9], np.zeros(3), atol=1e-5)        # att aligned
    np.testing.assert_allclose(err[9:12], [0.0, 0.1, 0.0], atol=1e-5)   # rate_err (target = 0)


def test_attitude_error_matches_controller_convention():
    # 90 deg about body x between measured and target -> axis-angle ~ (pi/2, 0, 0)
    cfg = ISSConfig()
    q_target = jnp.asarray(cfg.dock.quaternion)
    s = _state(cfg.dock.position, quat=(1.0, 0.0, 0.0, 0.0))
    err = np.asarray(dock_goal_error(s, jnp.asarray(dock_target(cfg))))
    assert np.abs(np.linalg.norm(err[6:9]) - np.pi / 2) < 1e-4  # dock quat is a 90 deg rotation


def test_attitude_error_sign_is_rotation_from_measured_to_target():
    # measured = identity, target = +30 deg about body x -> the error is the
    # rotation FROM measured TO target: axis-angle ~ (+0.5236, 0, 0).
    half = np.deg2rad(30.0) / 2.0
    target_quat = jnp.asarray([np.cos(half), np.sin(half), 0.0, 0.0], jnp.float32)
    measured = _state((0.0, 0.0, 0.0))  # identity quaternion
    err = np.asarray(goal_error(
        measured, jnp.zeros(3, jnp.float32), jnp.zeros(3, jnp.float32),
        target_quat, jnp.zeros(3, jnp.float32),
    ))
    np.testing.assert_allclose(err[6:9], [np.deg2rad(30.0), 0.0, 0.0], atol=1e-5)


def test_random_policy_augment_appends_zeros():
    cfg = ISSConfig(observation={"goal_error": True})
    augment = make_augment(cfg, PolicyConfig(type="random"))
    s = _state((50.0, 0.0, 0.0))
    out = np.asarray(augment(s, jnp.zeros((0,), jnp.float32)))
    assert out.shape == (25,)
    np.testing.assert_array_equal(out[:13], np.asarray(s))
    np.testing.assert_array_equal(out[13:], np.zeros(12))


def test_union_augment_switches_on_the_episode_policy():
    cfg = ISSConfig(observation={"goal_error": True})
    augment = make_augment(cfg, PolicyConfig(type="union"))
    s = _state((50.0, 0.0, 0.0))
    extras_random = jnp.asarray([0.0, 1.0, 0.0, 0.0, 100.0, 0.1], jnp.float32)
    extras_dock = jnp.asarray([2.0, 1.0, 0.0, 0.0, 100.0, 0.1], jnp.float32)
    out_random = np.asarray(augment(s, extras_random))
    out_dock = np.asarray(augment(s, extras_dock))
    np.testing.assert_array_equal(out_random[13:], np.zeros(12))
    np.testing.assert_allclose(out_dock[13:], np.asarray(dock_goal_error(s, jnp.asarray(dock_target(cfg)))), atol=1e-6)


def test_orbit_goal_targets_the_next_reference_state():
    cfg = ISSConfig(observation={"goal_error": True})
    augment = make_augment(cfg, PolicyConfig(type="orbit"))
    radius, omega = 100.0, 0.1
    axis = jnp.asarray([0.0, 0.0, 1.0], jnp.float32)
    # On the circle at +x, moving tangentially (+y at omega*r), nose to center (-x).
    q = quat_from_body_z_to(jnp.asarray([-1.0, 0.0, 0.0], jnp.float32))
    s = _state((radius, 0.0, 0.0), vel=(0.0, omega * radius, 0.0),
               quat=np.asarray(q), rate=(0.0, 0.0, 0.0))
    extras = jnp.concatenate([axis, jnp.asarray([radius, omega], jnp.float32)])
    out = np.asarray(augment(s, extras))

    dt = cfg.dt
    phi = omega * dt
    # Target is the reference one dt ahead: p(phi) = R*(cos phi, sin phi, 0).
    pos_err_expected = radius * np.array([1.0 - np.cos(phi), -np.sin(phi), 0.0])
    v_now = np.array([0.0, omega * radius, 0.0])
    v_next = omega * radius * np.array([-np.sin(phi), np.cos(phi), 0.0])
    vel_err_expected = v_now - v_next

    np.testing.assert_allclose(out[13:16], pos_err_expected, atol=1e-4)
    np.testing.assert_allclose(out[16:19], vel_err_expected, atol=1e-4)

    # Attitude target: nose pointed at the center from the NEXT reference
    # position, built independently from the same (separately tested)
    # quaternion primitives the implementation uses.
    r_hat_next = np.array([np.cos(phi), np.sin(phi), 0.0])
    q_target_expected = quat_from_body_z_to(jnp.asarray(-r_hat_next, jnp.float32))
    q_err_expected = quat_multiply(quat_conjugate(jnp.asarray(q)), q_target_expected)
    att_err_expected = np.asarray(axis_angle_from_quat(q_err_expected))
    np.testing.assert_allclose(out[19:22], att_err_expected, atol=1e-4)

    np.testing.assert_allclose(out[22:25], np.zeros(3), atol=1e-5)


def test_orbit_goal_error_dt_zero_matches_current_projection_target():
    cfg = ISSConfig(observation={"goal_error": True})
    radius, omega = 100.0, 0.1
    axis = jnp.asarray([0.0, 0.0, 1.0], jnp.float32)
    q = quat_from_body_z_to(jnp.asarray([-1.0, 0.0, 0.0], jnp.float32))
    s = _state((radius, 0.0, 0.0), vel=(0.0, omega * radius, 0.0),
               quat=np.asarray(q), rate=(0.0, 0.0, 0.0))
    extras = jnp.concatenate([axis, jnp.asarray([radius, omega], jnp.float32)])

    err = np.asarray(_orbit_goal_error(s, extras, dt=0.0))
    np.testing.assert_allclose(err, np.zeros(12), atol=1e-4)


def test_make_augment_returns_none_when_disabled():
    assert make_augment(ISSConfig(), PolicyConfig(type="dock")) is None
