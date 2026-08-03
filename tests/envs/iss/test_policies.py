import jax
import jax.numpy as jnp
import numpy as np
import pytest

from owm_envs.envs.iss.config import DockConfig, ISSConfig, PhysicsConfig
from owm_envs.envs.iss.dynamics import ISSDynamics
from owm_envs.envs.iss.policies import EXTRAS_DIM, PolicyConfig, make_policy

CFG = ISSConfig(
    physics=PhysicsConfig(collision_boxes_path=None),
    dock=DockConfig(position=(0.0, 0.0, 0.0)),
)
PCFG = PolicyConfig()


def state_at(pos, vel=(0.0, 0.0, 0.0), omega=(0.0, 0.0, 0.0)) -> jnp.ndarray:
    return jnp.asarray([*pos, *vel, 1.0, 0.0, 0.0, 0.0, *omega], dtype=jnp.float32)


@pytest.mark.parametrize("kind", ["random", "orbit", "dock", "union"])
def test_every_policy_returns_a_6d_action(kind):
    policy_fn, extras_fn = make_policy(CFG, PCFG, kind)
    extras = (extras_fn(jax.random.PRNGKey(0)) if extras_fn is not None
              else jnp.zeros((0,), dtype=jnp.float32))
    action = policy_fn(state_at((50.0, 0.0, 0.0)), jax.random.PRNGKey(1), extras)
    assert action.shape == (6,)
    assert np.all(np.isfinite(np.asarray(action)))


@pytest.mark.parametrize("kind,width", list(EXTRAS_DIM.items()))
def test_extras_width_matches_the_declared_dimension(kind, width):
    _, extras_fn = make_policy(CFG, PCFG, kind)
    if width == 0:
        assert extras_fn is None
    else:
        assert extras_fn(jax.random.PRNGKey(0)).shape == (width,)


def test_random_policy_respects_control_limits():
    policy_fn, _ = make_policy(CFG, PCFG, "random")
    empty = jnp.zeros((0,), dtype=jnp.float32)
    for seed in range(20):
        a = np.asarray(policy_fn(state_at((0.0, 0.0, 0.0)), jax.random.PRNGKey(seed), empty))
        assert np.all(np.abs(a[0:3]) <= CFG.control.limit_force_n + 1e-3)
        assert np.all(np.abs(a[3:6]) <= CFG.control.limit_torque_nm + 1e-3)


def test_dock_policy_pushes_toward_the_dock():
    policy_fn, _ = make_policy(CFG, PCFG, "dock")
    empty = jnp.zeros((0,), dtype=jnp.float32)
    # Identity attitude, so body frame == world frame; chaser at +x must be pushed -x.
    action = policy_fn(state_at((10.0, 0.0, 0.0)), jax.random.PRNGKey(0), empty)
    assert float(action[0]) < 0.0


def test_dock_policy_drives_the_chaser_to_the_dock():
    cfg = ISSConfig(
        physics=PhysicsConfig(collision_boxes_path=None),
        dock=DockConfig(position=(0.0, 0.0, 0.0), quaternion=(1.0, 0.0, 0.0, 0.0), enabled=True),
    )
    dyn = ISSDynamics(cfg)
    policy_fn, _ = make_policy(cfg, PCFG, "dock")
    empty = jnp.zeros((0,), dtype=jnp.float32)

    s = state_at((30.0, 0.0, 0.0))
    force_limit = cfg.control.limit_force_n
    torque_limit = cfg.control.limit_torque_nm
    for _ in range(2000):
        a = policy_fn(s, jax.random.PRNGKey(0), empty)
        a = jnp.concatenate([
            jnp.clip(a[0:3], -force_limit, force_limit),
            jnp.clip(a[3:6], -torque_limit, torque_limit),
        ])
        s, events = dyn.step(s, a)
        if bool(events.docked):
            break
    assert float(jnp.linalg.norm(s[0:3])) < 1.0


def test_orbit_extras_axis_is_a_unit_vector():
    _, extras_fn = make_policy(CFG, PCFG, "orbit")
    extras = extras_fn(jax.random.PRNGKey(5))
    assert np.isclose(float(jnp.linalg.norm(extras[0:3])), 1.0, atol=1e-5)
    lo, hi = PCFG.orbit.radius_range_m
    assert lo <= float(extras[3]) <= hi


def test_orbit_policy_action_is_finite_when_radial_direction_is_exact_body_minus_z():
    # Chaser on world +z, identity attitude, orbit axis +x: the inward radial
    # direction is exactly (0, 0, -1), i.e. antiparallel to body +z. The
    # shortest-arc attitude target must use the antiparallel fallback rather
    # than degenerate to a zero quaternion.
    policy_fn, _ = make_policy(CFG, PCFG, "orbit")
    extras = jnp.array([1.0, 0.0, 0.0, 100.0, 0.15], dtype=jnp.float32)
    action = policy_fn(state_at((0.0, 0.0, 100.0)), jax.random.PRNGKey(0), extras)
    assert np.all(np.isfinite(np.asarray(action)))


def test_orbit_controller_and_goal_share_the_reference():
    # The controller and the recorded goal-error block must chase the exact
    # same commanded state -- otherwise the recorded goal is not what the
    # policy is actually doing. Both must be built from the same
    # `orbit_reference` call for a given (measured pos, extras, dt).
    from owm_envs.core.quaternion import quat_normalize, quat_to_rotmat
    from owm_envs.envs.iss.goal import goal_error
    from owm_envs.envs.iss.policies import orbit_reference

    params = PCFG.orbit
    policy_fn, _ = make_policy(CFG, PCFG, "orbit")
    extras = jnp.array([0.0, 0.0, 1.0, 120.0, 0.15], dtype=jnp.float32)

    # A random off-circle state: not on the commanded radius, with nonzero
    # velocity, a non-identity attitude and nonzero body rates.
    key_pos, key_vel, key_quat, key_omega = jax.random.split(jax.random.PRNGKey(7), 4)
    pos = jax.random.normal(key_pos, (3,), dtype=jnp.float32) * 50.0 + jnp.array(
        [80.0, 30.0, 5.0], dtype=jnp.float32
    )
    vel = jax.random.normal(key_vel, (3,), dtype=jnp.float32)
    q_bw = quat_normalize(jax.random.normal(key_quat, (4,), dtype=jnp.float32))
    omega = 0.05 * jax.random.normal(key_omega, (3,), dtype=jnp.float32)
    state = jnp.concatenate([pos, vel, q_bw, omega])

    p_des, v_des, q_des = orbit_reference(state[0:3], extras, CFG.dt)

    action = policy_fn(state, jax.random.PRNGKey(0), extras)
    force_world_expected = -params.kp_position * (state[0:3] - p_des) - params.kd_velocity * (
        state[3:6] - v_des
    )
    force_body_expected = quat_to_rotmat(q_bw).T @ force_world_expected
    np.testing.assert_allclose(
        np.asarray(action[0:3]), np.asarray(force_body_expected), atol=1e-3
    )

    goal_err = goal_error(state, p_des, v_des, q_des, jnp.zeros(3, jnp.float32))
    np.testing.assert_allclose(
        np.asarray(goal_err[0:3]), np.asarray(state[0:3] - p_des), atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(goal_err[3:6]), np.asarray(state[3:6] - v_des), atol=1e-6
    )


def test_union_selects_all_three_subpolicies_across_seeds():
    _, extras_fn = make_policy(CFG, PCFG, "union")
    chosen = {int(extras_fn(jax.random.PRNGKey(s))[0]) for s in range(200)}
    assert chosen == {0, 1, 2}


def test_union_weights_must_sum_positive():
    # Raised by the PolicyConfig field validator at construction time now,
    # rather than inside make_policy/_build_union -- still a ValueError
    # (pydantic.ValidationError subclasses it), so this still holds.
    with pytest.raises(ValueError):
        PolicyConfig(union_weights=(0.0, 0.0, 0.0))


def test_union_weights_rejects_negative_component():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="-1.0"):
        PolicyConfig(union_weights=(-1.0, 2.0, 0.0))


def test_union_weights_all_zeros_rejected_at_config_load():
    from pydantic import ValidationError

    # All-zeros passes the non-negative check (0.0 is not negative) but must
    # still fail the sum-must-be-positive check, at config-load time.
    with pytest.raises(ValidationError, match="sum to > 0"):
        PolicyConfig(union_weights=(0.0, 0.0, 0.0))


def test_union_weights_with_a_zero_component_is_still_accepted():
    # A zero component is a legitimate way to disable one sub-policy.
    cfg = PolicyConfig(union_weights=(0.0, 1.0, 1.0))
    _, extras_fn = make_policy(CFG, cfg, "union")
    chosen = {int(extras_fn(jax.random.PRNGKey(s))[0]) for s in range(200)}
    assert chosen == {1, 2}


def test_unknown_policy_type_raises():
    with pytest.raises(ValueError, match="Unknown ISS policy type"):
        make_policy(CFG, PCFG, "teleport")


def test_policy_type_defaults_to_random():
    assert PolicyConfig().type == "random"


def test_policy_type_rejects_invalid_value():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PolicyConfig(type="teleport")


def test_policy_type_roundtrips_through_yaml(tmp_path):
    original = PolicyConfig(type="dock")
    path = tmp_path / "policy.yaml"
    original.to_yaml(path)
    assert PolicyConfig.from_yaml(path) == original


def test_make_policy_falls_back_to_policy_cfg_type():
    policy_fn, _ = make_policy(CFG, PolicyConfig(type="dock"))
    empty = jnp.zeros((0,), dtype=jnp.float32)
    # Identity attitude, so body frame == world frame; chaser at +x must be pushed -x,
    # which is what the dock policy (not the default random one) does.
    action = policy_fn(state_at((10.0, 0.0, 0.0)), jax.random.PRNGKey(0), empty)
    assert float(action[0]) < 0.0


def test_make_policy_explicit_policy_type_overrides_config():
    # policy_cfg.type says "orbit" (extras width 5); the explicit argument
    # should win, giving the "random" policy (extras width 0, per EXTRAS_DIM).
    _, extras_fn = make_policy(CFG, PolicyConfig(type="orbit"), "random")
    assert extras_fn is None


def test_policy_config_roundtrips_through_yaml(tmp_path):
    # The policy shaped the dataset, so it belongs in the as-run record too.
    from owm_envs.envs.iss.policies import DockParams, OrbitParams

    original = PolicyConfig(
        orbit=OrbitParams(radius_range_m=(10.0, 20.0)),
        dock=DockParams(kp_position=5.0),
        union_weights=(0.5, 0.25, 0.25),
    )
    path = tmp_path / "policy.yaml"
    original.to_yaml(path)
    assert PolicyConfig.from_yaml(path) == original


def test_policies_are_jit_compatible():
    policy_fn, extras_fn = make_policy(CFG, PCFG, "union")
    extras = extras_fn(jax.random.PRNGKey(0))
    action = jax.jit(policy_fn)(state_at((50.0, 0.0, 0.0)), jax.random.PRNGKey(1), extras)
    assert action.shape == (6,)
