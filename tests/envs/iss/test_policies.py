import jax
import jax.numpy as jnp
import numpy as np
import pytest

from owm_envs.core.quaternion import quat_from_body_z_to
from owm_envs.envs.iss.config import DockConfig, ISSConfig, PhysicsConfig
from owm_envs.envs.iss.dynamics import ISSDynamics
from owm_envs.envs.iss.policies import EXTRAS_DIM, OrbitParams, PolicyConfig, make_policy

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


def test_sampled_orbit_is_always_within_the_thrust_budget():
    # Free-body dynamics: holding radius R at rate w needs a sustained
    # centripetal force m*w^2*R, and actions are clipped per axis at
    # limit_force_n. Sampling radius and rate independently (as it used to)
    # commanded circles needing up to 124 kN against an 18 kN limit.
    _, extras_fn = make_policy(CFG, PCFG, "orbit")
    budget = PCFG.orbit.thrust_utilization * CFG.control.limit_force_n
    for seed in range(200):
        extras = extras_fn(jax.random.PRNGKey(seed))
        radius, omega = float(extras[3]), float(extras[4])
        assert CFG.physics.mass * omega**2 * radius <= budget + 1e-3


def test_orbit_rate_falls_as_one_over_sqrt_radius():
    # omega = fraction * sqrt(alpha * F_limit / (m * R)), so at a fixed
    # fraction a 4x larger radius must orbit exactly half as fast.
    cfg = PolicyConfig(orbit=OrbitParams(speed_fraction_range=(1.0, 1.0)))
    near, far = 100.0, 400.0
    rates = []
    for radius in (near, far):
        _, extras_fn = make_policy(
            CFG, PolicyConfig(orbit=cfg.orbit.model_copy(
                update={"radius_range_m": (radius, radius)})), "orbit")
        rates.append(float(extras_fn(jax.random.PRNGKey(3))[4]))
    assert np.isclose(rates[0] / rates[1], np.sqrt(far / near), rtol=1e-4)


def test_speed_fraction_range_bounds_the_sampled_rate():
    lo, hi = 0.25, 0.75
    cfg = PolicyConfig(orbit=OrbitParams(speed_fraction_range=(lo, hi)))
    _, extras_fn = make_policy(CFG, cfg, "orbit")
    budget = cfg.orbit.thrust_utilization * CFG.control.limit_force_n
    for seed in range(100):
        extras = extras_fn(jax.random.PRNGKey(seed))
        radius, omega = float(extras[3]), float(extras[4])
        omega_max = np.sqrt(budget / (CFG.physics.mass * radius))
        assert lo - 1e-5 <= omega / omega_max <= hi + 1e-5


def test_orbit_feedforward_holds_the_commanded_circle():
    # The PD reference is the radial projection of the CURRENT position, so it
    # commands zero force exactly when the chaser is perfectly on the circle --
    # the moment it most needs centripetal force. Without a feedforward the law
    # must run a standing radial error to generate it, settling 5-13% wide.
    policy_fn, _ = make_policy(CFG, PCFG, "orbit")
    dynamics = ISSDynamics(CFG)
    radius = 200.0
    omega = float(np.sqrt(
        PCFG.orbit.thrust_utilization * CFG.control.limit_force_n
        / (CFG.physics.mass * radius)
    ))
    axis = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    r_hat = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    extras = jnp.array([0.0, 0.0, 1.0, radius, omega], dtype=jnp.float32)

    # Start exactly on the commanded circle, at the commanded tangential speed,
    # already pointing at the station: only the centripetal force is missing.
    state = jnp.concatenate([
        radius * r_hat,
        omega * radius * jnp.cross(axis, r_hat),
        quat_from_body_z_to(-r_hat),
        jnp.zeros((3,), dtype=jnp.float32),
    ])
    limit = CFG.control.limit_force_n
    radii = []
    for _ in range(2000):
        action = policy_fn(state, jax.random.PRNGKey(0), extras)
        assert np.all(np.abs(np.asarray(action[0:3])) < limit), "thrust saturated"
        state, _ = dynamics.step(state, action)
        radii.append(float(jnp.linalg.norm(state[0:3] - jnp.dot(state[0:3], axis) * axis)))

    settled = np.asarray(radii[len(radii) // 2:])
    assert np.abs(settled.mean() / radius - 1.0) < 0.01


def test_thrust_utilization_must_be_a_fraction():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            OrbitParams(thrust_utilization=bad)


def test_speed_fraction_range_must_be_ordered_and_positive():
    for bad in ((0.8, 0.4), (0.0, 1.0), (-0.1, 0.5), (0.5, 1.5)):
        with pytest.raises(ValueError):
            OrbitParams(speed_fraction_range=bad)


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
    # Centripetal feedforward along the reference radial, plus the PD on the
    # error against that same reference.
    radius_cmd, omega_cmd = extras[3], extras[4]
    centripetal = (
        -CFG.physics.mass * omega_cmd**2 * radius_cmd * (p_des / jnp.linalg.norm(p_des))
    )
    force_world_expected = (
        centripetal
        - params.kp_position * (state[0:3] - p_des)
        - params.kd_velocity * (state[3:6] - v_des)
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
