import jax.numpy as jnp
import numpy as np
import pytest

from owm_envs.envs.common.config import (
    DockConfig,
    RewardShapingConfig,
    RewardWeights,
)
from owm_envs.envs.common.events import Events
from owm_envs.envs.common.reward import docking_reward
from owm_envs.envs.iss.config import ISSConfig

NO_EVENTS = Events(collision=jnp.array(False), docked=jnp.array(False), escaped=jnp.array(False))
HIT = Events(collision=jnp.array(True), docked=jnp.array(False), escaped=jnp.array(False))
DOCKED = Events(collision=jnp.array(False), docked=jnp.array(True), escaped=jnp.array(False))
ESCAPED = Events(collision=jnp.array(False), docked=jnp.array(False), escaped=jnp.array(True))

ZERO_ACTION = jnp.zeros((6,), dtype=jnp.float32)
IDENTITY = (1.0, 0.0, 0.0, 0.0)

# Identity dock attitude throughout, so a state built by `state_at` sits at
# zero attitude error unless a test deliberately rotates it. The shipped
# DockConfig quaternion is a 90 deg rotation, which would put a standing
# attitude penalty under every other term's test.
AT_ORIGIN = DockConfig(position=(0.0, 0.0, 0.0), quaternion=IDENTITY)


def state_at(pos, vel=(0.0, 0.0, 0.0), quat=IDENTITY, omega=(0.0, 0.0, 0.0)) -> jnp.ndarray:
    return jnp.asarray([*pos, *vel, *quat, *omega], dtype=jnp.float32)


def only(**weights) -> RewardWeights:
    """Weights with everything zeroed but the named terms."""
    zeroed = dict(position=0.0, velocity=0.0, attitude=0.0, body_rate=0.0,
                  collision=0.0, dock_success=0.0, escape=0.0)
    return RewardWeights(**{**zeroed, **weights})


def pseudo_huber(e: float, delta: float, scale: float) -> float:
    return (np.sqrt(e * e + delta * delta) - delta) / scale


def gate(d: float, far: float = 0.1, d0: float = 25.0) -> float:
    return far + (1.0 - far) / (1.0 + (d / d0) ** 2)


def test_reward_is_zero_at_the_dock_pose_at_rest():
    cfg = ISSConfig(dock=AT_ORIGIN)
    r = docking_reward(state_at((0.0, 0.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)
    assert float(r) == pytest.approx(0.0, abs=1e-6)


def test_reward_decreases_with_distance_from_the_dock():
    cfg = ISSConfig(dock=AT_ORIGIN)
    near = docking_reward(state_at((1.0, 0.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)
    far = docking_reward(state_at((10.0, 0.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)
    assert float(far) < float(near) < 0.0


def test_position_term_is_the_normalised_pseudo_huber_of_the_norm():
    cfg = ISSConfig(dock=AT_ORIGIN, reward_weights=only(position=-1.0))
    r = docking_reward(state_at((3.0, 4.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)
    # The norm, 5.0 -- not the summed square, 25.0.
    assert float(r) == pytest.approx(-pseudo_huber(5.0, 1.0, 225.0), abs=1e-6)


def test_position_term_is_linear_in_the_far_field():
    # Doubling a far-field error must roughly double the cost. A saturating
    # shape would flatten here and leave nothing pulling the chaser in.
    cfg = ISSConfig(dock=AT_ORIGIN, reward_weights=only(position=-1.0))
    at_100 = abs(float(docking_reward(state_at((100.0, 0.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)))
    at_200 = abs(float(docking_reward(state_at((200.0, 0.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)))
    assert at_200 / at_100 == pytest.approx(2.0, rel=0.01)


def test_position_term_is_quadratic_near_the_goal():
    # Inside the knee the cost falls off as the square, so halving a small
    # error quarters it -- which is what makes the last metre worth flying.
    cfg = ISSConfig(dock=AT_ORIGIN, reward_weights=only(position=-1.0))
    at_p1 = abs(float(docking_reward(state_at((0.1, 0.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)))
    at_p05 = abs(float(docking_reward(state_at((0.05, 0.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)))
    assert at_p1 / at_p05 == pytest.approx(4.0, rel=0.02)


def test_velocity_term_penalises_speed():
    cfg = ISSConfig(dock=AT_ORIGIN, reward_weights=only(velocity=-1.0))
    r = docking_reward(state_at((0.0, 0.0, 0.0), vel=(3.0, 4.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)
    assert float(r) == pytest.approx(-pseudo_huber(5.0, 0.1, 5.0), abs=1e-6)


def test_attitude_term_penalises_misalignment_against_the_dock_quaternion():
    cfg = ISSConfig(dock=AT_ORIGIN, reward_weights=only(attitude=-1.0))
    # 90 deg about x, at the port so the gate is fully open.
    quarter = (0.7071068, 0.7071068, 0.0, 0.0)
    r = docking_reward(state_at((0.0, 0.0, 0.0), quat=quarter), ZERO_ACTION, NO_EVENTS, cfg)
    assert float(r) == pytest.approx(-pseudo_huber(np.pi / 2.0, 0.05, np.pi), abs=1e-5)


def test_body_rate_term_penalises_spin():
    cfg = ISSConfig(dock=AT_ORIGIN, reward_weights=only(body_rate=-1.0))
    r = docking_reward(state_at((0.0, 0.0, 0.0), omega=(0.03, 0.04, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)
    assert float(r) == pytest.approx(-pseudo_huber(0.05, 0.005, 0.05), abs=1e-6)


def test_rotational_terms_are_gated_down_when_far_from_the_port():
    cfg = ISSConfig(dock=AT_ORIGIN, reward_weights=only(attitude=-1.0))
    quarter = (0.7071068, 0.7071068, 0.0, 0.0)
    close = abs(float(docking_reward(state_at((1.0, 0.0, 0.0), quat=quarter),
                                     ZERO_ACTION, NO_EVENTS, cfg)))
    distant = abs(float(docking_reward(state_at((225.0, 0.0, 0.0), quat=quarter),
                                       ZERO_ACTION, NO_EVENTS, cfg)))
    assert distant < close
    assert distant / close == pytest.approx(gate(225.0) / gate(1.0), rel=1e-4)


def test_rotational_gate_reaches_full_weight_at_the_port():
    cfg = ISSConfig(dock=AT_ORIGIN, reward_weights=only(attitude=-1.0))
    quarter = (0.7071068, 0.7071068, 0.0, 0.0)
    r = docking_reward(state_at((0.0, 0.0, 0.0), quat=quarter), ZERO_ACTION, NO_EVENTS, cfg)
    assert float(r) == pytest.approx(-pseudo_huber(np.pi / 2.0, 0.05, np.pi), abs=1e-5)


def test_rotational_gate_floor_is_configurable():
    cfg = ISSConfig(
        dock=AT_ORIGIN,
        reward_weights=only(attitude=-1.0),
        reward_shaping=RewardShapingConfig(rotation_gate_far=0.0),
    )
    quarter = (0.7071068, 0.7071068, 0.0, 0.0)
    # far=0 makes the gate 1/(1 + (d/d0)^2), which at 10x d0 is ~1/101.
    r = abs(float(docking_reward(state_at((250.0, 0.0, 0.0), quat=quarter),
                                 ZERO_ACTION, NO_EVENTS, cfg)))
    assert r == pytest.approx(pseudo_huber(np.pi / 2.0, 0.05, np.pi) / 101.0, rel=1e-4)


def test_a_step_of_shaped_cost_is_bounded_at_about_one_at_the_numerical_start_shell_edge():
    # 225 m is position_scale_m, the outer edge of the start shell the
    # iss-numerical variants disperse over -- not the domain bound this
    # module is shared across; see the domain-bound test below for that.
    cfg = ISSConfig(dock=AT_ORIGIN)
    worst = state_at((225.0, 0.0, 0.0), vel=(5.0, 0.0, 0.0),
                     quat=(0.0, 1.0, 0.0, 0.0), omega=(0.05, 0.0, 0.0))
    assert abs(float(docking_reward(worst, ZERO_ACTION, NO_EVENTS, cfg))) < 1.2


def test_collision_dominates_a_full_horizon_of_shaped_cost():
    cfg = ISSConfig(dock=AT_ORIGIN)
    worst = state_at((225.0, 0.0, 0.0), vel=(5.0, 0.0, 0.0),
                     quat=(0.0, 1.0, 0.0, 0.0), omega=(0.05, 0.0, 0.0))
    per_step = abs(float(docking_reward(worst, ZERO_ACTION, NO_EVENTS, cfg)))
    assert abs(cfg.reward_weights.collision) > 50.0 * per_step * cfg.max_steps


def test_collision_dominates_a_full_horizon_at_the_domain_bound():
    # PhysicsConfig.start_radius_range_m reaches 500 m and max_range_m sits
    # at 750 m, well past the 225 m edge position_scale_m is tuned to. The
    # shaped cost is bigger out there (~1.89/step, ~-13,604/rollout) but
    # collision -- at -1e6 -- still dominates by ~73x.
    cfg = ISSConfig(dock=AT_ORIGIN)
    worst = state_at((cfg.max_range_m, 0.0, 0.0), vel=(5.0, 0.0, 0.0),
                     quat=(0.0, 1.0, 0.0, 0.0), omega=(0.05, 0.0, 0.0))
    per_step = abs(float(docking_reward(worst, ZERO_ACTION, NO_EVENTS, cfg)))
    assert abs(cfg.reward_weights.collision) > 50.0 * per_step * cfg.max_steps


def test_collision_applies_the_full_penalty_weight():
    cfg = ISSConfig(dock=AT_ORIGIN)
    r = docking_reward(state_at((0.0, 0.0, 0.0)), ZERO_ACTION, HIT, cfg)
    assert float(r) == pytest.approx(cfg.reward_weights.collision, abs=1.0)


def test_docking_pays_the_success_bonus():
    cfg = ISSConfig(dock=AT_ORIGIN)
    r = docking_reward(state_at((0.0, 0.0, 0.0)), ZERO_ACTION, DOCKED, cfg)
    assert float(r) == pytest.approx(cfg.reward_weights.dock_success, abs=1e-2)


def test_docking_beats_hovering_just_outside_the_gate():
    # Without the bonus both score ~0 and there is nothing to close the last
    # half-metre for.
    cfg = ISSConfig(dock=AT_ORIGIN)
    hover = docking_reward(state_at((0.5, 0.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)
    dock = docking_reward(state_at((0.05, 0.0, 0.0)), ZERO_ACTION, DOCKED, cfg)
    assert float(dock) > float(hover)


def test_escaping_applies_the_full_penalty_weight():
    cfg = ISSConfig(dock=AT_ORIGIN)
    state = state_at((2000.0, 0.0, 0.0))
    escaped = float(docking_reward(state, ZERO_ACTION, ESCAPED, cfg))
    stayed = float(docking_reward(state, ZERO_ACTION, NO_EVENTS, cfg))
    assert escaped - stayed == pytest.approx(cfg.reward_weights.escape, abs=1e-2)


# Orthogonal to the shipped DockConfig quaternion, so the attitude error is a
# full pi -- the worst a loitering episode can hold while the escape tests
# below bound what loitering can cost.
ANTIPODAL_TO_DOCK = (0.0, 0.0, 1.0, 0.0)


def worst_start_position(cfg) -> tuple[float, float, float]:
    """The point on the start shell furthest from the port it is shaped toward.

    The shell is centred on the ISS and the port sits ~24.6 m off that centre,
    so the point diametrically opposite the port is that much further from it
    than the shell radius -- while still being a radius an episode can
    actually start at. Taking the radius alone would understate the loiter
    cost the escape penalty has to beat, in the direction that makes the tests
    below pass more easily.
    """
    dock = np.asarray(cfg.dock.position, dtype=float)
    return tuple(-dock / np.linalg.norm(dock) * cfg.physics.start_radius_range_m[1])


def test_escaping_costs_more_than_a_full_horizon_of_loitering_can_save():
    # Escape is absorbing, so an episode that exits stops paying the shaped
    # cost of the horizon it skips. The penalty has to exceed everything that
    # skipping can save, or leaving is cheaper than staying -- and the most a
    # station-keeping episode can spend is a full horizon held at rest at the
    # furthest point it can have started from.
    cfg = ISSConfig()
    at_rest = state_at(worst_start_position(cfg), quat=ANTIPODAL_TO_DOCK)
    per_step = abs(float(docking_reward(at_rest, ZERO_ACTION, NO_EVENTS, cfg)))
    assert abs(cfg.reward_weights.escape) > per_step * cfg.max_steps


def test_escaping_is_worse_than_riding_the_horizon_out_at_that_distance():
    # The same comparison as a return rather than a weight: an episode that
    # flies to the edge of the start shell and leaves must score below one
    # that sits there to the horizon. Without the penalty the escape pays one
    # step of shaped cost and the loiter pays 7200 of them, so the ordering
    # is backwards -- fleeing outscores staying by three orders.
    cfg = ISSConfig()
    at_rest = state_at(worst_start_position(cfg), quat=ANTIPODAL_TO_DOCK)
    loiter = float(docking_reward(at_rest, ZERO_ACTION, NO_EVENTS, cfg)) * cfg.max_steps
    escape = float(docking_reward(at_rest, ZERO_ACTION, ESCAPED, cfg))
    assert escape < loiter


def test_escaping_stays_far_less_bad_than_a_collision():
    # Two orders apart, so no amount of shaped cost between them lets an
    # episode confuse flying out of the domain with flying into the station.
    w = ISSConfig(dock=AT_ORIGIN).reward_weights
    assert abs(w.collision) > 50.0 * abs(w.escape)


def test_docking_and_escaping_are_symmetric():
    # Leaving is exactly as bad as arriving is good: no mixture of the two
    # over many episodes pays, whatever the ratio.
    w = ISSConfig(dock=AT_ORIGIN).reward_weights
    assert w.escape == pytest.approx(-w.dock_success)


def test_reward_does_not_depend_on_the_action():
    # `action` is accepted for terms that may come back (fuel, actuator wear)
    # and is deliberately unread today. Pin that so nobody "fixes" the unused
    # parameter by deleting it, and so a future term arrives with its own test.
    cfg = ISSConfig(dock=AT_ORIGIN)
    state = state_at((4.0, 1.0, 0.0), vel=(0.2, 0.0, 0.0), omega=(0.01, 0.0, 0.0))
    big = jnp.asarray([1600.0, -1600.0, 900.0, 2000.0, -2000.0, 500.0], dtype=jnp.float32)
    assert float(docking_reward(state, big, NO_EVENTS, cfg)) == float(
        docking_reward(state, ZERO_ACTION, NO_EVENTS, cfg)
    )


def test_default_weights_combine_every_shaped_term():
    # Each term test above zeroes the other weights, so nothing else exercises
    # the real defaults with more than one term live.
    cfg = ISSConfig(dock=AT_ORIGIN)
    w, s = cfg.reward_weights, cfg.reward_shaping
    quarter = (0.7071068, 0.7071068, 0.0, 0.0)
    state = state_at((10.0, 0.0, 0.0), vel=(0.5, 0.0, 0.0), quat=quarter, omega=(0.01, 0.0, 0.0))
    expected = (
        w.position * pseudo_huber(10.0, s.position_delta_m, s.position_scale_m)
        + w.velocity * pseudo_huber(0.5, s.velocity_delta_m_s, s.velocity_scale_m_s)
        + gate(10.0) * (
            w.attitude * pseudo_huber(np.pi / 2.0, s.attitude_delta_rad, s.attitude_scale_rad)
            + w.body_rate * pseudo_huber(0.01, s.rate_delta_rad_s, s.rate_scale_rad_s)
        )
    )
    r = docking_reward(state, ZERO_ACTION, NO_EVENTS, cfg)
    assert float(r) == pytest.approx(expected, rel=1e-4)


def test_dock_pose_argument_moves_both_the_position_and_attitude_target():
    cfg = ISSConfig(dock=AT_ORIGIN, reward_weights=only(position=-1.0, attitude=-1.0))
    quarter = (0.7071068, 0.7071068, 0.0, 0.0)
    pose = jnp.asarray([4.0, 0.0, 0.0, *quarter], dtype=jnp.float32)
    # At the port, aligned with the port's own attitude: both terms vanish.
    r = docking_reward(state_at((4.0, 0.0, 0.0), quat=quarter), ZERO_ACTION, NO_EVENTS, cfg, pose)
    assert float(r) == pytest.approx(0.0, abs=1e-6)


def test_config_dock_quaternion_is_the_default_attitude_target():
    cfg = ISSConfig(
        dock=DockConfig(position=(0.0, 0.0, 0.0), quaternion=(0.7071068, 0.7071068, 0.0, 0.0)),
        reward_weights=only(attitude=-1.0),
    )
    aligned = state_at((0.0, 0.0, 0.0), quat=(0.7071068, 0.7071068, 0.0, 0.0))
    assert float(docking_reward(aligned, ZERO_ACTION, NO_EVENTS, cfg)) == pytest.approx(0.0, abs=1e-6)


def test_reward_goal_position_override_targets_the_override():
    cfg = ISSConfig(
        dock=DockConfig(position=(1.0, 0.0, 0.0), quaternion=IDENTITY),
        reward_goal_position=(0.0, 0.0, 0.0),
        reward_weights=only(position=-1.0),
    )
    r = docking_reward(state_at((4.0, 0.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)
    assert float(r) == pytest.approx(-pseudo_huber(4.0, 1.0, 225.0), abs=1e-6)


def test_reward_goal_position_outranks_a_per_episode_dock_pose():
    cfg = ISSConfig(
        dock=DockConfig(position=(1.0, 0.0, 0.0), quaternion=IDENTITY),
        reward_goal_position=(0.0, 0.0, 0.0),
        reward_weights=only(position=-1.0),
    )
    pose = jnp.asarray([4.0, 0.0, 0.0, *IDENTITY], dtype=jnp.float32)
    r = docking_reward(state_at((4.0, 0.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg, pose)
    assert float(r) == pytest.approx(-pseudo_huber(4.0, 1.0, 225.0), abs=1e-6)


def test_reward_goal_position_does_not_override_the_attitude_target():
    # The field redirects WHERE to fly, not HOW to be oriented on arrival, so
    # the per-episode pose still supplies the quaternion.
    cfg = ISSConfig(
        dock=DockConfig(position=(1.0, 0.0, 0.0), quaternion=IDENTITY),
        reward_goal_position=(0.0, 0.0, 0.0),
        reward_weights=only(attitude=-1.0),
    )
    quarter = (0.7071068, 0.7071068, 0.0, 0.0)
    pose = jnp.asarray([9.0, 0.0, 0.0, *quarter], dtype=jnp.float32)
    aligned = state_at((0.0, 0.0, 0.0), quat=quarter)
    assert float(docking_reward(aligned, ZERO_ACTION, NO_EVENTS, cfg, pose)) == pytest.approx(
        0.0, abs=1e-6
    )


def test_reward_goal_position_does_not_override_the_attitude_target_with_no_dock_pose():
    # Same precedence with no per-episode pose in play: the override still
    # only ever redirects WHERE to fly, so cfg.dock.quaternion supplies the
    # attitude target even though reward_goal_position has moved the
    # position target away from cfg.dock.position.
    cfg = ISSConfig(
        dock=DockConfig(position=(1.0, 0.0, 0.0), quaternion=(0.7071068, 0.7071068, 0.0, 0.0)),
        reward_goal_position=(0.0, 0.0, 0.0),
        reward_weights=only(attitude=-1.0),
    )
    aligned = state_at((0.0, 0.0, 0.0), quat=(0.7071068, 0.7071068, 0.0, 0.0))
    assert float(docking_reward(aligned, ZERO_ACTION, NO_EVENTS, cfg)) == pytest.approx(
        0.0, abs=1e-6
    )


def test_penalties_are_negative_rewards():
    """The weights are negative AND the sum is not negated. Getting exactly one
    of those right turns every penalty into a reward."""
    cfg = ISSConfig(dock=AT_ORIGIN)
    far = docking_reward(state_at((10.0, 0.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)
    near = docking_reward(state_at((1.0, 0.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)
    assert float(far) < float(near) < 0.0
