import jax.numpy as jnp
import numpy as np

from owm_envs.envs.common.config import DockConfig, RewardWeights
from owm_envs.envs.common.events import Events
from owm_envs.envs.common.reward import docking_reward
from owm_envs.envs.iss.config import ISSConfig

NO_EVENTS = Events(collision=jnp.array(False), docked=jnp.array(False), escaped=jnp.array(False))
ZERO_ACTION = jnp.zeros((6,), dtype=jnp.float32)


def state_at(pos, vel=(0.0, 0.0, 0.0), omega=(0.0, 0.0, 0.0)) -> jnp.ndarray:
    return jnp.asarray([*pos, *vel, 1.0, 0.0, 0.0, 0.0, *omega], dtype=jnp.float32)


def test_reward_is_zero_at_the_dock_pose_with_no_effort():
    cfg = ISSConfig(dock=DockConfig(position=(0.0, 0.0, 0.0)))
    r = docking_reward(state_at((0.0, 0.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)
    assert np.isclose(float(r), 0.0, atol=1e-6)


def test_reward_decreases_with_distance_from_dock():
    cfg = ISSConfig(dock=DockConfig(position=(0.0, 0.0, 0.0)))
    near = docking_reward(state_at((1.0, 0.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)
    far = docking_reward(state_at((10.0, 0.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)
    assert float(far) < float(near) < 0.0


def test_default_weights_combine_multiple_terms():
    # Every other term test below sets its own weight to 1.0 and zeros the
    # other four, so nothing else exercises the *default* RewardWeights with
    # more than one term active -- e.g. swapping velocity=0.35 and
    # angular_velocity=0.1 in the defaults would leave every other test green.
    cfg = ISSConfig(dock=DockConfig(position=(0.0, 0.0, 0.0)))
    r = docking_reward(
        state_at((1.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0), omega=(1.0, 0.0, 0.0)),
        ZERO_ACTION, NO_EVENTS, cfg,
    )
    # position=-1.0*1 + velocity=-0.35*1 + angular_velocity=-0.1*1 == -1.45
    assert np.isclose(float(r), -1.45, atol=1e-4)


def test_position_term_is_summed_squared_not_norm():
    # weight -1.0 on position, everything else zeroed => r == -sum(diff**2)
    cfg = ISSConfig(
        dock=DockConfig(position=(0.0, 0.0, 0.0)),
        reward_weights=RewardWeights(position=-1.0, velocity=0.0, angular_velocity=0.0,
                                     control_effort=0.0, collision=0.0),
    )
    r = docking_reward(state_at((3.0, 4.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)
    assert np.isclose(float(r), -25.0, atol=1e-4)  # 3^2 + 4^2, not the norm 5


def test_velocity_term_penalizes_speed():
    cfg = ISSConfig(
        dock=DockConfig(position=(0.0, 0.0, 0.0)),
        reward_weights=RewardWeights(position=0.0, velocity=-1.0, angular_velocity=0.0,
                                     control_effort=0.0, collision=0.0),
    )
    r = docking_reward(state_at((0.0, 0.0, 0.0), vel=(2.0, 0.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)
    assert np.isclose(float(r), -4.0, atol=1e-4)


def test_angular_velocity_term_penalizes_spin():
    cfg = ISSConfig(
        dock=DockConfig(position=(0.0, 0.0, 0.0)),
        reward_weights=RewardWeights(position=0.0, velocity=0.0, angular_velocity=-1.0,
                                     control_effort=0.0, collision=0.0),
    )
    r = docking_reward(state_at((0.0, 0.0, 0.0), omega=(1.0, 2.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)
    assert np.isclose(float(r), -5.0, atol=1e-4)


def test_control_effort_term_penalizes_actuation():
    cfg = ISSConfig(
        dock=DockConfig(position=(0.0, 0.0, 0.0)),
        reward_weights=RewardWeights(position=0.0, velocity=0.0, angular_velocity=0.0,
                                     control_effort=-1.0, collision=0.0),
    )
    action = jnp.array([3.0, 0.0, 0.0, 4.0, 0.0, 0.0], dtype=jnp.float32)
    r = docking_reward(state_at((0.0, 0.0, 0.0)), action, NO_EVENTS, cfg)
    assert np.isclose(float(r), -25.0, atol=1e-4)


def test_collision_applies_the_full_penalty_weight():
    cfg = ISSConfig(dock=DockConfig(position=(0.0, 0.0, 0.0)))
    hit = Events(collision=jnp.array(True), docked=jnp.array(False), escaped=jnp.array(False))
    r = docking_reward(state_at((0.0, 0.0, 0.0)), ZERO_ACTION, hit, cfg)
    assert np.isclose(float(r), cfg.reward_weights.collision, atol=1.0)


def test_docking_is_not_penalized():
    cfg = ISSConfig(dock=DockConfig(position=(0.0, 0.0, 0.0)))
    docked = Events(collision=jnp.array(False), docked=jnp.array(True), escaped=jnp.array(False))
    r = docking_reward(state_at((0.0, 0.0, 0.0)), ZERO_ACTION, docked, cfg)
    assert np.isclose(float(r), 0.0, atol=1e-6)


def test_escaping_is_neither_rewarded_nor_penalized():
    # Leaving the domain ends the episode but carries no reward term of its
    # own: the existing position shaping already scores being far away, and a
    # bonus or penalty here would be a second, unweighted opinion on it.
    cfg = ISSConfig(dock=DockConfig(position=(0.0, 0.0, 0.0)))
    state = state_at((2000.0, 0.0, 0.0))
    escaped = Events(collision=jnp.array(False), docked=jnp.array(False), escaped=jnp.array(True))
    assert float(docking_reward(state, ZERO_ACTION, escaped, cfg)) == float(
        docking_reward(state, ZERO_ACTION, NO_EVENTS, cfg)
    )


def test_reward_goal_position_none_targets_the_dock_position():
    cfg = ISSConfig(
        dock=DockConfig(position=(1.0, 0.0, 0.0)),
        reward_weights=RewardWeights(position=-1.0, velocity=0.0, angular_velocity=0.0,
                                     control_effort=0.0, collision=0.0),
    )
    assert cfg.reward_goal_position is None
    r = docking_reward(state_at((4.0, 0.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)
    assert np.isclose(float(r), -9.0, atol=1e-4)  # (4-1)^2


def test_reward_goal_position_override_targets_the_override_not_the_dock():
    cfg = ISSConfig(
        dock=DockConfig(position=(1.0, 0.0, 0.0)),
        reward_goal_position=(0.0, 0.0, 0.0),
        reward_weights=RewardWeights(position=-1.0, velocity=0.0, angular_velocity=0.0,
                                     control_effort=0.0, collision=0.0),
    )
    r = docking_reward(state_at((4.0, 0.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)
    assert np.isclose(float(r), -16.0, atol=1e-4)  # (4-0)^2, not (4-1)^2


def test_dock_position_argument_moves_the_position_target():
    # A multi-port rollout hands the episode's own port position in, so the
    # reward is shaped toward the point that episode is flying to rather than
    # the single pose in DockConfig.
    cfg = ISSConfig(
        dock=DockConfig(position=(1.0, 0.0, 0.0)),
        reward_weights=RewardWeights(position=-1.0, velocity=0.0, angular_velocity=0.0,
                                     control_effort=0.0, collision=0.0),
    )
    r = docking_reward(state_at((4.0, 0.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg,
                   jnp.asarray([4.0, 0.0, 0.0], dtype=jnp.float32))
    assert np.isclose(float(r), 0.0, atol=1e-4)  # peak at the port, not at (1, 0, 0)


def test_reward_goal_position_outranks_a_per_episode_dock_position():
    # The override exists to shape the reward toward some other point
    # entirely; an assigned port does not revoke it.
    cfg = ISSConfig(
        dock=DockConfig(position=(1.0, 0.0, 0.0)),
        reward_goal_position=(0.0, 0.0, 0.0),
        reward_weights=RewardWeights(position=-1.0, velocity=0.0, angular_velocity=0.0,
                                     control_effort=0.0, collision=0.0),
    )
    r = docking_reward(state_at((4.0, 0.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg,
                   jnp.asarray([4.0, 0.0, 0.0], dtype=jnp.float32))
    assert np.isclose(float(r), -16.0, atol=1e-4)


def test_penalties_are_negative_rewards():
    """The weights are negative AND the sum is not negated. Getting exactly one
    of those right turns every penalty into a reward."""
    cfg = ISSConfig(reward_goal_position=(0.0, 0.0, 0.0))
    far = docking_reward(state_at((10.0, 0.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)
    near = docking_reward(state_at((1.0, 0.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)
    assert float(far) < float(near) < 0.0, "being further away must score worse"


def test_collision_is_catastrophic_not_rewarded():
    cfg = ISSConfig(reward_goal_position=(0.0, 0.0, 0.0))
    hit = Events(collision=jnp.array(True), docked=jnp.array(False), escaped=jnp.array(False))
    r = docking_reward(state_at((0.0, 0.0, 0.0)), ZERO_ACTION, hit, cfg)
    assert float(r) < -1000.0, "a collision must be a large NEGATIVE reward"
