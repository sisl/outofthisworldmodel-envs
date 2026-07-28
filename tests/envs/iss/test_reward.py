import jax.numpy as jnp
import numpy as np

from owm_envs.envs.iss.config import ISSConfig, RewardWeights
from owm_envs.envs.iss.dynamics import Events
from owm_envs.envs.iss.reward import iss_reward

NO_EVENTS = Events(collision=jnp.array(False), docked=jnp.array(False))
ZERO_ACTION = jnp.zeros((6,), dtype=jnp.float32)


def state_at(pos, vel=(0.0, 0.0, 0.0), omega=(0.0, 0.0, 0.0)) -> jnp.ndarray:
    return jnp.asarray([*pos, *vel, 1.0, 0.0, 0.0, 0.0, *omega], dtype=jnp.float32)


def test_reward_is_zero_at_the_dock_pose_with_no_effort():
    cfg = ISSConfig(dock_position=(0.0, 0.0, 0.0))
    r = iss_reward(state_at((0.0, 0.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)
    assert np.isclose(float(r), 0.0, atol=1e-6)


def test_reward_decreases_with_distance_from_dock():
    cfg = ISSConfig(dock_position=(0.0, 0.0, 0.0))
    near = iss_reward(state_at((1.0, 0.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)
    far = iss_reward(state_at((10.0, 0.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)
    assert float(far) < float(near) < 0.0


def test_position_term_is_summed_squared_not_norm():
    # weight 1.0 on position, everything else zeroed => r == -sum(diff**2)
    cfg = ISSConfig(
        dock_position=(0.0, 0.0, 0.0),
        reward_weights=RewardWeights(position=1.0, velocity=0.0, angular_velocity=0.0,
                                     control_effort=0.0, collision=0.0),
    )
    r = iss_reward(state_at((3.0, 4.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)
    assert np.isclose(float(r), -25.0, atol=1e-4)  # 3^2 + 4^2, not the norm 5


def test_velocity_term_penalizes_speed():
    cfg = ISSConfig(
        dock_position=(0.0, 0.0, 0.0),
        reward_weights=RewardWeights(position=0.0, velocity=1.0, angular_velocity=0.0,
                                     control_effort=0.0, collision=0.0),
    )
    r = iss_reward(state_at((0.0, 0.0, 0.0), vel=(2.0, 0.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)
    assert np.isclose(float(r), -4.0, atol=1e-4)


def test_angular_velocity_term_penalizes_spin():
    cfg = ISSConfig(
        dock_position=(0.0, 0.0, 0.0),
        reward_weights=RewardWeights(position=0.0, velocity=0.0, angular_velocity=1.0,
                                     control_effort=0.0, collision=0.0),
    )
    r = iss_reward(state_at((0.0, 0.0, 0.0), omega=(1.0, 2.0, 0.0)), ZERO_ACTION, NO_EVENTS, cfg)
    assert np.isclose(float(r), -5.0, atol=1e-4)


def test_control_effort_term_penalizes_actuation():
    cfg = ISSConfig(
        dock_position=(0.0, 0.0, 0.0),
        reward_weights=RewardWeights(position=0.0, velocity=0.0, angular_velocity=0.0,
                                     control_effort=1.0, collision=0.0),
    )
    action = jnp.array([3.0, 0.0, 0.0, 4.0, 0.0, 0.0], dtype=jnp.float32)
    r = iss_reward(state_at((0.0, 0.0, 0.0)), action, NO_EVENTS, cfg)
    assert np.isclose(float(r), -25.0, atol=1e-4)


def test_collision_applies_the_full_penalty_weight():
    cfg = ISSConfig(dock_position=(0.0, 0.0, 0.0))
    hit = Events(collision=jnp.array(True), docked=jnp.array(False))
    r = iss_reward(state_at((0.0, 0.0, 0.0)), ZERO_ACTION, hit, cfg)
    assert np.isclose(float(r), -cfg.reward_weights.collision, atol=1.0)


def test_docking_is_not_penalized():
    cfg = ISSConfig(dock_position=(0.0, 0.0, 0.0))
    docked = Events(collision=jnp.array(False), docked=jnp.array(True))
    r = iss_reward(state_at((0.0, 0.0, 0.0)), ZERO_ACTION, docked, cfg)
    assert np.isclose(float(r), 0.0, atol=1e-6)
