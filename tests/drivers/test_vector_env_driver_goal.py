import jax.numpy as jnp
import numpy as np

from owm_envs.drivers.types import RolloutSpec
from owm_envs.drivers.vector_env_driver import VectorEnvDriver
from owm_envs.envs.common.config import ObservationConfig, dock_target
from owm_envs.envs.common.goal import dock_goal_error
from owm_envs.envs.common.policies import PolicyConfig
from owm_envs.envs.common.policy_source import TaskPolicySource
from owm_envs.envs.iss.config import ISSConfig
from owm_envs.envs.iss.vector_env import ISSVectorEnv


def _vector_driver(cfg: ISSConfig, policy_cfg: PolicyConfig, num_envs: int = 2) -> VectorEnvDriver:
    # TaskPolicySource owns the goal-error augmentation on this path, so the
    # env it drives must stay at the raw 13-dim observation -- otherwise the
    # block would be appended twice (see cli._resolve_driver's build_vector).
    env_cfg = cfg.model_copy(update={"observation": cfg.observation.model_copy(update={"goal_error": False})})
    return VectorEnvDriver(
        env_factory=lambda: ISSVectorEnv(num_envs=num_envs, cfg=env_cfg),
        policy_source=TaskPolicySource(cfg, policy_cfg),
    )


def test_vector_driver_records_policy_aware_goal_blocks():
    cfg = ISSConfig(max_steps=10, observation={"goal_error": True})
    driver = _vector_driver(cfg, PolicyConfig(type="dock"))
    batch = driver.generate(RolloutSpec(num_episodes=2, max_steps=10, seed=0))
    assert batch.observations.shape[-1] == 25
    row = batch.observations[0, 0]
    np.testing.assert_allclose(
        row[13:], np.asarray(dock_goal_error(jnp.asarray(row[:13]), dock_target(cfg))), atol=1e-5
    )


def test_vector_driver_goal_off_is_13_dim():
    cfg = ISSConfig(max_steps=10)
    driver = _vector_driver(cfg, PolicyConfig(type="dock"))
    batch = driver.generate(RolloutSpec(num_episodes=2, max_steps=10, seed=0))
    assert batch.observations.shape[-1] == 13
