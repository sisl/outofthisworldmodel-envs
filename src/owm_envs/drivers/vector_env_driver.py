"""Generic rollout driver: drives any gymnasium VectorEnv in a Python loop.

This is the universal path. It needs only `reset` and `step`, so a future
Basilisk or brahe environment inherits dataset generation by implementing the
Gymnasium pair and nothing else. It deliberately does not import JAX.

The policy is still JAX here because the ISS policies are, but the driver only
ever sees `policy_fn(state, key, extras) -> action` as an opaque callable.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..envs.iss.config import ISSConfig
from ..envs.iss.policies import EXTRAS_DIM, PolicyConfig, make_policy
from .types import RolloutSpec, TrajectoryBatch, pack_episodes

# Sub-policy index lives at extras[0] for the union mixture (see policies.py).
_UNION_POLICY_IDX = 0


class VectorEnvDriver:
    def __init__(
        self,
        env_factory: Callable[[], Any],
        cfg: ISSConfig,
        policy_cfg: PolicyConfig,
    ):
        self.env_factory = env_factory
        self.cfg = cfg
        self.policy_cfg = policy_cfg

    def generate(self, spec: RolloutSpec) -> TrajectoryBatch:
        if spec.num_episodes < 1:
            raise ValueError(f"num_episodes must be >= 1, got {spec.num_episodes}")
        if spec.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {spec.max_steps}")

        import jax  # local: keeps the module importable for non-JAX backends

        env = self.env_factory()
        num_envs = env.num_envs
        obs_dim = env.single_observation_space.shape[0]
        act_dim = env.single_action_space.shape[0]

        policy_fn, extras_fn = make_policy(self.cfg, self.policy_cfg)
        extras_width = EXTRAS_DIM[self.policy_cfg.type]
        records_policy_ids = self.policy_cfg.type == "union"

        key = jax.random.PRNGKey(spec.seed)
        key, reset_key = jax.random.split(key)

        obs, _ = env.reset(seed=int(jax.random.randint(reset_key, (), 0, 2**31 - 1)))

        # Per-lane episode accumulators.
        lane_obs: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
        lane_act: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
        lane_rew: list[list[float]] = [[] for _ in range(num_envs)]
        key, *lane_keys = jax.random.split(key, num_envs + 1)
        lane_extras = [self._sample_extras(extras_fn, k, extras_width) for k in lane_keys]

        finished: list[dict[str, Any]] = []

        action_low = np.asarray(env.action_space.low, dtype=np.float32)
        action_high = np.asarray(env.action_space.high, dtype=np.float32)

        while len(finished) < spec.num_episodes:
            actions = np.zeros((num_envs, act_dim), dtype=np.float32)
            key, *act_keys = jax.random.split(key, num_envs + 1)
            for lane in range(num_envs):
                actions[lane] = np.asarray(
                    policy_fn(obs[lane], act_keys[lane], lane_extras[lane]), dtype=np.float32
                )
            # The env clips internally but doesn't hand the clipped action
            # back, so clip here too -- what we record must match what was
            # actually applied, not the scripted policy's raw PD output.
            actions = np.clip(actions, action_low, action_high)

            next_obs, rewards, terminations, truncations, _ = env.step(actions)

            for lane in range(num_envs):
                lane_obs[lane].append(obs[lane].copy())
                lane_act[lane].append(actions[lane].copy())
                lane_rew[lane].append(float(rewards[lane]))

                # The env's own truncation is driven by cfg.max_steps, which is
                # independent of spec.max_steps (the rollout horizon requested
                # here) and typically much larger. There is no per-lane reset
                # API on VectorEnv, so once a lane hits the requested horizon
                # without the env itself terminating/truncating it, the driver
                # closes out the episode locally as truncated and keeps
                # stepping that lane's underlying physics into the next
                # episode's window.
                env_done = bool(terminations[lane]) or bool(truncations[lane])
                horizon_hit = len(lane_obs[lane]) >= spec.max_steps
                if env_done or horizon_hit:
                    if len(finished) < spec.num_episodes:
                        finished.append(
                            {
                                "obs": np.stack(lane_obs[lane]),
                                "act": np.stack(lane_act[lane]),
                                "rew": np.asarray(lane_rew[lane], dtype=np.float32),
                                "terminated": bool(terminations[lane]),
                                "truncated": bool(truncations[lane]) or (horizon_hit and not env_done),
                                "policy_id": int(lane_extras[lane][_UNION_POLICY_IDX])
                                if records_policy_ids
                                else 0,
                            }
                        )
                    lane_obs[lane].clear()
                    lane_act[lane].clear()
                    lane_rew[lane].clear()
                    key, subkey = jax.random.split(key)
                    lane_extras[lane] = self._sample_extras(extras_fn, subkey, extras_width)

            obs = next_obs

        return pack_episodes(
            finished[: spec.num_episodes], obs_dim, act_dim, records_policy_ids
        )

    @staticmethod
    def _sample_extras(extras_fn, key, width):
        import jax.numpy as jnp

        if extras_fn is None:
            return jnp.zeros((width,), dtype=jnp.float32)
        return extras_fn(key)
