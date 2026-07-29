"""Generic rollout driver: drives any gymnasium VectorEnv in a Python loop.

This is the universal path: it needs only `reset`, `step`, and an opaque
`PolicySource` (see `types.PolicySource`), so a future Basilisk or brahe
environment inherits dataset generation by implementing the Gymnasium pair
and nothing else. Neither this module nor `types.py` imports JAX or any
backend-specific type -- the policy source owns all of that, and the driver
only ever sees numpy observations, numpy actions, and an opaque per-episode
state it hands back to the policy source unexamined.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .types import PolicySource, RolloutSpec, TrajectoryBatch, pack_episodes


class VectorEnvDriver:
    def __init__(self, env_factory: Callable[[], Any], policy_source: PolicySource):
        self.env_factory = env_factory
        self.policy_source = policy_source

    def generate(self, spec: RolloutSpec) -> TrajectoryBatch:
        if spec.num_episodes < 1:
            raise ValueError(f"num_episodes must be >= 1, got {spec.num_episodes}")
        if spec.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {spec.max_steps}")

        env = self.env_factory()
        num_envs = env.num_envs
        obs_dim = env.single_observation_space.shape[0]
        act_dim = env.single_action_space.shape[0]
        records_policy_ids = self.policy_source.records_policy_ids

        rng = np.random.default_rng(spec.seed)

        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))

        # Per-lane episode accumulators. Each episode stores N + 1
        # observations (the seed state plus each post-step state, including
        # the terminal one) against N + 1 actions (the N real actions plus a
        # zero pad in the final slot) -- otherwise the terminal
        # collision/docking state, the very thing a world model needs to
        # learn, would never appear in the dataset. `lane_obs` is seeded with
        # each lane's reset observation up front, before any action exists.
        lane_obs: list[list[np.ndarray]] = [[o.copy()] for o in obs]
        lane_act: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
        lane_rew: list[list[float]] = [[] for _ in range(num_envs)]
        lane_step = [0] * num_envs
        # True for a lane between finalizing an env-terminated/truncated
        # episode and the following step() call, which is when NEXT_STEP
        # autoreset actually hands back the fresh reset observation.
        lane_awaiting_reset = [False] * num_envs
        lane_episode_state = [
            self.policy_source.new_episode(int(seed))
            for seed in rng.integers(0, 2**31 - 1, size=num_envs)
        ]

        finished: list[dict[str, Any]] = []
        zero_action = np.zeros((act_dim,), dtype=np.float32)

        action_low = np.asarray(env.single_action_space.low, dtype=np.float32)
        action_high = np.asarray(env.single_action_space.high, dtype=np.float32)

        while len(finished) < spec.num_episodes:
            actions = np.zeros((num_envs, act_dim), dtype=np.float32)
            for lane in range(num_envs):
                actions[lane] = np.asarray(
                    self.policy_source.act(obs[lane], lane_episode_state[lane], lane_step[lane]),
                    dtype=np.float32,
                )
            # The env clips internally but doesn't hand the clipped action
            # back, so clip here too -- what we record must match what was
            # actually applied, not the policy source's raw output.
            actions = np.clip(actions, action_low, action_high)

            next_obs, rewards, terminations, truncations, _ = env.step(actions)

            for lane in range(num_envs):
                if lane_awaiting_reset[lane]:
                    # This step's action for the lane was discarded by
                    # NEXT_STEP autoreset -- next_obs is already the real
                    # reset state. Seed the new episode with it; nothing was
                    # actually applied to this lane, so there's no
                    # transition to record.
                    lane_obs[lane] = [next_obs[lane].copy()]
                    lane_awaiting_reset[lane] = False
                    lane_step[lane] = 0
                    continue

                lane_act[lane].append(actions[lane].copy())
                lane_rew[lane].append(float(rewards[lane]))
                lane_obs[lane].append(next_obs[lane].copy())
                lane_step[lane] += 1

                # The env's own truncation is driven by its own config, which
                # is independent of spec.max_steps (the rollout horizon
                # requested here) and typically much larger. There is no
                # per-lane reset API on VectorEnv, so once a lane hits the
                # requested horizon without the env itself
                # terminating/truncating it, the driver closes out the
                # episode locally as truncated and keeps stepping that
                # lane's underlying physics into the next episode's window.
                env_done = bool(terminations[lane]) or bool(truncations[lane])
                horizon_hit = len(lane_act[lane]) >= spec.max_steps
                if env_done or horizon_hit:
                    lane_act[lane].append(zero_action.copy())
                    lane_rew[lane].append(0.0)
                    if len(finished) < spec.num_episodes:
                        finished.append(
                            {
                                "obs": np.stack(lane_obs[lane]),
                                "act": np.stack(lane_act[lane]),
                                "rew": np.asarray(lane_rew[lane], dtype=np.float32),
                                "terminated": bool(terminations[lane]),
                                "truncated": bool(truncations[lane]) or (horizon_hit and not env_done),
                                "policy_id": self.policy_source.policy_id(lane_episode_state[lane])
                                if records_policy_ids
                                else 0,
                            }
                        )
                    lane_act[lane] = []
                    lane_rew[lane] = []
                    lane_episode_state[lane] = self.policy_source.new_episode(
                        int(rng.integers(0, 2**31 - 1))
                    )
                    if env_done:
                        lane_obs[lane] = []
                        lane_awaiting_reset[lane] = True
                    else:
                        # No real reset happened: this lane's physics keeps
                        # running, so the next episode's window starts right
                        # where this one's terminal observation left off.
                        lane_obs[lane] = [next_obs[lane].copy()]
                        lane_step[lane] = 0

            obs = next_obs

        return pack_episodes(
            finished[: spec.num_episodes], obs_dim, act_dim, records_policy_ids
        )
