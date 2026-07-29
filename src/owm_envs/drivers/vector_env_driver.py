"""Generic rollout driver: drives any gymnasium VectorEnv in a Python loop.

This is the universal path: it needs only `reset`, `step`, and an opaque
`PolicySource` (see `types.PolicySource`), so a future Basilisk or brahe
environment inherits dataset generation by implementing the Gymnasium pair
and nothing else. Neither this module nor `types.py` imports JAX or any
backend-specific type -- the policy source owns all of that, and the driver
only ever sees numpy observations, numpy actions, and an opaque per-episode
state it hands back to the policy source unexamined.

VectorEnvDriver requires its backend to implement Gymnasium NEXT_STEP
autoreset correctly: after a lane terminates or truncates, the following
`step()` call must return that lane's freshly reset observation. A backend
that declares `AutoresetMode.NEXT_STEP` in its metadata but does not actually
substitute the reset observation will fail silently -- no exception, no
warning, just episodes recorded shorter than they should be.

`gymnasium.vector.VectorEnv` has no per-lane reset API, only a whole-vector
`reset()`. That is fine when a lane's episode ends because the ENV itself
terminates or truncates it (NEXT_STEP autoreset handles that lane in
isolation, without disturbing the others). It is a problem when `spec.max_steps`
(the horizon this rollout was asked for) is shorter than the env's own
horizon: the env has no idea the driver wants that lane cut short, so nothing
would reset it. Calling the whole-vector `reset()` at that instant would
correctly reset the lane that needs it, but would also blow away every other
lane's in-progress, not-yet-finished episode -- data loss, silently.

Instead, a lane that hits `spec.max_steps` before the env itself is done gets
FROZEN: it keeps receiving a no-op action (so the batched `step()` call still
has an action for every lane) but its output is discarded and it stops
accumulating. Other lanes keep running -- including cycling through several
of their own env-driven terminate/autoreset episodes -- until every lane is
either frozen or has just been finalized by the env itself (i.e. holds no
live, unrecorded state). Only then is the whole vector env reset once,
starting every lane's next episode from a real, independent reset. No
recorded or in-progress data is ever discarded; the cost is that a lane which
finishes early idles until the slowest lane in the cohort also finishes. When
lanes stay in lockstep (the common case: no early termination, so every lane
reaches `spec.max_steps` on the same global step) that idle time is zero.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .types import PolicySource, RolloutSpec, TrajectoryBatch, pack_episodes


class VectorEnvDriver:
    """See module docstring.

    `generate()` always creates its own env from `env_factory` and always
    closes it -- on both the success and the exception path -- before
    returning or re-raising. There is no support for passing in an
    already-open, externally-owned env, so ownership is never ambiguous.
    """

    def __init__(self, env_factory: Callable[[], Any], policy_source: PolicySource):
        self.env_factory = env_factory
        self.policy_source = policy_source

    def generate(self, spec: RolloutSpec) -> TrajectoryBatch:
        if spec.num_episodes < 1:
            raise ValueError(f"num_episodes must be >= 1, got {spec.num_episodes}")
        if spec.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {spec.max_steps}")

        env = self.env_factory()
        try:
            result = self._run(env, spec)
        except BaseException:
            # A rollout or policy-source failure is the error the caller
            # needs to see -- if env.close() itself also raises here, swallow
            # that secondary failure rather than letting it mask the
            # original one.
            try:
                env.close()
            except BaseException:
                pass
            raise
        env.close()
        return result

    def _run(self, env: Any, spec: RolloutSpec) -> TrajectoryBatch:
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
        # True for a lane that hit spec.max_steps before the env itself was
        # done -- see module docstring. It keeps stepping (no-op, discarded)
        # until the whole cohort is safe to reset together.
        lane_frozen = [False] * num_envs
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
                if lane_frozen[lane] or lane_awaiting_reset[lane]:
                    # No-op: the batched step() call needs an action for
                    # every lane, but this lane's result is discarded below
                    # (frozen) or thrown away by NEXT_STEP autoreset
                    # (awaiting reset). Skip the policy call entirely rather
                    # than just discarding its output -- `obs[lane]` here is
                    # still the terminal observation of the episode that just
                    # ended, while `lane_episode_state[lane]` already belongs
                    # to the next one, so invoking the policy would consume
                    # or mutate a stateful policy's internal state on behalf
                    # of the wrong episode.
                    actions[lane] = zero_action
                    continue
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
                if lane_frozen[lane]:
                    continue

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
                # requested here) and typically much larger.
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
                        # spec.max_steps reached before the env itself was
                        # done: there is no per-lane reset API to give this
                        # lane a real fresh start, so freeze it (see module
                        # docstring) instead of letting its physics run on.
                        lane_frozen[lane] = True

            obs = next_obs

            # Once every lane is either frozen or was just finalized by the
            # env itself, none of them holds live, unrecorded state -- safe
            # to reset the whole vector env in one call and start the next
            # cohort of episodes together. Gated on `any(lane_frozen)` so
            # this never engages, and lanes keep their full independent
            # throughput, unless spec.max_steps actually forced a freeze.
            if (
                len(finished) < spec.num_episodes
                and any(lane_frozen)
                and all(lane_frozen[lane] or lane_awaiting_reset[lane] for lane in range(num_envs))
            ):
                obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
                lane_obs = [[o.copy()] for o in obs]
                lane_act = [[] for _ in range(num_envs)]
                lane_rew = [[] for _ in range(num_envs)]
                lane_step = [0] * num_envs
                lane_frozen = [False] * num_envs
                lane_awaiting_reset = [False] * num_envs

        return pack_episodes(
            finished[: spec.num_episodes], obs_dim, act_dim, records_policy_ids
        )
