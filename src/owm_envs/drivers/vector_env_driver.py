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

from .types import (
    EPOCH_STREAM,
    TRANSITIONS_STREAM,
    PolicySource,
    RolloutSpec,
    TrajectoryBatch,
    pack_episodes,
)


def _sample_epoch_offset(seed: int, orbit_cfg: Any) -> float:
    """Per-episode epoch offset, uniform in `orbit_cfg.epoch_offset_range_s`.

    Derived from `seed` -- the SAME integer handed to
    `policy_source.new_episode` for that episode -- via a dedicated numpy
    Generator salted by EPOCH_STREAM. This draws nothing from the driver's
    own `rng`: orbit-disabled runs (which never call this) are completely
    unaffected, and a run at a given spec.seed always reproduces the same
    offsets.
    """
    lo, hi = orbit_cfg.epoch_offset_range_s
    return float(np.random.default_rng([seed, EPOCH_STREAM]).uniform(lo, hi))


def _lane_info(info: dict, lane: int) -> dict:
    """One lane's view of a Gymnasium vector info dict.

    Vector infos may legally contain recursively batched sub-dicts, so
    nested dicts recurse; every other value is indexed per lane.
    """
    return {
        key: _lane_info(value, lane) if isinstance(value, dict) else value[lane]
        for key, value in info.items()
    }


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

    def _augment(self, observation: np.ndarray, episode_state: Any) -> np.ndarray:
        """The observation to RECORD -- run through the policy source's
        augment hook. Note this is never what `policy_source.act()` sees;
        `act()` above is always called with the raw `obs[lane]`."""
        return np.asarray(
            self.policy_source.augment_observation(observation.copy(), episode_state),
            dtype=np.float32,
        )

    def generate(self, spec: RolloutSpec) -> TrajectoryBatch:
        if spec.num_episodes is not None and spec.num_episodes < 1:
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
        act_dim = env.single_action_space.shape[0]
        records_policy_ids = self.policy_source.records_policy_ids
        # Duck-typed rather than importing an ISS-specific config type here:
        # this driver stays backend-agnostic (see module docstring), so it
        # only ever reads `env.cfg.orbit` if the backend happens to expose
        # it -- a plain Gymnasium VectorEnv without a `.cfg` attribute (e.g.
        # the fakes in this module's tests) never records epoch offsets.
        orbit_cfg = getattr(getattr(env, "cfg", None), "orbit", None)
        records_epoch_offsets = bool(getattr(orbit_cfg, "enabled", False))

        # Salted in transitions mode so its stream doesn't collide with the
        # episodes-mode stream at the same spec.seed (see TRANSITIONS_STREAM).
        rng = (
            np.random.default_rng([spec.seed, TRANSITIONS_STREAM])
            if spec.min_transitions is not None
            else np.random.default_rng(spec.seed)
        )

        obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))

        # Episode state is created before `lane_obs` below (rather than after,
        # as the accumulator list order might suggest) purely so the initial
        # observations can be augmented against it -- it draws from `rng` no
        # differently than before, so this reorder changes no seed's result.
        lane_seeds = rng.integers(0, 2**31 - 1, size=num_envs)
        lane_episode_state = [self.policy_source.new_episode(int(s)) for s in lane_seeds]
        # Sampled from the SAME seed as the episode state above, via an
        # independent numpy Generator (see `_sample_epoch_offset`) -- this
        # draws nothing further from `rng`, so orbit-disabled runs are
        # unaffected and the mapping seed -> offset is reproducible.
        lane_epoch_offset = (
            [_sample_epoch_offset(int(s), orbit_cfg) for s in lane_seeds]
            if records_epoch_offsets
            else [0.0] * num_envs
        )

        # Per-lane episode accumulators. Each episode stores N + 1
        # observations (the seed state plus each post-step state, including
        # the terminal one) against N + 1 actions (the N real actions plus a
        # zero pad in the final slot) -- otherwise the terminal
        # collision/docking state, the very thing a world model needs to
        # learn, would never appear in the dataset. `lane_obs` is seeded with
        # each lane's reset observation up front, before any action exists.
        #
        # Every observation stored here goes through
        # `policy_source.augment_observation()` first (identity unless the
        # source appends something, e.g. ISSPolicySource's goal-error block)
        # -- `obs_dim` is measured from that augmented width, not the env's
        # raw observation space, since the two can legitimately differ.
        lane_obs: list[list[np.ndarray]] = [
            [self._augment(o, lane_episode_state[lane])] for lane, o in enumerate(obs)
        ]
        obs_dim = lane_obs[0][0].shape[0]
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

        finished: list[dict[str, Any]] = []
        # Only meaningful in min_transitions mode, but tracked unconditionally
        # since it's cheap and keeps the quota check below mode-agnostic --
        # sum(len(e["obs"]) - 1 for e in finished), maintained incrementally
        # instead of recomputed, matching TrajectoryBatch.total_transitions.
        transitions_collected = 0
        zero_action = np.zeros((act_dim,), dtype=np.float32)

        def quota_met() -> bool:
            if spec.num_episodes is not None:
                return len(finished) >= spec.num_episodes
            return transitions_collected >= spec.min_transitions

        action_low = np.asarray(env.single_action_space.low, dtype=np.float32)
        action_high = np.asarray(env.single_action_space.high, dtype=np.float32)

        while not quota_met():
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
                lane_info = _lane_info(info, lane)
                actions[lane] = np.asarray(
                    self.policy_source.act(
                        obs[lane], lane_episode_state[lane], lane_step[lane], lane_info
                    ),
                    dtype=np.float32,
                )
            # The env clips internally but doesn't hand the clipped action
            # back, so clip here too -- what we record must match what was
            # actually applied, not the policy source's raw output.
            actions = np.clip(actions, action_low, action_high)

            next_obs, rewards, terminations, truncations, next_info = env.step(actions)

            for lane in range(num_envs):
                if lane_frozen[lane]:
                    continue

                if lane_awaiting_reset[lane]:
                    # This step's action for the lane was discarded by
                    # NEXT_STEP autoreset -- next_obs is already the real
                    # reset state. Seed the new episode with it; nothing was
                    # actually applied to this lane, so there's no
                    # transition to record. `lane_episode_state[lane]` is
                    # already this new episode's state (set when the previous
                    # one was finalized below), so it's the right state to
                    # augment against.
                    lane_obs[lane] = [self._augment(next_obs[lane], lane_episode_state[lane])]
                    lane_awaiting_reset[lane] = False
                    lane_step[lane] = 0
                    continue

                lane_act[lane].append(actions[lane].copy())
                lane_rew[lane].append(float(rewards[lane]))
                lane_obs[lane].append(self._augment(next_obs[lane], lane_episode_state[lane]))
                lane_step[lane] += 1

                # The env's own truncation is driven by its own config, which
                # is independent of spec.max_steps (the rollout horizon
                # requested here) and typically much larger.
                env_done = bool(terminations[lane]) or bool(truncations[lane])
                horizon_hit = len(lane_act[lane]) >= spec.max_steps
                if env_done or horizon_hit:
                    lane_act[lane].append(zero_action.copy())
                    lane_rew[lane].append(0.0)
                    if not quota_met():
                        # Whole episodes only, first-crossing included: this
                        # guard (not just the outer while) matters because
                        # several lanes can finish within the same step()
                        # call -- once the quota is met mid-loop, later lanes
                        # in this same pass must NOT be recorded too.
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
                                # This lane's just-finished episode's offset
                                # -- `lane_epoch_offset[lane]` isn't
                                # reassigned to the NEXT episode's value
                                # until immediately below.
                                "epoch_offset": lane_epoch_offset[lane],
                            }
                        )
                        transitions_collected += len(lane_obs[lane]) - 1
                    lane_act[lane] = []
                    lane_rew[lane] = []
                    new_seed = int(rng.integers(0, 2**31 - 1))
                    lane_episode_state[lane] = self.policy_source.new_episode(new_seed)
                    if records_epoch_offsets:
                        lane_epoch_offset[lane] = _sample_epoch_offset(new_seed, orbit_cfg)
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
            info = next_info

            # Once every lane is either frozen or was just finalized by the
            # env itself, none of them holds live, unrecorded state -- safe
            # to reset the whole vector env in one call and start the next
            # cohort of episodes together. Gated on `any(lane_frozen)` so
            # this never engages, and lanes keep their full independent
            # throughput, unless spec.max_steps actually forced a freeze.
            if (
                not quota_met()
                and any(lane_frozen)
                and all(lane_frozen[lane] or lane_awaiting_reset[lane] for lane in range(num_envs))
            ):
                obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
                # Every lane here is either frozen or awaiting-reset, so
                # `lane_episode_state[lane]` already holds each lane's next
                # episode's state (set when it was finalized above) -- the
                # same state this reset's observation belongs to.
                lane_obs = [
                    [self._augment(o, lane_episode_state[lane])] for lane, o in enumerate(obs)
                ]
                lane_act = [[] for _ in range(num_envs)]
                lane_rew = [[] for _ in range(num_envs)]
                lane_step = [0] * num_envs
                lane_frozen = [False] * num_envs
                lane_awaiting_reset = [False] * num_envs

        # The slice is a defensive no-op kept for clarity: the append guard
        # above already stops recording at quota in both modes, so `finished`
        # never actually exceeds spec.num_episodes by the time we get here.
        episodes = finished if spec.num_episodes is None else finished[: spec.num_episodes]
        return pack_episodes(
            episodes, obs_dim, act_dim, records_policy_ids,
            records_epoch_offsets=records_epoch_offsets,
        )
