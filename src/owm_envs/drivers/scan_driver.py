"""Fused JAX rollout driver.

vmap over environments, lax.scan over time: an entire batched horizon compiles
to one call, with no Python-level per-timestep overhead. This requires a
JAX-traceable backend, so it is an OPT-IN capability -- `VectorEnvDriver` is the
universal path and this is the fast path for backends that can support it.

Termination comes from Events plus a step counter; it is not encoded as an
extra absorbing element appended to the state vector.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from ..envs.iss.config import ISSConfig
from ..envs.iss.dynamics import ISSDynamics
from ..envs.iss.goal import make_augment
from ..envs.iss.policies import EXTRAS_DIM, PolicyConfig, dock_target_selector, make_policy
from ..envs.iss.reward import iss_reward
from ..envs.iss.sensing import NOISE_STREAM, apply_sensor_noise
from .types import TRANSITIONS_STREAM, RolloutSpec, TrajectoryBatch, pack_episodes

_UNION_POLICY_IDX = 0


def supports_fused_rollout(backend: object) -> bool:
    """Whether a backend can be traced into a fused scan.

    Backends that cannot be traced (a Rust or C++ simulator behind Python
    bindings) set this False and fall back to VectorEnvDriver with no change
    at the call site.
    """
    return bool(getattr(backend, "supports_fused_rollout", True))


class ScanDriver:
    def __init__(self, cfg: ISSConfig, policy_cfg: PolicyConfig, num_envs: int = 8):
        if int(num_envs) < 1:
            # A non-positive lane count would otherwise reach the horizon
            # calculation below (division by num_envs -> ZeroDivisionError)
            # or produce invalid JAX shapes for negatives.
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self.cfg = cfg
        self.policy_cfg = policy_cfg
        self.num_envs = int(num_envs)
        self.dynamics = ISSDynamics(cfg)
        if not supports_fused_rollout(self.dynamics):
            raise TypeError(
                "ScanDriver requires a JAX-traceable backend; use VectorEnvDriver instead."
            )

    def generate(self, spec: RolloutSpec) -> TrajectoryBatch:
        if spec.num_episodes is not None and spec.num_episodes < 1:
            raise ValueError(f"num_episodes must be >= 1, got {spec.num_episodes}")
        if spec.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {spec.max_steps}")

        # An episode ends at whichever limit comes first: the horizon this
        # rollout asked for, or the environment's own configured step limit.
        # ISSEnv and ISSVectorEnv both truncate at cfg.max_steps, so honouring
        # spec.max_steps alone would make this path produce longer episodes
        # than the identical config produces through the Gymnasium adapters --
        # the same request answered differently depending on --driver.
        max_steps = min(spec.max_steps, self.cfg.max_steps)
        records_policy_ids = self.policy_cfg.type == "union"

        # Horizon long enough that num_envs lanes yield at least
        # `needed_episodes` episodes even if every episode runs the full
        # max_steps. Episodes mode sizes to the requested count; transitions
        # mode sizes to one episode per lane (ceil(num_envs/num_envs) == 1)
        # since it loops chunks until enough transitions accumulate. The
        # runner is built once here and reused across every chunk (see
        # _build_runner) so a multi-chunk transitions run compiles the scan
        # once instead of once per chunk -- JAX's jit cache is keyed on the
        # function object, and a fresh closure per chunk would never hit it.
        needed_episodes = spec.num_episodes if spec.min_transitions is None else self.num_envs
        horizon = int(np.ceil(needed_episodes / self.num_envs)) * max_steps
        runner, sample_extras = self._build_runner(max_steps, horizon)

        if spec.min_transitions is not None:
            return self._generate_transitions(
                spec, runner, sample_extras, horizon, records_policy_ids
            )

        # Single chunk sized to guarantee at least spec.num_episodes complete
        # episodes; seeding mirrors VectorEnvDriver exactly (see _run_chunk).
        base_rng = np.random.default_rng(spec.seed)
        episodes = self._run_chunk(
            base_rng, runner, sample_extras, spec.num_episodes, records_policy_ids
        )
        if len(episodes) < spec.num_episodes:
            raise RuntimeError(
                f"scan horizon produced only {len(episodes)} complete episodes, "
                f"needed {spec.num_episodes}"
            )
        obs_dim = episodes[0]["obs"].shape[-1]
        act_dim = episodes[0]["act"].shape[-1]
        return pack_episodes(
            episodes[: spec.num_episodes],
            obs_dim=obs_dim,
            act_dim=act_dim,
            records_policy_ids=records_policy_ids,
            records_dock_targets=True,
            records_true_observations=True,
        )

    def _generate_transitions(
        self,
        spec: RolloutSpec,
        runner,
        sample_extras,
        horizon: int,
        records_policy_ids: bool,
    ) -> TrajectoryBatch:
        """min_transitions mode: run chunks of `self.num_envs` episodes each
        (every lane completes >= 1 episode per chunk by construction) until
        enough transitions accumulate, taking episodes one by one and
        stopping mid-chunk at the first episode that reaches the target --
        surplus episodes in that chunk are discarded.

        `runner` and `sample_extras` are built once in `generate` (see
        `_build_runner`) and reused across every chunk here.
        """
        collected: list[dict] = []
        total_transitions = 0
        chunk_index = 0
        # Each chunk contributes at least self.num_envs transitions (every
        # lane completes >= 1 episode per chunk by construction), so this
        # many chunks is a worst-case-safe upper bound on reaching the
        # target -- tripping it indicates a bug, not an unreachable target.
        max_chunks = max(1, math.ceil(spec.min_transitions / self.num_envs)) + 1
        # No early stop within a chunk: every completed episode is needed for
        # this loop's own transitions accounting above, unlike episodes mode
        # which can stop segmenting once it has spec.num_episodes.
        wanted = self.num_envs * horizon
        while True:
            if chunk_index > max_chunks:
                raise RuntimeError(
                    "transitions accumulation exceeded the worst-case chunk "
                    f"bound -- this indicates a bug, not an unreachable target: "
                    f"only {total_transitions} of {spec.min_transitions} "
                    f"transitions produced after {chunk_index} chunks"
                )
            base_rng = np.random.default_rng([spec.seed, TRANSITIONS_STREAM, chunk_index])
            for episode in self._run_chunk(
                base_rng, runner, sample_extras, wanted, records_policy_ids
            ):
                collected.append(episode)
                total_transitions += len(episode["obs"]) - 1
                if total_transitions >= spec.min_transitions:
                    break
            if total_transitions >= spec.min_transitions:
                break
            chunk_index += 1

        obs_dim = collected[0]["obs"].shape[-1]
        act_dim = collected[0]["act"].shape[-1]
        return pack_episodes(
            collected,
            obs_dim=obs_dim,
            act_dim=act_dim,
            records_policy_ids=records_policy_ids,
            records_dock_targets=True,
            records_true_observations=True,
        )

    def _build_runner(self, max_steps: int, horizon: int):
        """Build the compiled per-chunk scan runner, once per `generate()` call.

        Every chunk of a run (episodes mode has one, transitions mode loops
        several) needs the same closures and the same `jax.jit(jax.vmap(...))`
        wrapper; building them here and reusing the result across chunks
        means the scan compiles once per `generate()` instead of once per
        chunk -- JAX's jit cache is keyed on the function object, so a fresh
        closure per chunk would recompile an identical program every time.

        Returns `(runner, sample_extras)`: `runner` maps per-lane initial
        state/index/extras/keys to the emitted per-lane scan output, and
        `sample_extras` (needed by `_run_chunk` to build each chunk's initial
        extras) is exposed separately because it is also called outside the
        scan.
        """
        policy_fn, extras_fn = make_policy(self.cfg, self.policy_cfg)
        augment = make_augment(self.cfg, self.policy_cfg)
        # The episode's assigned port, resolved from the same extras the
        # control law and the goal-error block read, so both dock success and
        # the reward are scored against the port the episode was actually
        # flying to.
        select_dock_target = dock_target_selector(self.cfg, self.policy_cfg)
        extras_width = EXTRAS_DIM[self.policy_cfg.type]
        dynamics, cfg = self.dynamics, self.cfg
        noise = cfg.sensor_noise
        observe_measurement = self.policy_cfg.observe == "measurement"

        force = cfg.control.limit_force_n
        torque = cfg.control.limit_torque_nm
        ctrl_low = jnp.array([-force] * 3 + [-torque] * 3, dtype=jnp.float32)
        ctrl_high = -ctrl_low

        def sample_extras(key):
            if extras_fn is None:
                return jnp.zeros((extras_width,), dtype=jnp.float32)
            return extras_fn(key)

        def per_env_step(carry, _):
            state, step_index, extras, key, noise_key = carry

            # Measured pre-step state, drawn BEFORE the action so the
            # recorded observation and the policy's (optional) input are the
            # SAME draw. Noise draws come from `noise_key`, a fold_in side
            # stream (see `generate`) kept separate from `key` so that the
            # act/reset/extras draws below are identical whether or not
            # noise is enabled -- `jax.random.split(key, 4)` never changes.
            if noise.enabled:
                noise_key, meas_key = jax.random.split(noise_key)
                measured = apply_sensor_noise(state, meas_key, noise)
            else:
                measured = state

            key, act_key, reset_key, extras_key = jax.random.split(key, 4)

            policy_input = measured if observe_measurement else state
            action = jnp.clip(policy_fn(policy_input, act_key, extras), ctrl_low, ctrl_high)
            dock_pose = select_dock_target(extras)
            next_state, events = dynamics.step(state, action, dock_pose)
            reward = iss_reward(next_state, action, events, cfg, dock_pose[0:3])

            # `measured_next` is its own draw (`next_meas_key`) because the
            # terminal observation on the `done` iteration is `next_state`,
            # which needs measuring too. On non-terminal iterations this
            # draw is discarded (the next iteration measures the same state
            # afresh as its pre-step `measured`) -- redundant work, accepted
            # to keep the carry simple; the draw sequence stays per-state
            # deterministic either way.
            if noise.enabled:
                noise_key, next_meas_key = jax.random.split(noise_key)
                measured_next = apply_sensor_noise(next_state, next_meas_key, noise)
            else:
                measured_next = next_state

            next_index = step_index + 1
            terminated = jnp.logical_or(events.collision, events.docked)
            truncated = jnp.logical_and(~terminated, next_index >= max_steps)
            done = jnp.logical_or(terminated, truncated)

            # Emit the MEASURED pre-step observation (`measured`) alongside
            # `measured_next`: the segmenter needs the terminal measurement
            # too, because on the iteration where `done` fires the terminal
            # observation is `measured_next`, not anything the following
            # iteration emits (that iteration already holds the
            # post-autoreset state for the new episode). Both collapse to
            # `state`/`next_state` when noise is disabled.
            #
            # Both are augmented with the CURRENT `extras` (the episode's
            # active sub-policy) -- on the `done` iteration `measured_next`
            # is this episode's terminal observation, not the fresh episode's,
            # so it must use `extras` from before the autoreset swap below.
            if augment is not None:
                obs_out, obs_next_out = augment(measured, extras), augment(measured_next, extras)
            else:
                obs_out, obs_next_out = measured, measured_next
            # `state`/`next_state` ride along un-noised and un-augmented: the
            # true dynamics state behind each emitted observation, on the same
            # pre-step/terminal layout so the segmenter can cut it identically.
            emitted = (obs_out, obs_next_out, action, reward, terminated, truncated, done,
                       extras, dock_pose, state, next_state)

            # In-scan autoreset: a done lane starts a fresh episode on the next
            # iteration, with newly sampled extras.
            fresh_state = dynamics.reset(reset_key)
            fresh_extras = sample_extras(extras_key)
            new_state = jnp.where(done, fresh_state, next_state)
            new_extras = jnp.where(done, fresh_extras, extras)
            new_index = jnp.where(done, 0, next_index)

            return (new_state, new_index, new_extras, key, noise_key), emitted

        def run_lane(state, index, extras, lane_key, noise_lane_key):
            _, emitted = jax.lax.scan(
                per_env_step,
                (state, index, extras, lane_key, noise_lane_key),
                None,
                length=horizon,
            )
            return emitted

        runner = jax.jit(jax.vmap(run_lane))
        return runner, sample_extras

    def _run_chunk(
        self,
        base_rng: np.random.Generator,
        runner,
        sample_extras,
        wanted: int,
        records_policy_ids: bool,
    ) -> list[dict]:
        """Run one fused scan over `self.num_envs` lanes and return up to
        `wanted` resulting episodes in completion order -- no packing.

        `runner` and `sample_extras` come from `_build_runner`, built once
        per `generate()` call and shared across every chunk.
        """
        # Mirror VectorEnvDriver's seeding exactly (a numpy Generator draws the
        # integer env seed that becomes the JAX key's origin -- see
        # ISSVectorEnv.reset / VectorEnvDriver.generate) so that, given the
        # same spec.seed, both drivers' lanes reset into the same initial
        # states. That is what makes the driver-equivalence test tractable
        # without matching every downstream PRNG draw.
        env_seed = int(base_rng.integers(0, 2**31 - 1))
        key = jax.random.PRNGKey(env_seed)
        key, subkey = jax.random.split(key)
        init_states = jax.vmap(self.dynamics.reset)(jax.random.split(subkey, self.num_envs))

        key, extras_key, scan_key = jax.random.split(key, 3)
        init_extras = jax.vmap(sample_extras)(jax.random.split(extras_key, self.num_envs))
        init_indices = jnp.zeros((self.num_envs,), dtype=jnp.int32)
        lane_keys = jax.random.split(scan_key, self.num_envs)
        # Noise draws come from their own per-lane stream, derived via
        # fold_in rather than split, so deriving it consumes nothing from
        # `lane_keys` -- the act/reset/extras draws are unaffected by
        # whether noise is enabled.
        noise_lane_keys = jax.vmap(lambda k: jax.random.fold_in(k, NOISE_STREAM))(lane_keys)

        emitted = runner(init_states, init_indices, init_extras, lane_keys, noise_lane_keys)
        return self._segment_episodes(emitted, wanted, records_policy_ids)

    @staticmethod
    def _segment_episodes(emitted, wanted: int, records_policy_ids: bool) -> list[dict]:
        """Cut the flat per-lane scan output into per-episode dicts, in
        time-major completion order.

        Iterating lane-major (all of lane 0's episodes, then lane 1's, ...)
        and stopping as soon as `wanted` is reached would exhaust the count
        from the first few lanes and never touch the rest -- requesting 10
        episodes over 8 lanes would take 2 each from lanes 0-4 and none from
        lanes 5-7, biasing the dataset toward a subset of the reset-key
        stream. Iterating time-major (t outer, lane inner) instead completes
        episodes in the same order VectorEnvDriver collects them
        chronologically across lanes, so truncating to `wanted` keeps
        coverage spread across every lane.

        Each dict also carries "lane", the lane it was cut from. pack_episodes
        ignores unknown keys; tests use it to check lane coverage.
        """
        (
            states,
            next_states,
            actions,
            rewards,
            terminated,
            truncated,
            done,
            extras,
            dock_poses,
            true_states,
            true_next_states,
        ) = (np.asarray(x) for x in emitted)
        act_dim = actions.shape[-1]
        zero_action = np.zeros((1, act_dim), dtype=np.float32)
        zero_reward = np.zeros((1,), dtype=np.float32)

        num_lanes, horizon = done.shape
        starts = [0] * num_lanes
        episodes: list[dict] = []
        for t in range(horizon):
            for lane in range(num_lanes):
                if not done[lane, t]:
                    continue
                start = starts[lane]
                episodes.append(
                    {
                        # N+1 observations: the N pre-step states plus the
                        # terminal `next_state` from the iteration where
                        # `done` fired.
                        "obs": np.concatenate(
                            [states[lane, start : t + 1], next_states[lane, t : t + 1]],
                            axis=0,
                        ),
                        # The same N+1 cut over the true states, so
                        # "true_obs"[i] is the un-noised state behind "obs"[i].
                        "true_obs": np.concatenate(
                            [true_states[lane, start : t + 1], true_next_states[lane, t : t + 1]],
                            axis=0,
                        ),
                        # N+1 actions: the N real actions plus a zero pad,
                        # matching VectorEnvDriver's convention.
                        "act": np.concatenate(
                            [actions[lane, start : t + 1], zero_action], axis=0
                        ),
                        "rew": np.concatenate(
                            [rewards[lane, start : t + 1], zero_reward], axis=0
                        ),
                        "terminated": bool(terminated[lane, t]),
                        "truncated": bool(truncated[lane, t]),
                        "policy_id": int(extras[lane, start, _UNION_POLICY_IDX])
                        if records_policy_ids
                        else 0,
                        # The row the scan itself resolved and handed to
                        # `dynamics.step`, taken at `start` (the episode's
                        # first emitted step) -- not reconstructed from the
                        # extras here, so the record cannot drift from the
                        # target the episode actually regulated toward.
                        "dock_target": dock_poses[lane, start],
                        "lane": lane,
                    }
                )
                starts[lane] = t + 1
                if len(episodes) >= wanted:
                    break
            if len(episodes) >= wanted:
                break

        return episodes
