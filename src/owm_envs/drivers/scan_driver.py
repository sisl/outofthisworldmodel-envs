"""Fused JAX rollout driver.

vmap over environments, lax.scan over time: an entire batched horizon compiles
to one call, with no Python-level per-timestep overhead. This requires a
JAX-traceable backend, so it is an OPT-IN capability -- `VectorEnvDriver` is the
universal path and this is the fast path for backends that can support it.

Ported in structure from seamstress branch iss2,
environments/environment_parallel.py::rollout_with_policy_jax. This version has
no state_limits machinery and no 14th terminal state element; termination comes
from Events plus a step counter.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..envs.iss.config import ISSConfig
from ..envs.iss.dynamics import ISSDynamics
from ..envs.iss.policies import EXTRAS_DIM, PolicyConfig, make_policy
from ..envs.iss.reward import iss_reward
from .types import RolloutSpec, TrajectoryBatch, pack_episodes

_UNION_POLICY_IDX = 0


def supports_fused_rollout(backend: object) -> bool:
    """Whether a backend can be traced into a fused scan.

    Mirrors seamstress's `jax_compatible` flag. Backends that cannot be traced
    (a Rust or C++ simulator behind Python bindings) set this False and fall
    back to VectorEnvDriver with no change at the call site.
    """
    return bool(getattr(backend, "supports_fused_rollout", True))


class ScanDriver:
    def __init__(self, cfg: ISSConfig, policy_cfg: PolicyConfig, num_envs: int = 8):
        self.cfg = cfg
        self.policy_cfg = policy_cfg
        self.num_envs = int(num_envs)
        self.dynamics = ISSDynamics(cfg)
        if not supports_fused_rollout(self.dynamics):
            raise TypeError(
                "ScanDriver requires a JAX-traceable backend; use VectorEnvDriver instead."
            )

    def generate(self, spec: RolloutSpec) -> TrajectoryBatch:
        if spec.num_episodes < 1:
            raise ValueError(f"num_episodes must be >= 1, got {spec.num_episodes}")
        if spec.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {spec.max_steps}")

        policy_fn, extras_fn = make_policy(self.cfg, self.policy_cfg)
        extras_width = EXTRAS_DIM[self.policy_cfg.type]
        records_policy_ids = self.policy_cfg.type == "union"
        dynamics, cfg = self.dynamics, self.cfg

        force = cfg.control_limit_force_n
        torque = cfg.control_limit_torque_nm
        ctrl_low = jnp.array([-force] * 3 + [-torque] * 3, dtype=jnp.float32)
        ctrl_high = -ctrl_low

        def sample_extras(key):
            if extras_fn is None:
                return jnp.zeros((extras_width,), dtype=jnp.float32)
            return extras_fn(key)

        def per_env_step(carry, _):
            state, step_index, extras, key = carry
            key, act_key, reset_key, extras_key = jax.random.split(key, 4)

            action = jnp.clip(policy_fn(state, act_key, extras), ctrl_low, ctrl_high)
            next_state, events = dynamics.step(state, action)
            reward = iss_reward(next_state, action, events, cfg)

            next_index = step_index + 1
            terminated = jnp.logical_or(events.collision, events.docked)
            truncated = jnp.logical_and(~terminated, next_index >= spec.max_steps)
            done = jnp.logical_or(terminated, truncated)

            # Emit the PRE-step state (`state`) alongside `next_state`: the
            # segmenter needs `next_state` too, because on the iteration where
            # `done` fires the terminal observation is `next_state`, not
            # anything the following iteration emits (that iteration already
            # holds the post-autoreset state for the new episode).
            emitted = (state, next_state, action, reward, terminated, truncated, done, extras)

            # In-scan autoreset: a done lane starts a fresh episode on the next
            # iteration, with newly sampled extras, exactly as seamstress does.
            fresh_state = dynamics.reset(reset_key)
            fresh_extras = sample_extras(extras_key)
            new_state = jnp.where(done, fresh_state, next_state)
            new_extras = jnp.where(done, fresh_extras, extras)
            new_index = jnp.where(done, 0, next_index)

            return (new_state, new_index, new_extras, key), emitted

        # Horizon long enough that num_envs lanes yield at least num_episodes
        # episodes even if every episode runs the full max_steps.
        horizon = int(np.ceil(spec.num_episodes / self.num_envs)) * spec.max_steps

        # Mirror VectorEnvDriver's seeding exactly (a numpy Generator draws the
        # integer env seed that becomes the JAX key's origin -- see
        # ISSVectorEnv.reset / VectorEnvDriver.generate) so that, given the
        # same spec.seed, both drivers' lanes reset into the same initial
        # states. That is what makes the driver-equivalence test tractable
        # without matching every downstream PRNG draw.
        rng = np.random.default_rng(spec.seed)
        env_seed = int(rng.integers(0, 2**31 - 1))
        key = jax.random.PRNGKey(env_seed)
        key, subkey = jax.random.split(key)
        init_states = jax.vmap(dynamics.reset)(jax.random.split(subkey, self.num_envs))

        key, extras_key, scan_key = jax.random.split(key, 3)
        init_extras = jax.vmap(sample_extras)(jax.random.split(extras_key, self.num_envs))
        init_indices = jnp.zeros((self.num_envs,), dtype=jnp.int32)
        lane_keys = jax.random.split(scan_key, self.num_envs)

        def run_lane(state, index, extras, lane_key):
            _, emitted = jax.lax.scan(
                per_env_step, (state, index, extras, lane_key), None, length=horizon
            )
            return emitted

        emitted = jax.jit(jax.vmap(run_lane))(init_states, init_indices, init_extras, lane_keys)
        return self._segment(emitted, spec, records_policy_ids)

    @staticmethod
    def _segment(emitted, spec: RolloutSpec, records_policy_ids: bool) -> TrajectoryBatch:
        """Cut the flat per-lane scan output into episodes at the `done` flags."""
        states, next_states, actions, rewards, terminated, truncated, done, extras = (
            np.asarray(x) for x in emitted
        )
        obs_dim = states.shape[-1]
        act_dim = actions.shape[-1]
        zero_action = np.zeros((1, act_dim), dtype=np.float32)
        zero_reward = np.zeros((1,), dtype=np.float32)

        episodes = []
        num_lanes, horizon = done.shape
        for lane in range(num_lanes):
            start = 0
            for t in range(horizon):
                if done[lane, t]:
                    episodes.append(
                        {
                            # N+1 observations: the N pre-step states plus the
                            # terminal `next_state` from the iteration where
                            # `done` fired.
                            "obs": np.concatenate(
                                [states[lane, start : t + 1], next_states[lane, t : t + 1]],
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
                        }
                    )
                    start = t + 1
                    if len(episodes) >= spec.num_episodes:
                        break
            if len(episodes) >= spec.num_episodes:
                break

        if len(episodes) < spec.num_episodes:
            raise RuntimeError(
                f"scan horizon produced only {len(episodes)} complete episodes, "
                f"needed {spec.num_episodes}"
            )

        return pack_episodes(
            episodes[: spec.num_episodes],
            obs_dim=obs_dim,
            act_dim=act_dim,
            records_policy_ids=records_policy_ids,
        )
