"""Gymnasium vector-environment adapter for the iss-numerical docking task.

Subclasses gymnasium.vector.VectorEnv directly rather than wrapping N single
envs in SyncVectorEnv -- the physics is natively batched via jax.vmap, and
SyncVectorEnv would discard that. Structurally mirrors
`envs/iss_hcw/vector_env.py`; see `env.py`'s module docstring for how the 21D
float64 `NumericalDynamics` state and its `observation.mode` change the f32
boundary and the observation width versus `iss_hcw`.

Autoreset is NEXT_STEP: when a sub-env terminates or truncates, the step after
that returns its reset observation with reward 0 and both flags false.
"""

from __future__ import annotations

import secrets
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from ..common.adapter import action_space, render_fps
from ..common.config import dock_target
from ..common.goal import dock_goal_error
from ..common.reward import docking_reward
from ..common.sensing import NOISE_STREAM, apply_sensor_noise
from .config import NUM_LAYOUT, NumericalConfig
from .dynamics import NumericalDynamics, relative_view
from .env import _observation_space
from .observe import make_observe


class NumericalVectorEnv(VectorEnv):
    # render_fps is overridden per instance in __init__; the class-level value
    # is the rate implied by NumericalConfig's own default dt.
    metadata = {
        "render_modes": [],
        "render_fps": 20,
        "autoreset_mode": AutoresetMode.NEXT_STEP,
    }

    def __init__(self, num_envs: int = 1, cfg: NumericalConfig | None = None):
        self.cfg = cfg or NumericalConfig()
        # Per-instance because it depends on cfg.dt, which the class does not know.
        self.metadata = {**self.metadata, "render_fps": render_fps(self.cfg)}
        self.num_envs = int(num_envs)
        self.dynamics = NumericalDynamics(self.cfg)

        self.single_observation_space = _observation_space(self.cfg)
        self.single_action_space = action_space(self.cfg)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        self._states: jnp.ndarray | None = None
        self._step_index = np.zeros(self.num_envs, dtype=np.int64)
        self._needs_reset = np.zeros(self.num_envs, dtype=bool)
        self._key: jax.Array | None = None
        # Side stream for sensor-noise draws, derived from self._key via
        # fold_in at each reset -- never split/consumed, so noise draws in
        # _obs() cannot perturb the dynamics key that drives resets/autoresets.
        self._noise_key: jax.Array | None = None

        self._observe = make_observe(self.cfg)

        self._batched_step = jax.jit(jax.vmap(self.dynamics.step))
        self._batched_reset = jax.jit(jax.vmap(self.dynamics.reset))
        self._batched_observe = jax.jit(jax.vmap(self._observe))
        self._batched_reward = jax.jit(
            jax.vmap(
                lambda s, a, e: docking_reward(relative_view(s), a, e, self.cfg)
            )
        )
        self._batched_noise = (
            jax.jit(
                jax.vmap(
                    lambda s, k: apply_sensor_noise(
                        s,
                        k,
                        self.cfg.sensor_noise,
                        layout=NUM_LAYOUT,
                        range_m=jnp.linalg.norm(relative_view(s)[0:3]),
                    )
                )
            )
            if self.cfg.sensor_noise.enabled
            else None
        )
        self._batched_dock_goal_error = (
            jax.jit(
                jax.vmap(
                    lambda measured: dock_goal_error(
                        relative_view(measured), jnp.asarray(dock_target(self.cfg))
                    )
                )
            )
            if self.cfg.observation.goal_error
            else None
        )

    def reset(
        self,
        *,
        seed: int | Sequence[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is None or isinstance(seed, int):
            if seed is not None:
                self._key = jax.random.PRNGKey(seed)
            if self._key is None:
                # Never explicitly seeded: draw from system entropy so an
                # unseeded reset() is actually random, per Gymnasium's
                # contract, instead of always replaying the same trajectory.
                self._key = jax.random.PRNGKey(secrets.randbits(32))

            self._noise_key = jax.random.fold_in(self._key, NOISE_STREAM)

            self._key, subkey = jax.random.split(self._key)
            reset_keys = jax.random.split(subkey, self.num_envs)
        else:
            # Gymnasium's per-environment seeding: one seed per sub-env,
            # e.g. reset(seed=[seed0, seed1, ...]).
            seeds = list(seed)
            if len(seeds) != self.num_envs:
                raise ValueError(
                    f"seed list length ({len(seeds)}) must equal num_envs "
                    f"({self.num_envs})"
                )
            reset_keys = jnp.stack([jax.random.PRNGKey(s) for s in seeds])
            # Fold every seed into a base key so subsequent autoresets stay
            # deterministic and depend on the whole seed list, not just seeds[0].
            key = jax.random.PRNGKey(0)
            for s in seeds:
                key = jax.random.fold_in(key, s)
            self._key = key
            self._noise_key = jax.random.fold_in(self._key, NOISE_STREAM)

        self._states = self._batched_reset(reset_keys)
        self._step_index[:] = 0
        self._needs_reset[:] = False
        obs, measured_states = self._obs()
        return obs, {
            "success": np.zeros(self.num_envs, dtype=bool),
            "collision": np.zeros(self.num_envs, dtype=bool),
            "escaped": np.zeros(self.num_envs, dtype=bool),
            "state": self._true_states(),
            "measured_state": measured_states,
        }

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        if self._states is None:
            raise RuntimeError("step() called before reset(); call reset() first.")

        # NEXT_STEP autoreset: sub-envs flagged last step report their reset
        # observation now, with the given action for that lane discarded
        # entirely -- no physics step is applied to it. Matches gymnasium's
        # own SyncVectorEnv NEXT_STEP behaviour (it calls env.reset(), not
        # env.step(), for autoreset lanes).
        autoreset = self._needs_reset.copy()

        clipped = np.clip(
            np.asarray(actions, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        )
        actions_j = jnp.asarray(clipped)

        next_states, events = self._batched_step(self._states, actions_j)
        rewards = np.array(self._batched_reward(next_states, actions_j, events), dtype=np.float32)

        self._step_index += 1

        collision = np.array(events.collision, dtype=bool)
        docked = np.array(events.docked, dtype=bool)
        escaped = np.array(events.escaped, dtype=bool)
        terminations = collision | docked | escaped
        truncations = (~terminations) & (self._step_index >= self.cfg.max_steps)

        # Sub-envs that autoreset this step report their fresh reset
        # observation and a neutral transition -- the step computed above for
        # those lanes is discarded, not just its reward/flags.
        if autoreset.any():
            self._key, subkey = jax.random.split(self._key)
            fresh = self._batched_reset(jax.random.split(subkey, self.num_envs))
            mask = jnp.asarray(autoreset)[:, None]
            next_states = jnp.where(mask, fresh, next_states)
            self._step_index[autoreset] = 0
            rewards[autoreset] = 0.0
            terminations[autoreset] = False
            truncations[autoreset] = False
            collision[autoreset] = False
            docked[autoreset] = False
            escaped[autoreset] = False

        self._states = next_states
        self._needs_reset = terminations | truncations

        obs, measured_states = self._obs()
        return (
            obs,
            rewards,
            terminations,
            truncations,
            {
                "success": docked,
                "collision": collision,
                "escaped": escaped,
                "state": self._true_states(),
                "measured_state": measured_states,
            },
        )

    def _obs(self) -> tuple[np.ndarray, np.ndarray]:
        """(observation, measured_states) from `self._states`.

        `measured_states` is the RAW 21D `NUM_LAYOUT` states after sensor
        noise -- identical to `self._states` when noise is disabled --
        narrowed to float32 for `info["measured_state"]`. `observation` is
        `make_observe(cfg)` applied to that same measurement, with the
        goal-error block (from `relative_view` of the same measurement, never
        a second noise draw) appended when configured.
        """
        if self._batched_noise is None:
            measured = self._states
        else:
            self._noise_key, subkey = jax.random.split(self._noise_key)
            noise_keys = jax.random.split(subkey, self.num_envs)
            measured = self._batched_noise(self._states, noise_keys)
        obs = self._batched_observe(measured)
        if self._batched_dock_goal_error is not None:
            obs = jnp.concatenate(
                [obs, self._batched_dock_goal_error(measured).astype(obs.dtype)], axis=-1
            )
        return np.asarray(obs, dtype=np.float32), np.asarray(measured, dtype=np.float32)

    def _true_states(self) -> np.ndarray:
        return np.asarray(self._states, dtype=np.float32)
