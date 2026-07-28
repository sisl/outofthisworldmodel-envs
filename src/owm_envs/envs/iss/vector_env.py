"""Gymnasium vector-environment adapter for the ISS docking task.

Subclasses gymnasium.vector.VectorEnv directly rather than wrapping N single
envs in SyncVectorEnv -- the physics is natively batched via jax.vmap, and
SyncVectorEnv would discard that.

Autoreset is NEXT_STEP: when a sub-env terminates or truncates, the step after
that returns its reset observation with reward 0 and both flags false.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from .config import ISSConfig
from .dynamics import ISSDynamics
from .env import _action_space, _observation_space
from .reward import iss_reward


class ISSVectorEnv(VectorEnv):
    metadata = {
        "render_modes": [],
        "render_fps": 24,
        "autoreset_mode": AutoresetMode.NEXT_STEP,
    }

    def __init__(self, num_envs: int = 1, cfg: ISSConfig | None = None):
        self.cfg = cfg or ISSConfig()
        self.num_envs = int(num_envs)
        self.dynamics = ISSDynamics(self.cfg)

        self.single_observation_space = _observation_space(self.cfg)
        self.single_action_space = _action_space(self.cfg)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        self._states: jnp.ndarray | None = None
        self._step_index = np.zeros(self.num_envs, dtype=np.int64)
        self._needs_reset = np.zeros(self.num_envs, dtype=bool)
        self._key: jax.Array | None = None

        self._batched_step = jax.jit(jax.vmap(self.dynamics.step))
        self._batched_reset = jax.jit(jax.vmap(self.dynamics.reset))
        self._batched_reward = jax.jit(
            jax.vmap(lambda s, a, e: iss_reward(s, a, e, self.cfg))
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._key = jax.random.PRNGKey(seed)
        if self._key is None:
            self._key = jax.random.PRNGKey(0)

        self._key, subkey = jax.random.split(self._key)
        self._states = self._batched_reset(jax.random.split(subkey, self.num_envs))
        self._step_index[:] = 0
        self._needs_reset[:] = False
        return self._obs(), self._empty_info()

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
        terminations = collision | docked
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

        self._states = next_states
        self._needs_reset = terminations | truncations

        return (
            self._obs(),
            rewards,
            terminations,
            truncations,
            {"success": docked, "collision": collision},
        )

    def _obs(self) -> np.ndarray:
        return np.asarray(self._states, dtype=np.float32)

    def _empty_info(self) -> dict[str, np.ndarray]:
        return {
            "success": np.zeros(self.num_envs, dtype=bool),
            "collision": np.zeros(self.num_envs, dtype=bool),
        }
