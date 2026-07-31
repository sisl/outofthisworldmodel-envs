"""Gymnasium single-environment adapter for the ISS docking task.

Thin wrapper over ISSDynamics at a numpy boundary. Stable-Baselines3 consumes
single envs (it builds its own DummyVecEnv/SubprocVecEnv), so this is the entry
point for SB3 interop -- gymnasium.vector.VectorEnv is not what SB3 accepts.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from .config import ISSConfig
from .dynamics import ISSDynamics
from .reward import iss_reward


def _observation_space(cfg: ISSConfig) -> spaces.Box:
    inf = np.inf
    low = np.array(
        [-inf] * 3 + [-inf] * 3 + [-1.0] * 4 + [-inf] * 3, dtype=np.float32
    )
    high = np.array(
        [inf] * 3 + [inf] * 3 + [1.0] * 4 + [inf] * 3, dtype=np.float32
    )
    return spaces.Box(low=low, high=high, dtype=np.float32)


def _render_fps(cfg: ISSConfig) -> int:
    """Playback rate for one frame per simulation step, as a positive integer.

    That rate is 1/dt, but Gymnasium's render_fps has to be a usable frame
    rate: consumers divide by it or hand it to a video encoder. Any dt of 2 s
    or more rounds to zero -- exactly 2.0 included, since Python rounds a tie
    to even -- so the result is floored at 1.
    """
    return max(1, round(1.0 / cfg.dt))


def _action_space(cfg: ISSConfig) -> spaces.Box:
    high = np.array(
        [cfg.control.limit_force_n] * 3 + [cfg.control.limit_torque_nm] * 3,
        dtype=np.float32,
    )
    return spaces.Box(low=-high, high=high, dtype=np.float32)


class ISSEnv(gym.Env):
    # render_fps is overridden per instance in __init__; the class-level value
    # is the rate implied by ISSConfig's own default dt.
    metadata = {"render_modes": [], "render_fps": 20}

    def __init__(self, cfg: ISSConfig | None = None, render_mode: str | None = None):
        self.cfg = cfg or ISSConfig()
        # Per-instance because it depends on cfg.dt, which the class does not know.
        self.metadata = {**self.metadata, "render_fps": _render_fps(self.cfg)}
        self.dynamics = ISSDynamics(self.cfg)
        self.observation_space = _observation_space(self.cfg)
        self.action_space = _action_space(self.cfg)
        self.render_mode = render_mode

        self._state: jnp.ndarray | None = None
        self._step_index = 0

        # jit once at construction; both are pure functions of (state, action).
        self._jit_step = jax.jit(self.dynamics.step)
        self._jit_reset = jax.jit(self.dynamics.reset)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        # Gymnasium seeds self.np_random; derive a JAX key from it so that a given
        # Gymnasium seed reproduces exactly one initial state.
        jax_seed = int(self.np_random.integers(0, 2**31 - 1))
        self._state = self._jit_reset(jax.random.PRNGKey(jax_seed))
        self._step_index = 0
        return self._obs(), {"success": False, "collision": False}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("step() called before reset(); call reset() first.")

        clipped = np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        )
        action_j = jnp.asarray(clipped)

        next_state, events = self._jit_step(self._state, action_j)
        reward = float(iss_reward(next_state, action_j, events, self.cfg))

        self._state = next_state
        self._step_index += 1

        collision = bool(events.collision)
        docked = bool(events.docked)
        terminated = collision or docked
        truncated = (not terminated) and self._step_index >= self.cfg.max_steps

        return (
            self._obs(),
            reward,
            terminated,
            truncated,
            {"success": docked, "collision": collision},
        )

    def _obs(self) -> np.ndarray:
        return np.asarray(self._state, dtype=np.float32)
