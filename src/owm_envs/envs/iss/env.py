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
from .sensing import NOISE_STREAM, apply_sensor_noise


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
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, cfg: ISSConfig | None = None, render_mode: str | None = None):
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"unknown render_mode {render_mode!r}; expected one of "
                f"{self.metadata['render_modes']} or None"
            )

        self.cfg = cfg or ISSConfig()
        # Per-instance because it depends on cfg.dt, which the class does not know.
        self.metadata = {**self.metadata, "render_fps": _render_fps(self.cfg)}
        self.dynamics = ISSDynamics(self.cfg)
        self.observation_space = _observation_space(self.cfg)
        self.action_space = _action_space(self.cfg)
        self.render_mode = render_mode

        self._state: jnp.ndarray | None = None
        self._step_index = 0
        self._renderer: Any | None = None
        # Side stream for sensor-noise draws, derived by fold_in from the
        # dynamics key at reset -- never consumed from np_random, which also
        # seeds reset(), so later unseeded resets don't depend on how many
        # noisy observations the previous episode drew.
        self._noise_key: jax.Array | None = None

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
        dynamics_key = jax.random.PRNGKey(jax_seed)
        if self.cfg.sensor_noise.enabled:
            self._noise_key = jax.random.fold_in(dynamics_key, NOISE_STREAM)
        self._state = self._jit_reset(dynamics_key)
        self._step_index = 0
        return self._obs(), {
            "success": False,
            "collision": False,
            "state": self._true_state(),
        }

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
            {"success": docked, "collision": collision, "state": self._true_state()},
        )

    def _obs(self) -> np.ndarray:
        state = np.asarray(self._state, dtype=np.float32)
        if not self.cfg.sensor_noise.enabled:
            return state
        # Noise draws come from `_noise_key`, a fold_in side stream set up in
        # reset() -- split here, never touching np_random or the dynamics key.
        self._noise_key, subkey = jax.random.split(self._noise_key)
        measured = apply_sensor_noise(jnp.asarray(state), subkey, self.cfg.sensor_noise)
        return np.asarray(measured, dtype=np.float32)

    def _true_state(self) -> np.ndarray:
        return np.asarray(self._state, dtype=np.float32)

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self._state is None:
            raise RuntimeError("render() called before reset(); call reset() first.")

        if self._renderer is None:
            self._renderer = self._make_renderer()
        return self._renderer.render(
            np.asarray(self._state, dtype=np.float32), view=self.cfg.render_view
        )

    def _make_renderer(self) -> Any:
        try:
            from ...render.iss_scene import RenderConfig
            from ...render.renderer import ISSRenderer
        except ImportError as exc:
            raise ImportError(
                "Rendering requires the optional 'render' extra: pip install owm-envs[render]"
            ) from exc

        render_cfg = RenderConfig(**self.cfg.render) if self.cfg.render else RenderConfig()
        return ISSRenderer(render_cfg)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
