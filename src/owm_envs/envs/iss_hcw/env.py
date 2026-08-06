"""Gymnasium single-environment adapter for the iss-hcw docking task.

Structurally mirrors `envs/iss/env.py`; the differences all trace back to
`HCWDynamics` carrying a 15D state (an [jd, sec] epoch prefix ahead of the
canonical 13D view) at float64, where `ISSDynamics` carries the 13D view
alone at float32. `HCWDynamics.step`'s internal accumulation depends on that
epoch prefix never being rounded to f32 -- `envs/iss_hcw/dynamics.py`
documents a measured 290 s/orbit drift if it is -- so `self._state` is kept
at whatever dtype `HCWDynamics` hands back (float64) for its entire life;
only `_obs()` and `_true_state()` narrow to float32, and only on freshly
computed copies, never by writing back into `self._state`.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from ..common.config import dock_target
from ..common.goal import GOAL_ERROR_DIM, dock_goal_error
from ..common.reward import docking_reward
from ..common.sensing import NOISE_STREAM, apply_sensor_noise
from .config import HCW_LAYOUT, HCWConfig
from .dynamics import HCWDynamics


def _observation_space(cfg: HCWConfig) -> spaces.Box:
    inf = np.inf
    low = np.array(
        [0.0, 0.0] + [-inf] * 3 + [-inf] * 3 + [-1.0] * 4 + [-inf] * 3, dtype=np.float32
    )
    high = np.array(
        [inf, 86400.0] + [inf] * 3 + [inf] * 3 + [1.0] * 4 + [inf] * 3, dtype=np.float32
    )
    if cfg.observation.goal_error:
        low = np.concatenate([low, [-inf] * GOAL_ERROR_DIM]).astype(np.float32)
        high = np.concatenate([high, [inf] * GOAL_ERROR_DIM]).astype(np.float32)
    return spaces.Box(low=low, high=high, dtype=np.float32)


def _render_fps(cfg: HCWConfig) -> int:
    """Playback rate for one frame per simulation step, as a positive integer.

    That rate is 1/dt, but Gymnasium's render_fps has to be a usable frame
    rate: consumers divide by it or hand it to a video encoder. Any dt of 2 s
    or more rounds to zero -- exactly 2.0 included, since Python rounds a tie
    to even -- so the result is floored at 1.
    """
    return max(1, round(1.0 / cfg.dt))


def _action_space(cfg: HCWConfig) -> spaces.Box:
    high = np.array(
        [cfg.control.limit_force_n] * 3 + [cfg.control.limit_torque_nm] * 3,
        dtype=np.float32,
    )
    return spaces.Box(low=-high, high=high, dtype=np.float32)


class HCWEnv(gym.Env):
    # render_fps is overridden per instance in __init__; the class-level value
    # is the rate implied by HCWConfig's own default dt.
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, cfg: HCWConfig | None = None, render_mode: str | None = None):
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"unknown render_mode {render_mode!r}; expected one of "
                f"{self.metadata['render_modes']} or None"
            )

        self.cfg = cfg or HCWConfig()
        # Per-instance because it depends on cfg.dt, which the class does not know.
        self.metadata = {**self.metadata, "render_fps": _render_fps(self.cfg)}
        self.dynamics = HCWDynamics(self.cfg)
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
        self._jit_dock_goal_error = (
            jax.jit(
                lambda measured: dock_goal_error(
                    HCW_LAYOUT.slice_view(measured), jnp.asarray(dock_target(self.cfg))
                )
            )
            if self.cfg.observation.goal_error
            else None
        )

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
            "escaped": False,
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
        reward = float(
            docking_reward(HCW_LAYOUT.slice_view(next_state), action_j, events, self.cfg)
        )

        self._state = next_state
        self._step_index += 1

        collision = bool(events.collision)
        docked = bool(events.docked)
        escaped = bool(events.escaped)
        terminated = collision or docked or escaped
        truncated = (not terminated) and self._step_index >= self.cfg.max_steps

        return (
            self._obs(),
            reward,
            terminated,
            truncated,
            {
                "success": docked,
                "collision": collision,
                "escaped": escaped,
                "state": self._true_state(),
            },
        )

    def _obs(self) -> np.ndarray:
        if not self.cfg.sensor_noise.enabled:
            measured = self._state
        else:
            # Noise draws come from `_noise_key`, a fold_in side stream set up
            # in reset() -- split here, never touching np_random or the
            # dynamics key. `apply_sensor_noise` leaves the epoch prefix
            # (HCW_LAYOUT.epoch) untouched: a vehicle knows its own clock.
            self._noise_key, subkey = jax.random.split(self._noise_key)
            measured = apply_sensor_noise(
                self._state, subkey, self.cfg.sensor_noise, layout=HCW_LAYOUT
            )
        if self._jit_dock_goal_error is not None:
            # Computed from `measured`, not `self._state`: the goal block
            # must reflect the same (possibly noisy) observation the caller
            # receives, never a second noise draw or privileged truth.
            measured = jnp.concatenate([measured, self._jit_dock_goal_error(measured)])
        # The float64 state is cast down on this copy only -- see the module
        # docstring on why `self._state` itself never narrows.
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
        # The renderer's lighting is posed from its own static config, not
        # from the epoch and chief geometry carried in this env's state --
        # that seam (RenderInputs) lands in PR 5.
        return self._renderer.render(
            np.asarray(HCW_LAYOUT.slice_view(self._state), dtype=np.float32),
            view=self.cfg.render_view,
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
