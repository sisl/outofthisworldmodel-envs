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

from ..common.adapter import action_space, render_fps
from ..common.goal import GOAL_ERROR_DIM
from ..common.port_goals import PortGoalMixin
from ..common.reward import docking_reward
from ..common.sensing import NOISE_STREAM, apply_sensor_noise
from .config import ISSConfig
from .dynamics import ISSDynamics
from .render_adapter import make_render_adapter


def _observation_space(cfg: ISSConfig) -> spaces.Box:
    inf = np.inf
    low = np.array(
        [-inf] * 3 + [-inf] * 3 + [-1.0] * 4 + [-inf] * 3, dtype=np.float32
    )
    high = np.array(
        [inf] * 3 + [inf] * 3 + [1.0] * 4 + [inf] * 3, dtype=np.float32
    )
    if cfg.observation.goal_error:
        low = np.concatenate([low, [-inf] * GOAL_ERROR_DIM]).astype(np.float32)
        high = np.concatenate([high, [inf] * GOAL_ERROR_DIM]).astype(np.float32)
    return spaces.Box(low=low, high=high, dtype=np.float32)


class ISSEnv(PortGoalMixin, gym.Env):
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
        self.metadata = {**self.metadata, "render_fps": render_fps(self.cfg)}
        self.dynamics = ISSDynamics(self.cfg)
        self.observation_space = _observation_space(self.cfg)
        self.action_space = action_space(self.cfg)
        self.render_mode = render_mode

        self._state: jnp.ndarray | None = None
        self._step_index = 0
        self._renderer: Any | None = None
        # The same adapter the dataset render path uses, so a frame from
        # `render()` is posed by the code that poses a dataset's video.
        # Built here rather than at first render: it costs nothing and needs
        # none of the render extra.
        self._render_adapter = make_render_adapter(self.cfg)
        # Side stream for sensor-noise draws, derived by fold_in from the
        # dynamics key at reset -- never consumed from np_random, which also
        # seeds reset(), so later unseeded resets don't depend on how many
        # noisy observations the previous episode drew.
        self._noise_key: jax.Array | None = None

        # jit once at construction; all are pure functions of their arguments.
        # A drawn port reaches `step` as an argument, so a new draw each
        # episode costs no recompilation.
        self._jit_step = jax.jit(self.dynamics.step)
        self._jit_reset = jax.jit(self.dynamics.reset)
        # This env's state IS the 13-wide task view, so the port machinery's
        # view hook is the identity.
        self._init_port_goals(view=lambda state: state)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start an episode, optionally told where to fly via `options`.

        `options` may carry exactly one of:
          - "dock_port": a configured port name, or a sequence of them. One
            name targets that port; several draw uniformly among them. Names
            must come from this environment's own `cfg.dock.ports`.
          - "dock_pose": an explicit goal, 7 values [position xyz, quaternion
            wxyz] in the world frame, taken as given -- no port table is
            consulted. `info` then reports the pose under "goal_pose" but no
            "dock_port": the episode was aimed at a pose, not a named port.

        A naked reset keeps the existing behaviour: draw uniformly over
        `cfg.dock.ports`, or fly to `cfg.dock`'s single pose when none are
        configured.
        """
        super().reset(seed=seed)
        # Gymnasium seeds self.np_random; derive a JAX key from it so that a given
        # Gymnasium seed reproduces exactly one initial state.
        jax_seed = int(self.np_random.integers(0, 2**31 - 1))
        dynamics_key = jax.random.PRNGKey(jax_seed)
        if self.cfg.sensor_noise.enabled:
            self._noise_key = jax.random.fold_in(dynamics_key, NOISE_STREAM)
        self._state = self._jit_reset(dynamics_key)
        # Drawn after the dynamics seed, never before: the port is an extra
        # draw on the end of np_random's stream, so a given seed reproduces
        # the same initial state whether or not ports are configured.
        self._begin_episode_goal(options)
        self._step_index = 0
        return self._obs(), {
            "success": False,
            "collision": False,
            "escaped": False,
            "state": self._true_state(),
            "goal_pose": self._goal_pose(),
            "goal_error_true": self._goal_error_true(),
            **self._port_info(),
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

        next_state, events = self._jit_step(self._state, action_j, self._dock_pose)
        reward = float(
            docking_reward(
                next_state, action_j, events, self.cfg, self._dock_pose,
            )
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
                "goal_pose": self._goal_pose(),
                "goal_error_true": self._goal_error_true(),
                **self._port_info(),
            },
        )

    def _obs(self) -> np.ndarray:
        if not self.cfg.sensor_noise.enabled:
            measured = jnp.asarray(self._state, dtype=jnp.float32)
        else:
            # Noise draws come from `_noise_key`, a fold_in side stream set up
            # in reset() -- split here, never touching np_random or the
            # dynamics key.
            self._noise_key, subkey = jax.random.split(self._noise_key)
            measured = apply_sensor_noise(
                jnp.asarray(self._state), subkey, self.cfg.sensor_noise
            )
        if self._jit_dock_goal_error is not None:
            # Computed from `measured`, not `self._state`: the goal block
            # must reflect the same (possibly noisy) observation the caller
            # receives, never a second noise draw or privileged truth. It
            # measures against the episode's own port -- the same row the
            # dynamics score `docked` against and the reward is shaped toward.
            measured = jnp.concatenate([measured, self._jit_dock_goal_error(measured)])
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
            self._render_adapter(np.asarray(self._state, dtype=np.float32)),
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
