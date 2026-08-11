"""Gymnasium single-environment adapter for the iss-numerical docking task.

Structurally mirrors `envs/iss_hcw/env.py`; the differences all trace back to
two things `NumericalDynamics` adds: a 21D absolute-ECI state whose raw slices
are not the canonical relative view (`relative_view` derives it, `NUM_LAYOUT
.slice_view` must never be used as a substitute -- see `config.py`), and an
`observation.mode` that lets the recorded observation report the state in one
of four frames (`observe.py`), which is what makes this env's observation
width depend on `cfg` rather than being a fixed constant.

`self._state` is kept at whatever dtype `NumericalDynamics` hands back
(float64) for its entire life, for the same reason `iss_hcw`'s is: the epoch
prefix and the ~6.8e6 m ECI positions both need it, and `dynamics.py`'s module
docstring quantifies the cost of narrowing either early. `_obs()` and
`_true_state()` narrow to float32 only on freshly computed copies.

Sensor noise is applied to the RAW 21D state, through `NUM_LAYOUT`, before
`make_observe(cfg)` reshapes it -- narrow, mode, then goal, in that order, so
the reshape and the goal-error block both see the same noisy measurement a
navigation system would actually report, and neither one applies its own
independent noise draw. `NUM_LAYOUT.pos`/`vel`/`quat`/`omega` name the
chaser's ABSOLUTE ECI slices, so `apply_sensor_noise` needs a relative range
for `sigma_pos_frac_of_range` (its default reads the layout's own position
slice, ~6.8e6 m here) -- taken from `relative_view` of the TRUE state, not the
noisy one, matching what a real sensor's range estimate would be built from
before the noise it degrades is added.

Both `info["state"]` (float32, true) and `info["measured_state"]` (float32,
noisy) carry the full 21D `NUM_LAYOUT` state, regardless of `observation.mode`
-- so a consumer that needs the raw layout (the vector-driver policy source,
say) never has to invert whatever frame the recorded observation reports.
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
from .config import NUM_LAYOUT, OBS_MODE_DIM, NumericalConfig, ObservationMode
from .dynamics import NumericalDynamics, relative_view
from .observe import make_observe
from .render_adapter import make_render_adapter

# Where the (w-first) attitude quaternion sits in each mode's emitted
# observation, ahead of the epoch-count-dependent goal-error block appended
# separately. The three absolute-frame modes keep the raw state's own layout
# (`NUM_LAYOUT.quat` at 14:18); "relative" reports `[epoch, relative_view]`,
# whose quaternion sits at the view's own offset, 8 past the 2-wide epoch.
_QUAT_SLICE: dict[ObservationMode, slice] = {
    "absolute": slice(14, 18),
    "chaser_absolute": slice(14, 18),
    "chief_absolute": slice(14, 18),
    "relative": slice(8, 12),
}


def _observation_space(cfg: NumericalConfig) -> spaces.Box:
    width = OBS_MODE_DIM[cfg.observation.mode]
    low = np.full(width, -np.inf, dtype=np.float32)
    high = np.full(width, np.inf, dtype=np.float32)
    low[0], high[0] = 0.0, np.inf
    low[1], high[1] = 0.0, 86400.0
    quat = _QUAT_SLICE[cfg.observation.mode]
    low[quat], high[quat] = -1.0, 1.0
    if cfg.observation.goal_error:
        low = np.concatenate([low, [-np.inf] * GOAL_ERROR_DIM]).astype(np.float32)
        high = np.concatenate([high, [np.inf] * GOAL_ERROR_DIM]).astype(np.float32)
    return spaces.Box(low=low, high=high, dtype=np.float32)


class NumericalEnv(PortGoalMixin, gym.Env):
    # render_fps is overridden per instance in __init__; the class-level value
    # is the rate implied by NumericalConfig's own default dt.
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, cfg: NumericalConfig | None = None, render_mode: str | None = None):
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"unknown render_mode {render_mode!r}; expected one of "
                f"{self.metadata['render_modes']} or None"
            )

        self.cfg = cfg or NumericalConfig()
        # Per-instance because it depends on cfg.dt, which the class does not know.
        self.metadata = {**self.metadata, "render_fps": render_fps(self.cfg)}
        self.dynamics = NumericalDynamics(self.cfg)
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

        self._observe = make_observe(self.cfg)

        # jit once at construction; all are pure functions of their inputs.
        self._jit_step = jax.jit(self.dynamics.step)
        self._jit_reset = jax.jit(self.dynamics.reset)
        self._jit_observe = jax.jit(self._observe)
        self._jit_apply_noise = (
            jax.jit(
                lambda state, key: apply_sensor_noise(
                    state,
                    key,
                    self.cfg.sensor_noise,
                    layout=NUM_LAYOUT,
                    range_m=jnp.linalg.norm(relative_view(state)[0:3]),
                )
            )
            if self.cfg.sensor_noise.enabled
            else None
        )
        # The task layer reads the chief-relative view; the port machinery's
        # goal error does too. A drawn port reaches `step` as an argument, so
        # a new draw each episode costs no recompilation.
        self._init_port_goals(view=relative_view)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start an episode; `options` may target a port or pose.

        See `PortGoalMixin` for the options contract -- the same one `ISSEnv`
        honours: "dock_port" names one or several of this env's configured
        ports, "dock_pose" is an explicit (7,) [position, quaternion] goal in
        the world frame (chief-relative, like every dock pose here).
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
        obs, measured_state = self._observation_and_measured()
        return obs, {
            "success": False,
            "collision": False,
            "escaped": False,
            "state": self._true_state(),
            "measured_state": measured_state,
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
                relative_view(next_state), action_j, events, self.cfg, self._dock_pose,
            )
        )

        self._state = next_state
        self._step_index += 1

        collision = bool(events.collision)
        docked = bool(events.docked)
        escaped = bool(events.escaped)
        terminated = collision or docked or escaped
        truncated = (not terminated) and self._step_index >= self.cfg.max_steps

        obs, measured_state = self._observation_and_measured()
        return (
            obs,
            reward,
            terminated,
            truncated,
            {
                "success": docked,
                "collision": collision,
                "escaped": escaped,
                "state": self._true_state(),
                "measured_state": measured_state,
                "goal_pose": self._goal_pose(),
                "goal_error_true": self._goal_error_true(),
                **self._port_info(),
            },
        )

    def _observation_and_measured(self) -> tuple[np.ndarray, np.ndarray]:
        """(observation, measured_state) from `self._state`.

        `measured_state` is the RAW 21D `NUM_LAYOUT` state after sensor noise
        -- identical to `self._state` when noise is disabled -- narrowed to
        float32 for `info["measured_state"]`. `observation` is
        `make_observe(cfg)` applied to that same measurement, with the
        goal-error block (computed from `relative_view` of the same
        measurement, never a second noise draw) appended when configured.
        """
        if self._jit_apply_noise is None:
            measured = self._state
        else:
            self._noise_key, subkey = jax.random.split(self._noise_key)
            measured = self._jit_apply_noise(self._state, subkey)
        obs = self._jit_observe(measured)
        if self._jit_dock_goal_error is not None:
            block = self._jit_dock_goal_error(measured)
            obs = jnp.concatenate([obs, block.astype(obs.dtype)])
        return np.asarray(obs, dtype=np.float32), np.asarray(measured, dtype=np.float32)

    # Both paths below hand out float32 copies of the state -- the row a
    # dataset records, and the row `render()` poses -- and both are narrow by
    # design. `dynamics.py`'s module docstring quantifies what that costs when
    # it feeds `relative_view`'s subtraction of two ~6.8e6 m ECI positions.
    # What must never narrow is `self._state` itself, which the dynamics
    # accumulate from.
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
