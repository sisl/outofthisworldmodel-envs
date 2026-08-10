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

from .config import ISSConfig, dock_port_targets, dock_target
from .dynamics import ISSDynamics
from .goal import (
    GOAL_ERROR_DIM,
    GOAL_ERROR_NORM_LABELS,
    dock_goal_error,
    goal_error_norms,
)
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
    if cfg.observation.goal_error:
        low = np.concatenate([low, [-inf] * GOAL_ERROR_DIM]).astype(np.float32)
        high = np.concatenate([high, [inf] * GOAL_ERROR_DIM]).astype(np.float32)
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

        # Ports an episode may be assigned. Empty when the config names none,
        # and then nothing below is ever drawn or passed: `_dock_pose` stays
        # None, so the dynamics and the reward fall back to `cfg.dock` exactly
        # as they did before this field existed.
        self._port_names: tuple[str, ...] = tuple(p.name for p in self.cfg.dock.ports)
        self._port_targets = jnp.asarray(dock_port_targets(self.cfg), dtype=jnp.float32)
        self._port_index: int | None = None
        self._dock_pose: jnp.ndarray | None = None
        self._cfg_dock_target = jnp.asarray(dock_target(self.cfg), dtype=jnp.float32)

        # jit once at construction; all are pure functions of their arguments.
        # A drawn port reaches `step` as an argument, so a new draw each
        # episode costs no recompilation.
        self._jit_step = jax.jit(self.dynamics.step)
        self._jit_reset = jax.jit(self.dynamics.reset)
        self._jit_dock_goal_error = self._build_jit_dock_goal_error()
        # Built whatever `observation.goal_error` says: the block above is an
        # observation and this is telemetry, and a run that emits no goal
        # block still wants to be told whether its policy is closing in.
        self._jit_goal_error_norms = jax.jit(
            lambda state, target: goal_error_norms(dock_goal_error(state, target))
        )

    def _build_jit_dock_goal_error(self) -> Any | None:
        """measured -> the goal-error block, or None when it isn't emitted.

        With no ports a naked reset always flies to `cfg.dock`, and that pose
        is compiled in as a constant -- the arrangement a config without ports
        had before ports existed, and float32 constant folding is not
        bit-identical to the same arithmetic on a runtime argument. A port set
        makes it a per-episode argument read off `_dock_pose` at call time
        instead.
        """
        if not self.cfg.observation.goal_error:
            return None
        if not self._port_names:
            # A reset-options override can still retarget a no-ports episode,
            # so the constant-folded form serves only while `_dock_pose` is
            # None -- which a naked reset guarantees, keeping published
            # no-ports observations byte-identical.
            pose = self._cfg_dock_target
            folded = jax.jit(lambda measured: dock_goal_error(measured, pose))
            jitted = jax.jit(dock_goal_error)
            return lambda measured: (
                folded(measured)
                if self._dock_pose is None
                else jitted(measured, self._dock_pose)
            )
        jitted = jax.jit(dock_goal_error)
        return lambda measured: jitted(measured, self._dock_pose)

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
        # the same initial state whether or not ports are configured. An
        # override from a previous episode never survives into this one.
        self._port_index = None
        self._dock_pose = None
        if not self._apply_goal_options(options):
            self._draw_port()
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
            iss_reward(
                next_state, action_j, events, self.cfg,
                None if self._dock_pose is None else self._dock_pose[0:3],
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

    def _draw_port(self) -> None:
        """Assign this episode a port, uniformly over the configured set."""
        if not self._port_names:
            return
        self._port_index = int(self.np_random.integers(len(self._port_names)))
        self._dock_pose = self._port_targets[self._port_index]

    def _apply_goal_options(self, options: dict[str, Any] | None) -> bool:
        """Point the episode where `reset(options=...)` says; False if it doesn't.

        See `reset` for the contract. Only a draw among several names consumes
        randomness, so a single-name or explicit-pose reset leaves np_random's
        stream where a naked reset's port draw would have started.
        """
        if not options:
            return False
        # A goal-selection API must not let a typo fall through to a random
        # goal: an unrecognised key is an error, not a naked reset.
        unknown_keys = set(options) - {"dock_port", "dock_pose"}
        if unknown_keys:
            raise ValueError(
                f"unknown reset option(s) {sorted(unknown_keys)}; this environment "
                "understands 'dock_port' and 'dock_pose'"
            )
        port = options.get("dock_port")
        pose = options.get("dock_pose")
        if port is not None and pose is not None:
            raise ValueError("reset options carry dock_port or dock_pose, not both")
        if pose is not None:
            row = np.asarray(pose, dtype=np.float32).reshape(-1)
            if row.shape != (7,):
                raise ValueError(
                    "dock_pose must be 7 values [position xyz, quaternion wxyz], "
                    f"got shape {np.asarray(pose).shape}"
                )
            # In a no-ports config `_jit_step` has only ever seen dock_pose as
            # None; the first overridden episode hands it an array and pays a
            # one-off retrace. Both signatures stay cached after that.
            self._dock_pose = jnp.asarray(row)
            return True
        if port is None:
            return False
        if isinstance(port, str):
            names: tuple[str, ...] = (port,)
        elif isinstance(port, (list, tuple)) and all(isinstance(n, str) for n in port):
            names = tuple(port)
        else:
            raise ValueError(
                "dock_port must be a port name or a list/tuple of port names, "
                f"got {type(port).__name__}"
            )
        if not names:
            raise ValueError("dock_port names an empty set of ports")
        unknown = [n for n in names if n not in self._port_names]
        if unknown:
            raise ValueError(
                f"unknown dock_port(s) {unknown}; this environment's configured "
                f"ports are {list(self._port_names)}"
                + ("" if self._port_names else " -- configure dock.ports or pass dock_pose")
            )
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"duplicate dock_port(s) {duplicates}; name each port at most once"
            )
        name = (
            names[0]
            if len(names) == 1
            else names[int(self.np_random.integers(len(names)))]
        )
        self._port_index = self._port_names.index(name)
        self._dock_pose = self._port_targets[self._port_index]
        return True

    def _goal_pose(self) -> np.ndarray:
        """The (7,) [position, quaternion] pose this episode is flying to.

        The drawn port's row, or `cfg.dock`'s own pose when no ports are
        configured -- always present, unlike the port name and index, so a
        consumer reads one key to learn where the episode's goal is whichever
        kind of config produced it. Same layout and dtype as
        `config.dock_target` and the rows of `policies.dock_target_table`.
        """
        pose = self._cfg_dock_target if self._dock_pose is None else self._dock_pose
        return np.asarray(pose, dtype=np.float32)

    def _goal_error_true(self) -> dict[str, float]:
        """How far the TRUE state is from the episode's goal, per quantity.

        Measured against `self._state` and never the observation, deliberately:
        this is diagnostics for whoever is watching a run -- training logs the
        per-episode minimum of these to see whether a policy actually
        approaches its port -- and a sensor-noise draw would answer a
        different question, one about the navigation system rather than the
        controller. Nothing a policy sees; the observation's goal block (which
        does carry the noise, as it must) is separate.

        The goal is the episode's own: the drawn port, or `cfg.dock` when no
        ports are configured.
        """
        target = self._cfg_dock_target if self._dock_pose is None else self._dock_pose
        norms = np.asarray(self._jit_goal_error_norms(self._state, target))
        return {label: float(v) for label, v in zip(GOAL_ERROR_NORM_LABELS, norms)}

    def _port_info(self) -> dict[str, Any]:
        """The episode's port, for attributing a trajectory to its goal.

        Absent, not None or -1, when no ports are configured: a run on the
        single `cfg.dock` pose was never assigned a port, and a sentinel would
        claim otherwise.
        """
        if self._port_index is None:
            return {}
        return {
            "dock_port": self._port_names[self._port_index],
            "dock_port_index": self._port_index,
        }

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
