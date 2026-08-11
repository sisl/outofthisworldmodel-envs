"""Gymnasium vector-environment adapter for the ISS docking task.

Subclasses gymnasium.vector.VectorEnv directly rather than wrapping N single
envs in SyncVectorEnv -- the physics is natively batched via jax.vmap, and
SyncVectorEnv would discard that.

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
from ..common.config import dock_port_targets, dock_target
from ..common.goal import dock_goal_error
from ..common.port_goals import parse_goal_options
from ..common.reward import docking_reward
from ..common.sensing import NOISE_STREAM, apply_sensor_noise
from .config import ISSConfig
from .dynamics import ISSDynamics
from .env import _observation_space


class ISSVectorEnv(VectorEnv):
    # render_fps is overridden per instance in __init__; the class-level value
    # is the rate implied by ISSConfig's own default dt.
    metadata = {
        "render_modes": [],
        "render_fps": 20,
        "autoreset_mode": AutoresetMode.NEXT_STEP,
    }

    def __init__(self, num_envs: int = 1, cfg: ISSConfig | None = None):
        self.cfg = cfg or ISSConfig()
        # Per-instance because it depends on cfg.dt, which the class does not know.
        self.metadata = {**self.metadata, "render_fps": render_fps(self.cfg)}
        self.num_envs = int(num_envs)
        self.dynamics = ISSDynamics(self.cfg)

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

        self._batched_step = jax.jit(jax.vmap(self.dynamics.step))
        self._batched_reset = jax.jit(jax.vmap(self.dynamics.reset))
        self._batched_reward = jax.jit(
            jax.vmap(lambda s, a, e: docking_reward(s, a, e, self.cfg))
        )
        self._batched_noise = (
            jax.jit(
                jax.vmap(lambda s, k: apply_sensor_noise(s, k, self.cfg.sensor_noise))
            )
            if self.cfg.sensor_noise.enabled
            else None
        )
        self._batched_dock_goal_error = (
            jax.jit(jax.vmap(lambda measured: dock_goal_error(measured, jnp.asarray(dock_target(self.cfg)))))
            if self.cfg.observation.goal_error
            else None
        )

        # Per-lane port goals. `_lane_targets` is (num_envs, 7), or None when
        # every lane flies cfg.dock's single pose -- the arrangement that
        # keeps the constant-folded forms above byte-identical for a no-ports
        # run. `_lane_ports` are the drawn indices into the configured set
        # (None under a dock_pose override), and `_menu` the indices lanes
        # redraw from at reset and autoreset. The *_to forms take per-lane
        # targets as runtime arguments and are traced only when lanes
        # actually carry them.
        self._port_names: tuple[str, ...] = tuple(p.name for p in self.cfg.dock.ports)
        self._port_targets = jnp.asarray(dock_port_targets(self.cfg), dtype=jnp.float32)
        self._cfg_dock_target = jnp.asarray(dock_target(self.cfg), dtype=jnp.float32)
        self._menu: tuple[int, ...] | None = None
        self._lane_ports: np.ndarray | None = None
        self._lane_targets: jnp.ndarray | None = None
        self._batched_reward_to = jax.jit(
            jax.vmap(lambda s, a, e, p: docking_reward(s, a, e, self.cfg, p))
        )
        self._batched_dock_goal_error_to = (
            jax.jit(
                jax.vmap(lambda measured, target: dock_goal_error(measured, target))
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
        """Start every lane; `options` may aim them like the single env's.

        The same grammar as `PortGoalMixin` (see `parse_goal_options`):
        "dock_port" narrows the menu lanes draw from -- one name pins every
        lane to that port, several have each lane draw uniformly among them,
        at reset and again at every autoreset -- and "dock_pose" is one
        explicit (7,) goal shared by every lane. A naked reset draws each
        lane's port from the full configured set, or flies cfg.dock's single
        pose when no ports are configured.
        """
        # Parsed before any seed state is consumed, so a bad option is a
        # clean error rather than a half-advanced RNG.
        parsed = parse_goal_options(options, self._port_names)
        self._menu = None
        self._lane_ports = None
        self._lane_targets = None
        if parsed is None:
            if self._port_names:
                self._menu = tuple(range(len(self._port_names)))
        elif parsed[0] == "pose":
            self._lane_targets = jnp.tile(parsed[1][None, :], (self.num_envs, 1))
        else:
            self._menu = parsed[1]
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
        if self._menu is not None:
            # Drawn from a split taken after the state keys, so a run with no
            # ports consumes exactly the key material it always did and stays
            # byte-identical.
            self._key, port_key = jax.random.split(self._key)
            self._draw_lane_ports(np.ones(self.num_envs, dtype=bool), port_key)
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

        if self._lane_targets is None:
            next_states, events = self._batched_step(self._states, actions_j)
            rewards = np.array(
                self._batched_reward(next_states, actions_j, events), dtype=np.float32
            )
        else:
            # Same vmapped step; handing it a third batched argument is a
            # separate trace, cached alongside the two-argument one.
            next_states, events = self._batched_step(
                self._states, actions_j, self._lane_targets
            )
            rewards = np.array(
                self._batched_reward_to(
                    next_states, actions_j, events, self._lane_targets[:, 0:3]
                ),
                dtype=np.float32,
            )

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
            if self._menu is not None:
                self._key, port_key = jax.random.split(self._key)
                self._draw_lane_ports(autoreset, port_key)
            self._step_index[autoreset] = 0
            rewards[autoreset] = 0.0
            terminations[autoreset] = False
            truncations[autoreset] = False
            collision[autoreset] = False
            docked[autoreset] = False
            escaped[autoreset] = False

        self._states = next_states
        self._needs_reset = terminations | truncations

        return (
            self._obs(),
            rewards,
            terminations,
            truncations,
            {
                "success": docked,
                "collision": collision,
                "escaped": escaped,
                "state": self._true_states(),
                **self._goal_info(),
            },
        )

    def _obs(self) -> np.ndarray:
        if self._batched_noise is None:
            measured = self._states
        else:
            self._noise_key, subkey = jax.random.split(self._noise_key)
            noise_keys = jax.random.split(subkey, self.num_envs)
            measured = self._batched_noise(self._states, noise_keys)
        if self._batched_dock_goal_error is not None:
            # Computed from `measured`, not `self._states`: the goal block
            # must reflect the same (possibly noisy) observation the caller
            # receives, never a second noise draw or privileged truth. Each
            # lane measures against its own drawn target when lanes carry
            # them.
            block = (
                self._batched_dock_goal_error(measured)
                if self._lane_targets is None
                else self._batched_dock_goal_error_to(measured, self._lane_targets)
            )
            measured = jnp.concatenate([measured, block], axis=-1)
        return np.asarray(measured, dtype=np.float32)

    def _true_states(self) -> np.ndarray:
        return np.asarray(self._states, dtype=np.float32)

    def _draw_lane_ports(self, lanes: np.ndarray, key: jax.Array) -> None:
        """Redraw the flagged lanes' ports uniformly from the active menu."""
        menu = np.asarray(self._menu)
        drawn = menu[np.asarray(jax.random.randint(key, (self.num_envs,), 0, len(menu)))]
        self._lane_ports = (
            drawn if self._lane_ports is None else np.where(lanes, drawn, self._lane_ports)
        )
        self._lane_targets = self._port_targets[jnp.asarray(self._lane_ports)]

    def _goal_info(self) -> dict[str, np.ndarray]:
        """Per-lane goal keys: always the pose; the port only when drawn."""
        if self._lane_targets is None:
            poses = np.tile(
                np.asarray(self._cfg_dock_target, dtype=np.float32), (self.num_envs, 1)
            )
        else:
            poses = np.asarray(self._lane_targets, dtype=np.float32)
        info: dict[str, np.ndarray] = {"goal_pose": poses}
        if self._lane_ports is not None:
            info["dock_port"] = np.array([self._port_names[i] for i in self._lane_ports])
            info["dock_port_index"] = np.asarray(self._lane_ports, dtype=np.int64)
        return info

    def _empty_info(self) -> dict[str, np.ndarray]:
        return {
            "success": np.zeros(self.num_envs, dtype=bool),
            "collision": np.zeros(self.num_envs, dtype=bool),
            "escaped": np.zeros(self.num_envs, dtype=bool),
            "state": self._true_states(),
            **self._goal_info(),
        }
