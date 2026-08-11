"""Per-episode dock-port goals, shared by the Gymnasium single-env adapters.

`DockConfig.ports` and `reset(options=...)` grew on `ISSEnv`; this module is
that machinery lifted out so `dock.ports` and the reset-options contract mean
exactly the same thing on every environment that poses the docking task. An
env differs only in where the 13-wide task view lives inside its state
vector, so that is the one thing a host supplies.

The host contract:
  - call `_init_port_goals(view)` at construction, after `self.cfg` exists --
    `view` maps the host's state vector to the `[pos, vel, quat, omega]` view
    the goal error reads;
  - call `_begin_episode_goal(options)` in `reset`, after `super().reset`
    has seeded `np_random` and after the dynamics key has been drawn from it
    (the port draw must stay an extra draw on the END of the stream, so a
    given seed reproduces the same initial state with or without ports);
  - thread `self._dock_pose` into the dynamics step and the reward, and put
    `_goal_pose()`, `_goal_error_true()` and `_port_info()` in `info`;
  - emit the observation's goal block through `self._jit_dock_goal_error`
    when it is not None.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from .config import BaseTaskConfig, dock_port_targets, dock_target
from .goal import GOAL_ERROR_NORM_LABELS, dock_goal_error, goal_error_norms

# state vector -> the 13-wide [pos, vel, quat, omega] view the task layer reads.
TaskView = Callable[[jnp.ndarray], jnp.ndarray]


class PortGoalMixin:
    """Per-episode dock-port drawing and `reset(options=...)` targeting.

    reset options carry exactly one of:
      - "dock_port": a configured port name, or a sequence of them. One name
        targets that port; several draw uniformly among them. Names must come
        from the host's own `cfg.dock.ports`.
      - "dock_pose": an explicit goal, 7 values [position xyz, quaternion
        wxyz] in the world frame, taken as given -- no port table is
        consulted. `info` then reports the pose under "goal_pose" but no
        "dock_port": the episode was aimed at a pose, not a named port.

    A naked reset keeps the pre-options behaviour: draw uniformly over
    `cfg.dock.ports`, or fly to `cfg.dock`'s single pose when none are
    configured.
    """

    cfg: BaseTaskConfig

    def _init_port_goals(self, view: TaskView) -> None:
        # Ports an episode may be assigned. Empty when the config names none,
        # and then nothing below is ever drawn or passed: `_dock_pose` stays
        # None, so the dynamics and the reward fall back to `cfg.dock` exactly
        # as they did before this field existed.
        self._port_names: tuple[str, ...] = tuple(p.name for p in self.cfg.dock.ports)
        self._port_targets = jnp.asarray(dock_port_targets(self.cfg), dtype=jnp.float32)
        self._port_index: int | None = None
        self._dock_pose: jnp.ndarray | None = None
        self._cfg_dock_target = jnp.asarray(dock_target(self.cfg), dtype=jnp.float32)
        # Telemetry regardless of what the observation carries: a run that
        # emits no goal block still wants to be told whether its policy is
        # closing in.
        self._jit_goal_error_norms = jax.jit(
            lambda state, target: goal_error_norms(dock_goal_error(view(state), target))
        )
        self._jit_dock_goal_error = self._build_jit_dock_goal_error(view)

    def _build_jit_dock_goal_error(self, view: TaskView) -> Any | None:
        """measured -> the goal-error block, or None when it isn't emitted.

        With no ports a naked reset always flies to `cfg.dock`, and that pose
        is compiled in as a constant -- the arrangement a config without ports
        had before ports existed, and float32 constant folding is not
        bit-identical to the same arithmetic on a runtime argument. A
        reset-options override can still retarget a no-ports episode, so the
        constant-folded form serves only while `_dock_pose` is None -- which a
        naked reset guarantees, keeping published no-ports observations
        byte-identical. A port set makes the target a per-episode argument
        read off `_dock_pose` at call time instead.
        """
        if not self.cfg.observation.goal_error:
            return None
        jitted = jax.jit(
            lambda measured, target: dock_goal_error(view(measured), target)
        )
        if not self._port_names:
            pose = self._cfg_dock_target
            folded = jax.jit(lambda measured: dock_goal_error(view(measured), pose))
            return lambda measured: (
                folded(measured)
                if self._dock_pose is None
                else jitted(measured, self._dock_pose)
            )
        return lambda measured: jitted(measured, self._dock_pose)

    def _begin_episode_goal(self, options: dict[str, Any] | None) -> None:
        """Assign this episode its goal; an override never survives a reset."""
        self._port_index = None
        self._dock_pose = None
        if not self._apply_goal_options(options):
            self._draw_port()

    def _draw_port(self) -> None:
        """Assign this episode a port, uniformly over the configured set."""
        if not self._port_names:
            return
        self._port_index = int(self.np_random.integers(len(self._port_names)))
        self._dock_pose = self._port_targets[self._port_index]

    def _apply_goal_options(self, options: dict[str, Any] | None) -> bool:
        """Point the episode where `reset(options=...)` says; False if it doesn't.

        See the class docstring for the contract. Only a draw among several
        names consumes randomness, so a single-name or explicit-pose reset
        leaves np_random's stream where a naked reset's port draw would have
        started.
        """
        parsed = parse_goal_options(options, self._port_names)
        if parsed is None:
            return False
        kind, value = parsed
        if kind == "pose":
            # In a no-ports config the jitted step has only ever seen
            # dock_pose as None; the first overridden episode hands it an
            # array and pays a one-off retrace. Both signatures stay cached
            # after that.
            self._dock_pose = value
            return True
        indices: tuple[int, ...] = value
        self._port_index = (
            indices[0]
            if len(indices) == 1
            else indices[int(self.np_random.integers(len(indices)))]
        )
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


def parse_goal_options(
    options: dict[str, Any] | None, port_names: tuple[str, ...]
) -> tuple[str, Any] | None:
    """Validated goal selection from reset options; None for a naked reset.

    Returns ("pose", (7,) jnp row) for an explicit target, or
    ("ports", tuple of indices into `port_names`) for a name-based menu. The
    one grammar behind the single-env mixin and the vector adapters, so the
    options contract and its errors cannot drift between them. A
    goal-selection API must not let a typo fall through to a random goal:
    unrecognised keys, malformed values and unknown names are errors, never
    a silent naked reset.
    """
    if not options:
        return None
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
        return ("pose", jnp.asarray(row))
    if port is None:
        return None
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
    unknown = [n for n in names if n not in port_names]
    if unknown:
        raise ValueError(
            f"unknown dock_port(s) {unknown}; this environment's configured "
            f"ports are {list(port_names)}"
            + ("" if port_names else " -- configure dock.ports or pass dock_pose")
        )
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"duplicate dock_port(s) {duplicates}; name each port at most once"
        )
    return ("ports", tuple(port_names.index(n) for n in names))
