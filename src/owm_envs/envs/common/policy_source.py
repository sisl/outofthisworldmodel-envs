"""Adapts the JAX task-layer policies to the backend-agnostic PolicySource protocol.

This is where numpy observations become JAX arrays and JAX actions become
numpy again, and where the PRNG key derives from the integer seed
`VectorEnvDriver` hands to `new_episode`. Everything JAX-specific about
dataset generation lives here, not in `drivers/`.

The scripted policies wrapped here are task-layer code serving every env in
the suite, not one backend's: they read the canonical 13D relative view, and
`view` is what extracts it from whatever state the env itself carries.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import BaseTaskConfig
from .goal import make_augment
from .policies import EXTRAS_DIM, PolicyConfig, dock_target_selector, make_policy

# extras[0] holds the sub-policy index for the union mixture (see policies.py).
_UNION_POLICY_IDX = 0


class _EpisodeState(NamedTuple):
    key: jax.Array
    extras: jnp.ndarray


class TaskPolicySource:
    """Wraps `make_policy` behind the backend-agnostic PolicySource protocol."""

    def __init__(
        self,
        cfg: BaseTaskConfig,
        policy_cfg: PolicyConfig,
        view: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    ):
        # `view` extracts the canonical 13D relative view the scripted
        # policies and the goal-error block read; None means the env's state
        # already is that view, as it is for the iss env.
        self._view = view if view is not None else (lambda m: m)
        self.records_policy_ids = policy_cfg.type == "union"
        # Every episode in the suite has a dock target, so this source supplies
        # one: the assigned port's pose under a port set, the `DockConfig`
        # pose otherwise. It comes from the same selector the goal-error block
        # uses, so the recorded pose is the one the episode was scored against.
        self.records_dock_targets = True
        self._select_dock_target = dock_target_selector(cfg, policy_cfg)
        self._policy_fn, self._extras_fn = make_policy(cfg, policy_cfg)
        self._extras_width = EXTRAS_DIM[policy_cfg.type]
        self._observe = policy_cfg.observe
        # No `observe` hook threaded through here (unlike `ScanDriver`): this
        # source never re-derives the recorded row from `measured` -- see
        # `augment_observation`, which always hands `make_augment`'s `observed`
        # override the driver's own observation instead.
        self._augment = make_augment(cfg, policy_cfg, view=view)

    def new_episode(self, seed: int) -> _EpisodeState:
        key = jax.random.PRNGKey(seed)
        key, extras_key = jax.random.split(key)
        if self._extras_fn is None:
            extras = jnp.zeros((self._extras_width,), dtype=jnp.float32)
        else:
            extras = self._extras_fn(extras_key)
        return _EpisodeState(key=key, extras=extras)

    def act(
        self, observation: np.ndarray, episode_state: _EpisodeState, step: int, info: dict
    ) -> np.ndarray:
        # Fold the step index into the episode's base key rather than
        # threading mutable RNG state through the (deliberately opaque,
        # immutable) episode_state the driver holds.
        act_key = jax.random.fold_in(episode_state.key, step)
        # observe="state" flies the policy on the true state (info["state"]);
        # "measurement" (the default) flies it on the measured one, which is
        # info["measured_state"] when the env publishes it (iss-numerical: the
        # recorded observation can be mode-shaped and so not something `view`
        # -- built for the raw layout -- can read at all) and the recorded
        # observation otherwise (iss/iss-hcw, whose observation IS that raw
        # layout, unchanged from before this key existed).
        if self._observe == "state":
            policy_input = info.get("state", observation)
        else:
            policy_input = info.get("measured_state", observation)
        action = self._policy_fn(
            self._view(jnp.asarray(policy_input)), act_key, episode_state.extras
        )
        return np.asarray(action, dtype=np.float32)

    def augment_observation(
        self, observation: np.ndarray, episode_state: _EpisodeState, info: dict
    ) -> np.ndarray:
        if self._augment is None:
            return observation
        # `measured` is what the goal-error block is computed FROM (via
        # `view`, e.g. iss-numerical's raw 21D layout, which `observation` may
        # not structurally be -- see `act()`); `observed` is what gets
        # RECORDED, and is always `observation` itself, verbatim -- never
        # re-derived from `measured` through an `observe` hook. Re-deriving it
        # would difference `measured`'s already-float32-narrowed ~6.8e6 m ECI
        # columns a second time (`make_augment`'s docstring), quantizing the
        # recorded row to ~1 m against the 0.1 m dock gate for no reason: the
        # env already computed this exact row once, correctly, and handed it
        # here as `observation`.
        measured = info.get("measured_state", observation)
        augmented = self._augment(
            jnp.asarray(measured), episode_state.extras, observed=jnp.asarray(observation)
        )
        return np.asarray(augmented, dtype=np.float32)

    def policy_id(self, episode_state: _EpisodeState) -> int:
        if episode_state.extras.shape[0] == 0:
            return 0
        return int(episode_state.extras[_UNION_POLICY_IDX])

    def dock_target(self, episode_state: _EpisodeState) -> np.ndarray:
        return np.asarray(self._select_dock_target(episode_state.extras), dtype=np.float32)
