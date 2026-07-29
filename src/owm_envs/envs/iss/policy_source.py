"""Adapts the ISS JAX policies to the backend-agnostic PolicySource protocol.

This is where numpy observations become JAX arrays and JAX actions become
numpy again, and where the PRNG key derives from the integer seed
`VectorEnvDriver` hands to `new_episode`. Everything JAX-specific about
dataset generation for the ISS backend lives here, not in `drivers/`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import ISSConfig
from .policies import EXTRAS_DIM, PolicyConfig, make_policy

# extras[0] holds the sub-policy index for the union mixture (see policies.py).
_UNION_POLICY_IDX = 0


class _EpisodeState(NamedTuple):
    key: jax.Array
    extras: jnp.ndarray


class IssPolicySource:
    """Wraps `make_policy` behind the backend-agnostic PolicySource protocol."""

    def __init__(self, cfg: ISSConfig, policy_cfg: PolicyConfig):
        self.records_policy_ids = policy_cfg.type == "union"
        self._policy_fn, self._extras_fn = make_policy(cfg, policy_cfg)
        self._extras_width = EXTRAS_DIM[policy_cfg.type]

    def new_episode(self, seed: int) -> _EpisodeState:
        key = jax.random.PRNGKey(seed)
        key, extras_key = jax.random.split(key)
        if self._extras_fn is None:
            extras = jnp.zeros((self._extras_width,), dtype=jnp.float32)
        else:
            extras = self._extras_fn(extras_key)
        return _EpisodeState(key=key, extras=extras)

    def act(self, observation: np.ndarray, episode_state: _EpisodeState, step: int) -> np.ndarray:
        # Fold the step index into the episode's base key rather than
        # threading mutable RNG state through the (deliberately opaque,
        # immutable) episode_state the driver holds.
        act_key = jax.random.fold_in(episode_state.key, step)
        action = self._policy_fn(jnp.asarray(observation), act_key, episode_state.extras)
        return np.asarray(action, dtype=np.float32)

    def policy_id(self, episode_state: _EpisodeState) -> int:
        if episode_state.extras.shape[0] == 0:
            return 0
        return int(episode_state.extras[_UNION_POLICY_IDX])
