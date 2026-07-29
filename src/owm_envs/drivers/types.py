"""The driver seam.

A driver turns an environment plus a policy into a TrajectoryBatch. Two exist:
VectorEnvDriver, which needs only a gymnasium VectorEnv and therefore works with
any backend; and ScanDriver, which fuses a whole horizon into one lax.scan and
therefore needs a JAX-traceable backend.

Nothing in this module imports JAX. That is the point: adding a Basilisk or
brahe environment must not require a JAX-traceable step function, only the
Gymnasium pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class RolloutSpec:
    """What to generate. Deliberately backend-agnostic."""

    num_episodes: int
    max_steps: int
    seed: int


@dataclass(frozen=True)
class TrajectoryBatch:
    """Episodes as zero-padded arrays plus their true lengths.

    Padded rather than ragged so that two drivers' outputs can be compared
    elementwise by the equivalence test -- a ragged comparison would need
    bespoke logic that could itself be wrong.

    Convention, carried from seamstress: an episode of N environment steps
    stores N + 1 observations, N + 1 actions, and N + 1 rewards, so `length`
    counts observations, not steps. `observations[length - 1]` is the
    TERMINAL state -- the collision, the successful dock, or the final state
    at truncation -- and storing it is the point: without it the dataset
    would contain no collision or docking states for a world model to learn
    from. `actions[i]` takes `observations[i]` to `observations[i + 1]` for i
    in [0, length - 2]; `actions[length - 1]` is a zero-padded no-op past the
    terminal state, and `rewards[length - 1]` is 0.0.
    """

    observations: np.ndarray  # (E, T, obs_dim) float32
    actions: np.ndarray  # (E, T, act_dim) float32
    rewards: np.ndarray  # (E, T) float32
    lengths: np.ndarray  # (E,) int32
    terminated: np.ndarray  # (E,) bool
    truncated: np.ndarray  # (E,) bool
    policy_ids: np.ndarray | None  # (E,) int32, or None when not a mixture

    @property
    def num_episodes(self) -> int:
        return int(self.observations.shape[0])

    @property
    def total_transitions(self) -> int:
        return int(self.lengths.sum())

    def validate(self) -> None:
        """Raise ValueError on any internal inconsistency."""
        n, t = self.observations.shape[0], self.observations.shape[1]

        for name, arr, expected_leading in (
            ("actions", self.actions, n),
            ("rewards", self.rewards, n),
            ("lengths", self.lengths, n),
            ("terminated", self.terminated, n),
            ("truncated", self.truncated, n),
        ):
            if arr.shape[0] != expected_leading:
                raise ValueError(
                    f"episode count mismatch: observations has {n}, {name} has {arr.shape[0]}"
                )
        if self.policy_ids is not None and self.policy_ids.shape[0] != n:
            raise ValueError(f"episode count mismatch: policy_ids has {self.policy_ids.shape[0]}, expected {n}")

        if self.actions.shape[1] != t or self.rewards.shape[1] != t:
            raise ValueError("time dimension mismatch between observations, actions and rewards")

        if np.any(self.lengths > t):
            raise ValueError(f"episode length exceeds padded width {t}: max is {int(self.lengths.max())}")
        if np.any(self.lengths < 1):
            raise ValueError("episode length must be at least 1")

        both = np.logical_and(self.terminated, self.truncated)
        if np.any(both):
            raise ValueError(f"episodes {np.flatnonzero(both).tolist()} are flagged both terminated and truncated")

        for name, arr in (("observations", self.observations), ("actions", self.actions), ("rewards", self.rewards)):
            for i, length in enumerate(self.lengths):
                if length < t and np.any(arr[i, length:] != 0):
                    raise ValueError(f"{name} episode {i} has non-zero padding past length {length}")


class Driver(Protocol):
    """Turns a spec into trajectories. Implementations must be deterministic in
    `spec.seed`."""

    def generate(self, spec: RolloutSpec) -> TrajectoryBatch: ...


class PolicySource(Protocol):
    """Supplies actions to a driver. Backend-specific; the driver treats it as opaque.

    The driver never sees a JAX key, a JAX array, or a backend-specific config
    type -- the source owns all of that internally, and hands back only numpy
    arrays and an opaque `episode_state` it doesn't examine.
    """

    records_policy_ids: bool

    def new_episode(self, seed: int) -> Any:
        """Return opaque per-episode state (e.g. sampled hyperparameters)."""

    def act(self, observation: np.ndarray, episode_state: Any, step: int) -> np.ndarray:
        """Return an action as a numpy array, given a numpy observation."""

    def policy_id(self, episode_state: Any) -> int:
        """Sub-policy index for mixture policies; 0 when not a mixture."""


def pack_episodes(
    episodes: list[dict],
    obs_dim: int,
    act_dim: int,
    records_policy_ids: bool,
) -> TrajectoryBatch:
    """Pad a list of variable-length episodes into a TrajectoryBatch.

    Shared by BOTH drivers on purpose: a second copy of this padding logic is
    exactly the kind of drift the driver-equivalence test exists to catch, so
    there is only one.

    Each episode dict carries `obs` (L, obs_dim), `act` (L, act_dim), `rew` (L,),
    `terminated` bool, `truncated` bool, and `policy_id` int.
    """
    n = len(episodes)
    if n == 0:
        raise ValueError("cannot pack an empty episode list")
    max_len = max(int(e["obs"].shape[0]) for e in episodes)

    observations = np.zeros((n, max_len, obs_dim), dtype=np.float32)
    actions = np.zeros((n, max_len, act_dim), dtype=np.float32)
    rewards = np.zeros((n, max_len), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int32)
    terminated = np.zeros(n, dtype=bool)
    truncated = np.zeros(n, dtype=bool)
    policy_ids = np.zeros(n, dtype=np.int32) if records_policy_ids else None

    for i, episode in enumerate(episodes):
        length = int(episode["obs"].shape[0])
        observations[i, :length] = episode["obs"]
        actions[i, :length] = episode["act"]
        rewards[i, :length] = episode["rew"]
        lengths[i] = length
        terminated[i] = episode["terminated"]
        truncated[i] = episode["truncated"]
        if policy_ids is not None:
            policy_ids[i] = episode["policy_id"]

    batch = TrajectoryBatch(
        observations=observations,
        actions=actions,
        rewards=rewards,
        lengths=lengths,
        terminated=terminated,
        truncated=truncated,
        policy_ids=policy_ids,
    )
    batch.validate()
    return batch
