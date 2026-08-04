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

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

# Salts a transitions-mode chunk seed so it does not collide with the
# episodes-mode stream at the same spec.seed. numpy's SeedSequence absorbs
# trailing zeros, so `default_rng([seed, 0])` -- chunk 0's seed with no salt
# -- draws byte-identically to `default_rng(seed)`, meaning a transitions-mode
# run would silently reproduce episodes-mode's dataset at a shared seed.
TRANSITIONS_STREAM = 0x7C5


@dataclass(frozen=True)
class RolloutSpec:
    """What to generate. Deliberately backend-agnostic.

    Exactly one of `num_episodes` or `min_transitions` sizes the rollout;
    see `__post_init__`.
    """

    num_episodes: int | None = field(default=None, kw_only=True)
    max_steps: int
    seed: int
    min_transitions: int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if (self.num_episodes is None) == (self.min_transitions is None):
            raise ValueError(
                "exactly one of num_episodes or min_transitions must be set"
            )
        if self.min_transitions is not None and self.min_transitions < 1:
            raise ValueError(
                f"min_transitions must be >= 1, got {self.min_transitions}"
            )


@dataclass(frozen=True)
class TrajectoryBatch:
    """Episodes as zero-padded arrays plus their true lengths.

    Array dimensions:
      E        episode count, `num_episodes`
      T        padded time width: the longest episode's length. Every episode
               occupies a T-wide row regardless of its own length, with the
               unused tail zero-filled, so shorter episodes carry padding
               and only `lengths[e]` entries of row `e` are real.
      obs_dim  observation vector width (13 for the ISS environment, 25 with the goal-error block)
      act_dim  action vector width (6 for the ISS environment)

    Padded rather than ragged so that two drivers' outputs can be compared
    elementwise by the equivalence test -- a ragged comparison would need
    bespoke logic that could itself be wrong.

    WHERE AN EPISODE ENDS. `lengths[e]` is the single authority and counts
    OBSERVATIONS, not steps or transitions. An episode of N environment steps
    stores N + 1 observations, N + 1 actions, and N + 1 rewards, so
    `lengths[e] == N + 1`. Nothing else needs to be inspected to find the
    boundary: padding is not a reliable end marker, because a real trailing
    observation can itself be all-zero.

    Slicing an episode:
      obs = observations[e, : lengths[e]]        # all real, terminal included
      act = actions[e, : lengths[e] - 1]         # real actions only
      rew = rewards[e, : lengths[e] - 1]         # real rewards only

    `observations[e, lengths[e] - 1]` is the TERMINAL state -- the collision,
    the successful dock, or the final state at truncation -- and storing it is
    the point: without it the dataset would contain no collision or docking
    states for a world model to learn from. `terminated[e]` and `truncated[e]`
    say which of those it is; both drivers set exactly one of the two, and
    `validate` rejects any episode that sets both.

    `actions[e, i]` takes `observations[e, i]` to `observations[e, i + 1]` for
    i in [0, lengths[e] - 2], which is why the action and reward slices above
    stop one short. The final in-length slot is not a real step:
    `actions[e, lengths[e] - 1]` is a zero pad and `rewards[e, lengths[e] - 1]`
    is 0.0, because no action was taken from the terminal state. Reading
    actions out to `lengths[e]` therefore injects one spurious zero action per
    episode; `total_transitions` and `compute_norm_stats` both subtract it.
    """

    observations: np.ndarray  # (E, T, obs_dim) float32
    actions: np.ndarray  # (E, T, act_dim) float32
    rewards: np.ndarray  # (E, T) float32
    lengths: np.ndarray  # (E,) int32, real observations per episode
    terminated: np.ndarray  # (E,) bool
    truncated: np.ndarray  # (E,) bool
    policy_ids: np.ndarray | None  # (E,) int32, or None when not a mixture
    # (E, 7) float32 [position, quaternion] dock target per episode: the pose
    # that episode's control law regulated toward and that its success gate
    # and goal-error block were scored against. The pose itself rather than an
    # index into the port table, because the table's indices shift whenever a
    # port is added or removed, so a stored index stops reproducing the
    # episode as soon as the table changes. None only when the source cannot
    # supply one -- every ISS driver can, recording the `DockConfig` pose for
    # a run with no port set.
    dock_targets: np.ndarray | None = None

    @property
    def num_episodes(self) -> int:
        return int(self.observations.shape[0])

    @property
    def total_transitions(self) -> int:
        # `lengths` counts observations, not transitions: the last stored
        # observation per episode is the terminal state, paired with the
        # synthetic zero-pad action rather than a real one, so it isn't a
        # usable (obs, act, next_obs) transition. An episode of length L
        # contributes L - 1 transitions; clamp at 0 for the L == 1 edge case
        # (a single observation, no step taken).
        return int(np.maximum(self.lengths - 1, 0).sum())

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
        if self.dock_targets is not None:
            if self.dock_targets.shape[0] != n:
                raise ValueError(
                    f"episode count mismatch: dock_targets has {self.dock_targets.shape[0]}, expected {n}"
                )
            if self.dock_targets.shape[1:] != (7,):
                raise ValueError(
                    f"dock_targets rows must be (7,) [position, quaternion], got {self.dock_targets.shape[1:]}"
                )

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
    arrays and an opaque `episode_state` it doesn't examine. `info` is the
    standard Gym info dict for that lane (numpy values), passed through
    unexamined by the driver -- a policy source that needs more than the
    observation (e.g. the true state under sensor noise) reads it from there.
    """

    records_policy_ids: bool
    records_dock_targets: bool

    def new_episode(self, seed: int) -> Any:
        """Return opaque per-episode state (e.g. sampled hyperparameters)."""

    def act(
        self, observation: np.ndarray, episode_state: Any, step: int, info: dict
    ) -> np.ndarray:
        """Return an action as a numpy array, given a numpy observation."""

    def augment_observation(self, observation: np.ndarray, episode_state: Any) -> np.ndarray:
        """Return the observation to RECORD, given the raw one the driver just
        received. This only affects what gets stored into the episode buffer
        -- `act()` above always receives the raw observation, unaugmented.
        Identity (`return observation`) for sources with nothing to append;
        `ISSPolicySource` is the only implementor that does otherwise (see
        `envs.iss.goal.make_augment`), appending a policy-aware goal-error
        block built from `episode_state`.
        """

    def policy_id(self, episode_state: Any) -> int:
        """Sub-policy index for mixture policies; 0 when not a mixture."""

    def dock_target(self, episode_state: Any) -> np.ndarray:
        """The (7,) [position, quaternion] pose this episode is docking to."""


def pack_episodes(
    episodes: list[dict],
    obs_dim: int,
    act_dim: int,
    records_policy_ids: bool,
    records_dock_targets: bool = False,
) -> TrajectoryBatch:
    """Pad a list of variable-length episodes into a TrajectoryBatch.

    Shared by BOTH drivers on purpose: a second copy of this padding logic is
    exactly the kind of drift the driver-equivalence test exists to catch, so
    there is only one.

    Each episode dict carries `obs` (L, obs_dim), `act` (L, act_dim), `rew` (L,),
    `terminated` bool, `truncated` bool, `policy_id` int, and, when
    `records_dock_targets`, `dock_target` (7,).
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
    dock_targets = np.zeros((n, 7), dtype=np.float32) if records_dock_targets else None

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
        if dock_targets is not None:
            dock_targets[i] = episode["dock_target"]

    batch = TrajectoryBatch(
        observations=observations,
        actions=actions,
        rewards=rewards,
        lengths=lengths,
        terminated=terminated,
        truncated=truncated,
        policy_ids=policy_ids,
        dock_targets=dock_targets,
    )
    batch.validate()
    return batch
