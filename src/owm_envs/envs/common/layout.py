"""StateLayout: named slices describing an environment's raw state vector.

The layout drives state labels, dataset state widths, and per-slice sensor
noise. Task-layer functions (reward, goal, policies, event checks) never
index raw state directly -- they consume the canonical 13D world-frame
relative view `[pos(3), vel(3), q_bw(4), omega(3)]`, which `slice_view()`
extracts for envs whose raw slices already hold it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp

VIEW_DIM = 13

VIEW_LABELS: tuple[str, ...] = (
    "rel_x_m", "rel_y_m", "rel_z_m",
    "rel_vx_m_s", "rel_vy_m_s", "rel_vz_m_s",
    "q_w", "q_x", "q_y", "q_z",
    "omega_x_rad_s", "omega_y_rad_s", "omega_z_rad_s",
)


@dataclass(frozen=True)
class StateLayout:
    state_dim: int
    pos: slice
    vel: slice
    quat: slice
    omega: slice
    epoch: slice | None = None
    chief: slice | None = None
    labels: tuple[str, ...] = field(default=VIEW_LABELS)

    def __post_init__(self) -> None:
        if len(self.labels) != self.state_dim:
            raise ValueError(
                f"labels has {len(self.labels)} entries for state_dim {self.state_dim}"
            )
        if not (self.pos.stop == self.vel.start and self.vel.stop == self.quat.start
                and self.quat.stop == self.omega.start):
            raise ValueError("pos/vel/quat/omega must be contiguous, in that order")
        for name, sl, width in (("pos", self.pos, 3), ("vel", self.vel, 3),
                                 ("quat", self.quat, 4), ("omega", self.omega, 3)):
            if len(range(sl.start, sl.stop, sl.step or 1)) != width:
                raise ValueError(f"{name} must have width {width}, got {sl}")

    def slice_view(self, state: jnp.ndarray) -> jnp.ndarray:
        """The canonical 13D relative view, sliced straight out of `state`.

        The four slices are contiguous (enforced above), so this is a single
        slice -- for the iss layout it selects the whole state, and XLA sees
        the same values either way, which is what keeps iss bit-identical.

        This is correct ONLY when the env's raw pos/vel/quat/omega slices
        already hold world-frame relative quantities, as the iss layout's
        do. An env whose slices hold absolute or inertial state -- a future
        iss-numerical, say, carrying chief and deputy orbital elements --
        must NOT use this: slicing would hand the task layer inertial
        numbers labelled as relative ones. Such envs supply a computed view
        through the registry and the task layer consumes that instead.
        """
        return state[..., self.pos.start : self.omega.stop]


ISS_LAYOUT = StateLayout(
    state_dim=13,
    pos=slice(0, 3),
    vel=slice(3, 6),
    quat=slice(6, 10),
    omega=slice(10, 13),
)
