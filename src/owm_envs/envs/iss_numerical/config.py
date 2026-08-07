"""iss-numerical environment configuration: full numerical propagation of a
chief and chaser through Earth's zonal gravity field (plus optional
third-body and drag perturbations), with the chaser's attitude driven by
gravity-gradient torque.

Unlike `iss-hcw`, whose chaser motion is linearized (Clohessy-Wiltshire)
about an analytic Keplerian chief, `iss-numerical` integrates both the
chief and the chaser as independent ECI state vectors, so it can capture the
J2..J6 zonal harmonics, eclipse-dependent third-body pull, and drag that HCW
linearization cannot represent. `NumericalConfig` carries the shared chief
orbit (`OrbitConfig`) plus the perturbation switches (`PerturbationsConfig`).

`NUM_LAYOUT` names the raw state's slices: a `[jd, sec]` epoch prefix, the
chief's ECI position/velocity, the chaser's ECI position/velocity, the
chaser's body -> world quaternion, and the chaser's body-frame rates -- 21
elements in total (2 + 6 + 6 + 4 + 3). `NUM_LAYOUT.pos`/`vel`/`quat`/`omega`
therefore name the CHASER-ABSOLUTE (ECI) slices, not a world-frame relative
view: per `StateLayout.slice_view`'s own warning, an env whose raw slices
hold absolute/inertial quantities must not use `slice_view()` as the task
layer's view, since that would hand inertial numbers to code expecting a
relative one. `NUM_LAYOUT.slice_view()` is exposed only as the raw 13-vector
`[chaser_pos, chaser_vel, quat, omega]` for callers that explicitly want the
chaser's absolute state; the registry's canonical `view` for this env is the
computed relative view (chaser relative to chief) built in a later task.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from ...core.models import ConfigModel
from ..common.config import BaseTaskConfig, ObservationConfig
from ..common.epoch_state import EPOCH_LABELS
from ..common.layout import StateLayout
from ..common.orbit import OrbitConfig


class BallisticConfig(ConfigModel):
    """Cross-sectional area and drag coefficient for one vehicle."""

    # Strictly positive: `PerturbationsConfig.drag` scales the drag
    # acceleration by area, so zero or negative describes no vehicle.
    area_m2: float = Field(gt=0)
    cd: float = Field(default=2.2, gt=0)


class PerturbationsConfig(ConfigModel):
    """Which force-model terms beyond two-body point-mass gravity are active."""

    # 0 = point-mass gravity only (see envs.common.zonal_gravity
    # .accel_zonal_gravity's n_max < 2 branch); zonal_gravity has no J1 term
    # (a body's own center-of-mass frame has no dipole moment), so 1 is
    # meaningless, and terms above J6 aren't implemented there -- see the
    # validator below.
    zonal_max_degree: int = 0
    third_body_sun: bool = False
    third_body_moon: bool = False
    drag: bool = False
    chief_mass_kg: float = Field(default=419_700.0, gt=0)  # ISS
    chief_ballistic: BallisticConfig = Field(default_factory=lambda: BallisticConfig(area_m2=1500.0))
    chaser_ballistic: BallisticConfig = Field(default_factory=lambda: BallisticConfig(area_m2=50.0))
    # The chaser's mass comes from `physics.mass` -- the same value the
    # thruster dynamics use -- not a second field here, so drag and control
    # can never disagree about how heavy the chaser is.

    @field_validator("zonal_max_degree")
    @classmethod
    def _zonal_max_degree_is_supported(cls, value: int) -> int:
        if value == 1 or value < 0 or value > 6:
            raise ValueError(
                "zonal_max_degree must be 0 (point mass only) or in 2..6, got "
                f"{value}"
            )
        return value


ObservationMode = Literal["absolute", "chaser_absolute", "chief_absolute", "relative"]


class NumericalObservationConfig(ObservationConfig):
    """`ObservationConfig` plus which frame the emitted observation reports."""

    mode: ObservationMode = "relative"


class NumericalConfig(BaseTaskConfig):
    """`BaseTaskConfig` plus the chief's reference orbit and the perturbation
    force-model switches."""

    orbit: OrbitConfig = Field(default_factory=OrbitConfig)
    perturbations: PerturbationsConfig = Field(default_factory=PerturbationsConfig)
    observation: NumericalObservationConfig = Field(default_factory=NumericalObservationConfig)


# Width of the observation vector under each mode: the three absolute-frame
# modes report the full 21D state, and "relative" reports the canonical 15D
# view -- see the module docstring on why that view is computed rather than
# sliced.
OBS_MODE_DIM: dict[ObservationMode, int] = {
    "absolute": 21,
    "chaser_absolute": 21,
    "chief_absolute": 21,
    "relative": 15,
}


NUM_LAYOUT = StateLayout(
    state_dim=21,
    pos=slice(8, 11),
    vel=slice(11, 14),
    quat=slice(14, 18),
    omega=slice(18, 21),
    epoch=slice(0, 2),
    chief=slice(2, 8),
    labels=EPOCH_LABELS + (
        "chief_rx_eci_m", "chief_ry_eci_m", "chief_rz_eci_m",
        "chief_vx_eci_m_s", "chief_vy_eci_m_s", "chief_vz_eci_m_s",
        "chaser_rx_eci_m", "chaser_ry_eci_m", "chaser_rz_eci_m",
        "chaser_vx_eci_m_s", "chaser_vy_eci_m_s", "chaser_vz_eci_m_s",
        "q_bi_w", "q_bi_x", "q_bi_y", "q_bi_z",
        "omega_x_rad_s", "omega_y_rad_s", "omega_z_rad_s",
    ),
)
