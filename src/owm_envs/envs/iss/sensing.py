"""Sensor/measurement noise for the ISS chaser state.

One JAX-traceable measurement function used everywhere noise is applied:
the Gymnasium envs' observations, and the rollout drivers' recorded
observations. Attitude noise composes a small random rotation onto the
quaternion -- additive component noise would leave the unit sphere and bias
the attitude estimate.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from pydantic import Field, ValidationInfo, field_validator

from ...core.models import ConfigModel
from ...core.quaternion import quat_multiply, quat_normalize


# fold_in constant deriving a measurement-noise key stream from a dynamics
# key without consuming a draw from that key's own chain.
NOISE_STREAM = 0x5EED

_SQRT3 = math.sqrt(3.0)

Sigma = float | tuple[float, float, float]


class SensorNoiseConfig(ConfigModel):
    """Gaussian measurement noise. Disabled by default.

    sigma_pos_m, sigma_vel_m_s, sigma_att_rad, and sigma_rate_rad_s each take
    one of two forms:

      - a scalar s: the RMS of the TOTAL error -- the Euclidean norm of the
        3-vector error (for the attitude block, the total rotation angle).
        This is the number a requirements document quotes. Noise is
        isotropic in this form, with the per-axis sigma derived as
        s / sqrt(3) so that the norm of the per-axis draws has RMS s.
      - a 3-tuple (sx, sy, sz): explicit PER-AXIS sigmas, for anisotropic
        sensors (e.g. a star tracker's cross-boresight rating differing
        from its boresight rating).

    sigma_pos_frac_of_range stays scalar-only, with the same total-RMS
    convention: its per-axis contribution is (frac * range) / sqrt(3),
    RSS-combined per axis with sigma_pos_m's per-axis contribution. "1% of
    range" then means 1 m of total position error at 100 m range.

    Position noise has a constant term and a range-proportional term
    (non-cooperative relative navigation degrades with distance); the two
    combine as independent variances, per axis.
    """

    enabled: bool = False
    sigma_pos_m: Sigma = 0.0
    sigma_pos_frac_of_range: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    sigma_vel_m_s: Sigma = 0.0
    sigma_att_rad: Sigma = 0.0
    sigma_rate_rad_s: Sigma = 0.0

    @field_validator(
        "sigma_pos_m", "sigma_vel_m_s", "sigma_att_rad", "sigma_rate_rad_s"
    )
    @classmethod
    def _validate_sigma(cls, value: Sigma, info: ValidationInfo) -> Sigma:
        components = value if isinstance(value, tuple) else (value,)
        if any(not math.isfinite(c) or c < 0 for c in components):
            raise ValueError(f"{info.field_name} must be finite and non-negative")
        return value


PRESETS: dict[str, SensorNoiseConfig] = {
    "off": SensorNoiseConfig(),
    # Differential-GNSS-class relative navigation (cooperative target).
    # Numbers are total-RMS: Euclidean-norm position/velocity error, total
    # rotation-angle/rate error.
    "cooperative": SensorNoiseConfig(
        enabled=True, sigma_pos_m=0.05, sigma_vel_m_s=0.002,
        sigma_att_rad=5e-5, sigma_rate_rad_s=1e-5,
    ),
    # Chaser-derived (vision/LIDAR-class) relative navigation: position
    # error grows with range, velocity estimate is coarser. Numbers are
    # total-RMS (see "cooperative" above).
    "noncooperative": SensorNoiseConfig(
        enabled=True, sigma_pos_frac_of_range=0.01, sigma_vel_m_s=0.03,
        sigma_att_rad=5e-5, sigma_rate_rad_s=1e-5,
    ),
}


def _per_axis_sigma(value: Sigma) -> jnp.ndarray:
    """Resolve a scalar-or-vector sigma to a static per-axis 3-vector.

    A scalar s is the RMS of the total (Euclidean-norm) error; dividing by
    sqrt(3) gives the isotropic per-axis value that makes that hold exactly.
    A 3-tuple is already per-axis and passes through unchanged. `value`
    comes from a frozen config, so this always resolves to a concrete
    (non-traced) array.
    """
    if isinstance(value, tuple):
        return jnp.asarray(value, jnp.float32)
    return jnp.full(3, value / _SQRT3, dtype=jnp.float32)


def apply_sensor_noise(
    state: jnp.ndarray, key: jax.Array, noise: SensorNoiseConfig
) -> jnp.ndarray:
    """True state (13,) -> measured state (13,). Identity when disabled."""
    if not noise.enabled:
        return state

    key_pos, key_vel, key_att, key_rate = jax.random.split(key, 4)
    pos, vel = state[0:3], state[3:6]
    q_bw, omega = state[6:10], state[10:13]

    sigma_range = noise.sigma_pos_frac_of_range * jnp.linalg.norm(pos) / _SQRT3
    sigma_pos = jnp.sqrt(_per_axis_sigma(noise.sigma_pos_m) ** 2 + sigma_range**2)
    pos_m = pos + sigma_pos * jax.random.normal(key_pos, (3,), jnp.float32)

    sigma_vel = _per_axis_sigma(noise.sigma_vel_m_s)
    vel_m = vel + sigma_vel * jax.random.normal(key_vel, (3,), jnp.float32)

    # Small random rotation: rotvec ~ N(0, diag(sigma_att^2)), composed in
    # the body frame. exp-map with a safe norm for the zero-angle case.
    sigma_att = _per_axis_sigma(noise.sigma_att_rad)
    rotvec = sigma_att * jax.random.normal(key_att, (3,), jnp.float32)
    angle = jnp.maximum(jnp.linalg.norm(rotvec), 1e-12)
    axis = rotvec / angle
    dq = jnp.concatenate([jnp.cos(angle / 2.0)[None], jnp.sin(angle / 2.0) * axis])
    q_m = quat_normalize(quat_multiply(q_bw, dq))
    q_m = jnp.where(jnp.dot(q_m, q_bw) < 0.0, -q_m, q_m)

    sigma_rate = _per_axis_sigma(noise.sigma_rate_rad_s)
    omega_m = omega + sigma_rate * jax.random.normal(key_rate, (3,), jnp.float32)
    return jnp.concatenate([pos_m, vel_m, q_m, omega_m])
