"""Sensor/measurement noise for the ISS chaser state.

One JAX-traceable measurement function used everywhere noise is applied:
the Gymnasium envs' observations, and the rollout drivers' recorded
observations. Attitude noise composes a small random rotation onto the
quaternion -- additive component noise would leave the unit sphere and bias
the attitude estimate.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from pydantic import Field

from ...core.models import ConfigModel
from ...core.quaternion import quat_multiply, quat_normalize


# fold_in constant deriving a measurement-noise key stream from a dynamics
# key without consuming a draw from that key's own chain.
NOISE_STREAM = 0x5EED


class SensorNoiseConfig(ConfigModel):
    """Per-axis Gaussian measurement noise. Disabled by default.

    Position noise has a constant term and a range-proportional term
    (non-cooperative relative navigation degrades with distance); the two
    combine as independent variances.
    """

    enabled: bool = False
    sigma_pos_m: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    sigma_pos_frac_of_range: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    sigma_vel_m_s: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    sigma_att_rad: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    sigma_rate_rad_s: float = Field(default=0.0, ge=0, allow_inf_nan=False)


PRESETS: dict[str, SensorNoiseConfig] = {
    "off": SensorNoiseConfig(),
    # Differential-GNSS-class relative navigation (cooperative target).
    "cooperative": SensorNoiseConfig(
        enabled=True, sigma_pos_m=0.05, sigma_vel_m_s=0.002,
        sigma_att_rad=5e-5, sigma_rate_rad_s=1e-5,
    ),
    # Chaser-derived (vision/LIDAR-class) relative navigation: position
    # error grows with range, velocity estimate is coarser.
    "noncooperative": SensorNoiseConfig(
        enabled=True, sigma_pos_frac_of_range=0.01, sigma_vel_m_s=0.03,
        sigma_att_rad=5e-5, sigma_rate_rad_s=1e-5,
    ),
}


def apply_sensor_noise(
    state: jnp.ndarray, key: jax.Array, noise: SensorNoiseConfig
) -> jnp.ndarray:
    """True state (13,) -> measured state (13,). Identity when disabled."""
    if not noise.enabled:
        return state

    key_pos, key_vel, key_att, key_rate = jax.random.split(key, 4)
    pos, vel = state[0:3], state[3:6]
    q_bw, omega = state[6:10], state[10:13]

    sigma_pos = jnp.sqrt(
        jnp.asarray(noise.sigma_pos_m, jnp.float32) ** 2
        + (noise.sigma_pos_frac_of_range * jnp.linalg.norm(pos)) ** 2
    )
    pos_m = pos + sigma_pos * jax.random.normal(key_pos, (3,), jnp.float32)
    vel_m = vel + noise.sigma_vel_m_s * jax.random.normal(key_vel, (3,), jnp.float32)

    # Small random rotation: rotvec ~ N(0, sigma_att^2 I), composed in the
    # body frame. exp-map with a safe norm for the zero-angle case.
    rotvec = noise.sigma_att_rad * jax.random.normal(key_att, (3,), jnp.float32)
    angle = jnp.maximum(jnp.linalg.norm(rotvec), 1e-12)
    axis = rotvec / angle
    dq = jnp.concatenate([jnp.cos(angle / 2.0)[None], jnp.sin(angle / 2.0) * axis])
    q_m = quat_normalize(quat_multiply(q_bw, dq))
    q_m = jnp.where(jnp.dot(q_m, q_bw) < 0.0, -q_m, q_m)

    omega_m = omega + noise.sigma_rate_rad_s * jax.random.normal(key_rate, (3,), jnp.float32)
    return jnp.concatenate([pos_m, vel_m, q_m, omega_m])
