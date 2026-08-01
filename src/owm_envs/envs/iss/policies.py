"""Scripted JAX policies for ISS dataset generation.

Each builder returns (policy_fn, extras_fn):
  policy_fn(state, key, extras) -> action (6,)
  extras_fn(key)                -> per-episode hyperparameters, sampled at reset
extras_fn is None when the policy needs no per-episode randomisation.
"""

from __future__ import annotations

from typing import Callable, Literal

import jax
import jax.numpy as jnp
from pydantic import Field, field_validator

from ...core.models import ConfigModel
from ...core.quaternion import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_from_body_z_to,
    quat_multiply,
    quat_normalize,
    quat_to_rotmat,
)
from .config import ISSConfig

PolicyFn = Callable[[jnp.ndarray, jax.Array, jnp.ndarray], jnp.ndarray]
ExtrasFn = Callable[[jax.Array], jnp.ndarray]

# Width of each policy's extras vector. The rollout driver in plan 2 needs this
# to allocate the per-env extras buffer before the first reset.
EXTRAS_DIM: dict[str, int] = {"random": 0, "orbit": 5, "dock": 0, "union": 6}


class OrbitParams(ConfigModel):
    radius_range_m: tuple[float, float] = (60.0, 180.0)
    angular_speed_range_rad_s: tuple[float, float] = (0.09, 0.24)
    kp_position: float = 1080.0
    kd_velocity: float = 1500.0
    kp_attitude: float = 54_000.0
    kd_attitude: float = 47_000.0


class DockParams(ConfigModel):
    kp_position: float = 1080.0
    kd_velocity: float = 7200.0
    kp_attitude: float = 54_000.0
    kd_attitude: float = 132_000.0


class PolicyConfig(ConfigModel):
    """Policy settings. A ConfigModel like ISSConfig so the policy that shaped
    a dataset is part of its as-run record, not just the environment physics."""

    # Which policy make_policy builds. Defaults to "random". Invalid values
    # are rejected here, at config-load time, rather than inside make_policy
    # at rollout time.
    type: Literal["random", "orbit", "dock", "union"] = "random"
    # What the scripted policies consume during data generation. Defaults to
    # "measurement": policies act on the same noisy measurement the dataset
    # records, so the recorded action-outcome pairs carry the aleatoric
    # uncertainty of acting on an observed rather than true state -- the
    # uncertainty a world model should learn. "state" gives noise-independent
    # trajectories, for controlled ablations where clean/noisy runs must
    # share trajectories.
    observe: Literal["state", "measurement"] = "measurement"
    orbit: OrbitParams = Field(default_factory=OrbitParams)
    dock: DockParams = Field(default_factory=DockParams)
    # Positional order: [random, orbit, dock]. Normalised at build time.
    union_weights: tuple[float, float, float] = (0.3, 0.35, 0.35)

    @field_validator("union_weights")
    @classmethod
    def _validate_union_weights(
        cls, v: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        for w in v:
            if w < 0.0:
                raise ValueError(f"union_weights must be non-negative, got {w}")
        if sum(v) <= 0.0:
            raise ValueError(f"union_weights must sum to > 0, got {v}")
        return v


def _safe_norm(v: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    return jnp.maximum(jnp.linalg.norm(v), eps)


def _attitude_torque(
    q_bw: jnp.ndarray,
    q_target: jnp.ndarray,
    omega_b: jnp.ndarray,
    *,
    inertia_diag: jnp.ndarray,
    kp_attitude: float,
    kd_attitude: float,
) -> jnp.ndarray:
    q_err = quat_multiply(quat_conjugate(q_bw), q_target)
    att_err_body = axis_angle_from_quat(q_err)
    return (
        inertia_diag
        * (kp_attitude * att_err_body - kd_attitude * omega_b)
        / jnp.max(inertia_diag)
    )


def _build_random(cfg: ISSConfig) -> PolicyFn:
    low = jnp.array(
        [-cfg.control.limit_force_n] * 3 + [-cfg.control.limit_torque_nm] * 3,
        dtype=jnp.float32,
    )
    high = -low

    def policy_fn(state, key, extras):
        del state, extras
        return jax.random.uniform(key, shape=(6,), minval=low, maxval=high)

    return policy_fn


def _build_orbit(cfg: ISSConfig, params: OrbitParams) -> tuple[PolicyFn, ExtrasFn]:
    # This "orbit" is not a passively stable relative orbit -- it is a
    # circular trajectory at a commanded angular speed around the target,
    # held by continuous control. That's useful for training precisely
    # because it forces trajectories that use control actuation to maintain
    # both centre-pointing attitude and speed, exercising more of the action
    # space and showing the consequences of actions, which helps a policy
    # learn what its actions do.
    inertia_diag = jnp.asarray(cfg.physics.inertia_diag, dtype=jnp.float32)
    radius_range = jnp.asarray(params.radius_range_m, dtype=jnp.float32)
    omega_range = jnp.asarray(params.angular_speed_range_rad_s, dtype=jnp.float32)

    def extras_fn(key: jax.Array) -> jnp.ndarray:
        key_axis, key_radius, key_omega = jax.random.split(key, 3)
        axis = jax.random.normal(key_axis, (3,), dtype=jnp.float32)
        axis = axis / _safe_norm(axis)
        radius = jax.random.uniform(
            key_radius, (), minval=radius_range[0], maxval=radius_range[1], dtype=jnp.float32
        )
        omega = jax.random.uniform(
            key_omega, (), minval=omega_range[0], maxval=omega_range[1], dtype=jnp.float32
        )
        return jnp.concatenate([axis, radius[None], omega[None]], axis=0)

    def policy_fn(state, key, extras):
        del key
        axis_world = extras[0:3]
        target_radius = extras[3]
        orbit_omega = extras[4]

        pos_w, vel_w = state[0:3], state[3:6]
        q_bw = quat_normalize(state[6:10])
        omega_b = state[10:13]

        # Project onto the orbital plane to get the desired point and tangential speed.
        pos_planar = pos_w - jnp.dot(pos_w, axis_world) * axis_world
        r_hat = pos_planar / _safe_norm(pos_planar)
        p_des = target_radius * r_hat
        v_des = orbit_omega * target_radius * jnp.cross(axis_world, r_hat)

        force_world = -params.kp_position * (pos_w - p_des) - params.kd_velocity * (vel_w - v_des)
        force_body = quat_to_rotmat(q_bw).T @ force_world

        torque_body = _attitude_torque(
            q_bw, quat_from_body_z_to(-r_hat), omega_b,
            inertia_diag=inertia_diag,
            kp_attitude=params.kp_attitude, kd_attitude=params.kd_attitude,
        )
        return jnp.concatenate([force_body, torque_body], axis=0)

    return policy_fn, extras_fn


def _build_dock(cfg: ISSConfig, params: DockParams) -> PolicyFn:
    inertia_diag = jnp.asarray(cfg.physics.inertia_diag, dtype=jnp.float32)
    dock_pos = jnp.asarray(cfg.dock.position, dtype=jnp.float32)
    dock_quat = quat_normalize(jnp.asarray(cfg.dock.quaternion, dtype=jnp.float32))

    def policy_fn(state, key, extras):
        del key, extras
        pos_w, vel_w = state[0:3], state[3:6]
        q_bw = quat_normalize(state[6:10])
        omega_b = state[10:13]

        force_world = -params.kp_position * (pos_w - dock_pos) - params.kd_velocity * vel_w
        force_body = quat_to_rotmat(q_bw).T @ force_world

        torque_body = _attitude_torque(
            q_bw, dock_quat, omega_b,
            inertia_diag=inertia_diag,
            kp_attitude=params.kp_attitude, kd_attitude=params.kd_attitude,
        )
        return jnp.concatenate([force_body, torque_body], axis=0)

    return policy_fn


def _build_union(cfg: ISSConfig, policy_cfg: PolicyConfig) -> tuple[PolicyFn, ExtrasFn]:
    weights_raw = jnp.asarray(policy_cfg.union_weights, dtype=jnp.float32)
    total = float(jnp.sum(weights_raw))
    if total <= 0.0:
        raise ValueError("union_weights must sum to > 0")
    weights = weights_raw / total

    random_fn = _build_random(cfg)
    orbit_fn, orbit_extras_fn = _build_orbit(cfg, policy_cfg.orbit)
    dock_fn = _build_dock(cfg, policy_cfg.dock)

    def extras_fn(key: jax.Array) -> jnp.ndarray:
        # Layout: [policy_idx, orbit_axis(3), orbit_radius, orbit_omega] = 6 floats.
        # The orbit slots are unused when policy_idx selects random or dock.
        key_idx, key_orbit = jax.random.split(key)
        policy_idx = jax.random.choice(key_idx, 3, p=weights).astype(jnp.float32)
        return jnp.concatenate([policy_idx[None], orbit_extras_fn(key_orbit)], axis=0)

    def policy_fn(state, key, extras):
        empty = jnp.zeros((0,), dtype=jnp.float32)
        return jax.lax.switch(
            extras[0].astype(jnp.int32),
            [
                lambda: random_fn(state, key, empty),
                lambda: orbit_fn(state, key, extras[1:6]),
                lambda: dock_fn(state, key, empty),
            ],
        )

    return policy_fn, extras_fn


def make_policy(
    cfg: ISSConfig,
    policy_cfg: PolicyConfig,
    policy_type: str | None = None,
) -> tuple[PolicyFn, ExtrasFn | None]:
    """Build a scripted policy. `policy_type` overrides `policy_cfg.type` when
    given (e.g. a CLI --policy flag); otherwise the config value is used."""
    if policy_type is None:
        policy_type = policy_cfg.type
    if policy_type == "random":
        return _build_random(cfg), None
    if policy_type == "orbit":
        return _build_orbit(cfg, policy_cfg.orbit)
    if policy_type == "dock":
        return _build_dock(cfg, policy_cfg.dock), None
    if policy_type == "union":
        return _build_union(cfg, policy_cfg)
    raise ValueError(
        f"Unknown ISS policy type '{policy_type}'. "
        "Must be one of: random, orbit, dock, union."
    )
