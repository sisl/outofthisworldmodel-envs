"""Shared reset-dispersion samplers."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def sample_vector_in_ball(key: jax.Array, max_norm: jnp.ndarray) -> jnp.ndarray:
    """Random 3-vector, uniform by volume within a ball of radius `max_norm`
    (the zero vector when `max_norm` is 0). A scalar draw returning (3,);
    vmap for batches. Cube-root the radial fraction so
    the distribution is uniform over the ball's volume rather than biased
    toward the origin, as a naive `direction * max_norm * uniform(0,1)` would
    be."""
    key_dir, key_frac = jax.random.split(key)
    raw = jax.random.normal(key_dir, (3,), dtype=jnp.float32)
    direction = raw / jnp.maximum(jnp.linalg.norm(raw), 1e-8)
    frac = jnp.cbrt(jax.random.uniform(key_frac, (), dtype=jnp.float32))
    return direction * (max_norm * frac)


def sample_small_rotation(key: jax.Array, max_angle_rad: jnp.ndarray) -> jnp.ndarray:
    """Random unit quaternion: a rotation by an angle uniform in
    `[0, max_angle_rad]` about a uniformly random axis (the identity
    quaternion when `max_angle_rad` is 0). A scalar draw returning (4,);
    vmap for batches."""
    key_axis, key_angle = jax.random.split(key)
    raw = jax.random.normal(key_axis, (3,), dtype=jnp.float32)
    axis = raw / jnp.maximum(jnp.linalg.norm(raw), 1e-8)
    angle = jax.random.uniform(key_angle, (), dtype=jnp.float32, minval=0.0, maxval=max_angle_rad)
    return jnp.concatenate([jnp.cos(angle / 2.0)[None], jnp.sin(angle / 2.0) * axis], axis=0)
