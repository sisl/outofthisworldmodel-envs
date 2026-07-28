from __future__ import annotations

import jax.numpy as jnp


# --------------------------------------------------------------------------------------
# Quaternion conventions
#
# - Quaternion q = [w, x, y, z]
# - Unit quaternion represents a rotation.
#
# q_bw maps BODY vectors into WORLD vectors (body -> world).
#
# So to rotate a vector v_body into world:
#   v_world = R(q_bw) @ v_body
#
# Angular velocity omega is stored in BODY coordinates (rad/s).
# Quaternion kinematics:
#   q_dot = 0.5 * Omega(omega_body) @ q
#
# where Omega(omega) is a 4x4 matrix defined below.
# --------------------------------------------------------------------------------------


def quat_normalize(q: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Return q normalized to unit length."""
    n = jnp.linalg.norm(q)
    return q / jnp.maximum(n, eps)


def quat_conjugate(q: jnp.ndarray) -> jnp.ndarray:
    """Conjugate of quaternion [w,x,y,z] -> [w,-x,-y,-z]."""
    return jnp.array([q[0], -q[1], -q[2], -q[3]], dtype=q.dtype)


def quat_multiply(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """
    Hamilton product q = q1 ⊗ q2.

    If q represents rotation, then:
      R(q1 ⊗ q2) = R(q1) @ R(q2)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=q1.dtype,
    )


def quat_to_rotmat(q: jnp.ndarray) -> jnp.ndarray:
    """
    Convert unit quaternion to 3x3 rotation matrix.

    Assumes q = [w,x,y,z] and maps body -> world if q is q_bw.
    """
    q = quat_normalize(q)
    w, x, y, z = q

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    return jnp.array(
        [
            [ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=q.dtype,
    )


def rotate_body_to_world(q_bw: jnp.ndarray, v_body: jnp.ndarray) -> jnp.ndarray:
    """Rotate a 3-vector from body frame to world frame using q_bw."""
    R_bw = quat_to_rotmat(q_bw)
    return R_bw @ v_body


def omega_matrix_body(omega_body: jnp.ndarray) -> jnp.ndarray:
    """
    4x4 matrix Omega(omega) for quaternion kinematics:
      q_dot = 0.5 * Omega(omega_body) @ q

    omega_body = [p,q,r] = roll/pitch/yaw rates expressed in BODY frame.
    """
    p, q, r = omega_body
    return jnp.array(
        [
            [0.0, -p, -q, -r],
            [p, 0.0, r, -q],
            [q, -r, 0.0, p],
            [r, q, -p, 0.0],
        ],
        dtype=omega_body.dtype,
    )


def quat_derivative_from_omega_body(q_bw: jnp.ndarray, omega_body: jnp.ndarray) -> jnp.ndarray:
    """Compute q_dot given current q_bw and body-frame omega."""
    return 0.5 * (omega_matrix_body(omega_body) @ q_bw)
