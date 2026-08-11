"""Quaternion helpers shared by every env, at astrojax's width.

Each helper below routes through astrojax's `Quaternion`, which casts to
astrojax's module-wide dtype -- float32 by default, and independent of this
package's own x64 flag. So a caller handing these an f64 quaternion gets an
f32 product back: bounded quantization at the 1e-7 level per call, which is
the right trade inside an integrator, where the quaternion is re-derived from
the kinematics every step and never summed into.

It is the wrong trade for a pair of conversions required to be exact
inverses of each other, where an f32 product floors the round trip at
~1.4e-7 rad against the ~3e-16 an f64 state carries. A caller in that
position composes at its own width locally rather than reaching here --
`envs/iss_numerical/dynamics._quat_compose` is the precedent, and its
docstring says why widening these instead would be a decision about all three
envs rather than about one view.
"""

from __future__ import annotations

import jax.numpy as jnp
from astrojax.attitude_representations.quaternion import Quaternion

# --------------------------------------------------------------------------------------
# Quaternion conventions
#
# - Quaternion q = [w, x, y, z] (astrojax scalar-first, matches this convention).
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
#
# astrojax returns the reference -> body direction-cosine matrix; this module
# needs body -> world, so `quat_to_rotmat` transposes it.


def quat_normalize(q: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Return q normalized to unit length."""
    return Quaternion.from_vector(q).normalize().to_vector()


def quat_conjugate(q: jnp.ndarray) -> jnp.ndarray:
    """Conjugate of quaternion [w,x,y,z] -> [w,-x,-y,-z]."""
    return Quaternion.from_vector(q).conjugate().to_vector()


def quat_multiply(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """
    Hamilton product q = q1 ⊗ q2.

    If q represents rotation, then:
      R(q1 ⊗ q2) = R(q1) @ R(q2)
    """
    return (Quaternion.from_vector(q1) * Quaternion.from_vector(q2)).to_vector()


def quat_to_rotmat(q: jnp.ndarray) -> jnp.ndarray:
    """
    Convert unit quaternion to 3x3 rotation matrix.

    Assumes q = [w,x,y,z] and maps body -> world if q is q_bw.
    """
    r = Quaternion.from_vector(q).to_rotation_matrix().to_matrix()
    return r.T


def quat_from_rotmat(r_bw: jnp.ndarray) -> jnp.ndarray:
    """Inverse of `quat_to_rotmat`: unit quaternion [w,x,y,z], w >= 0, from a
    body -> world matrix.

    Shepperd's method, branch-free: the four squared quaternion components
    are each computable straight from the diagonal of `r_bw` (they sum to
    4, so the largest is always >= 1), and dividing by the largest -- rather
    than always dividing by w -- keeps the divisor away from zero, including
    at the near-edge cases (w near 0, any axis-dominant rotation). Which
    component is largest is selected with `jnp.where` chains rather than a
    Python `if` or `lax.switch`/`cond` on a traced value, so this is
    jit/vmap-safe: every branch is always evaluated and simply discarded,
    never skipped. `q` and `-q` represent the same rotation, so the result
    is finally sign-flipped to resolve that double cover as w >= 0.
    """
    m00, m01, m02 = r_bw[0, 0], r_bw[0, 1], r_bw[0, 2]
    m10, m11, m12 = r_bw[1, 0], r_bw[1, 1], r_bw[1, 2]
    m20, m21, m22 = r_bw[2, 0], r_bw[2, 1], r_bw[2, 2]

    qw2 = 1.0 + m00 + m11 + m22
    qx2 = 1.0 + m00 - m11 - m22
    qy2 = 1.0 - m00 + m11 - m22
    qz2 = 1.0 - m00 - m11 + m22

    w_largest = (qw2 >= qx2) & (qw2 >= qy2) & (qw2 >= qz2)
    x_largest = (~w_largest) & (qx2 >= qy2) & (qx2 >= qz2)
    y_largest = (~w_largest) & (~x_largest) & (qy2 >= qz2)

    sw = jnp.sqrt(jnp.maximum(qw2, 0.0))
    sx = jnp.sqrt(jnp.maximum(qx2, 0.0))
    sy = jnp.sqrt(jnp.maximum(qy2, 0.0))
    sz = jnp.sqrt(jnp.maximum(qz2, 0.0))

    w_w, x_w, y_w, z_w = 0.5 * sw, (m21 - m12) / (2.0 * sw), (m02 - m20) / (2.0 * sw), (m10 - m01) / (2.0 * sw)
    w_x, x_x, y_x, z_x = (m21 - m12) / (2.0 * sx), 0.5 * sx, (m01 + m10) / (2.0 * sx), (m02 + m20) / (2.0 * sx)
    w_y, x_y, y_y, z_y = (m02 - m20) / (2.0 * sy), (m01 + m10) / (2.0 * sy), 0.5 * sy, (m12 + m21) / (2.0 * sy)
    w_z, x_z, y_z, z_z = (m10 - m01) / (2.0 * sz), (m02 + m20) / (2.0 * sz), (m12 + m21) / (2.0 * sz), 0.5 * sz

    w = jnp.where(w_largest, w_w, jnp.where(x_largest, w_x, jnp.where(y_largest, w_y, w_z)))
    x = jnp.where(w_largest, x_w, jnp.where(x_largest, x_x, jnp.where(y_largest, x_y, x_z)))
    y = jnp.where(w_largest, y_w, jnp.where(x_largest, y_x, jnp.where(y_largest, y_y, y_z)))
    z = jnp.where(w_largest, z_w, jnp.where(x_largest, z_x, jnp.where(y_largest, z_y, z_z)))

    q = jnp.stack([w, x, y, z]).astype(r_bw.dtype)
    return jnp.where(q[0] < 0.0, -q, q)


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


BODY_Z = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)


def axis_angle_from_quat(q: jnp.ndarray) -> jnp.ndarray:
    """Rotation vector (axis * angle, rad) of a quaternion, hemisphere-corrected.

    Safe at zero rotation: the axis degenerates but the magnitude -> 0, so
    the returned vector -> 0 rather than NaN.
    """
    q = jnp.where(q[0] < 0.0, -q, q)
    q = quat_normalize(q)
    sin_half = jnp.maximum(jnp.linalg.norm(q[1:4]), 1e-8)
    angle = 2.0 * jnp.arctan2(sin_half, q[0])
    return (q[1:4] / sin_half) * angle


def quat_from_body_z_to(target_dir: jnp.ndarray) -> jnp.ndarray:
    """Shortest-arc quaternion rotating body +z onto `target_dir` (assumed unit).

    The w = 1 + dot form degenerates when target_dir is exactly -z (both terms
    are zero); falls back to a 180-degree rotation about x, which is a valid
    shortest arc in that case. Traceable under jit/vmap: the fallback is
    selected with jnp.where rather than a Python `if` on the traced dot value.
    """
    dot = jnp.clip(jnp.dot(BODY_Z, target_dir), -1.0, 1.0)
    cross = jnp.cross(BODY_Z, target_dir)
    q = jnp.concatenate([jnp.array([1.0 + dot], dtype=jnp.float32), cross], axis=0)
    antiparallel = dot < -1.0 + 1e-6
    q = jnp.where(antiparallel, jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float32), q)
    return quat_normalize(q)


def quat_angle_between(q_a: jnp.ndarray, q_b: jnp.ndarray) -> jnp.ndarray:
    """Rotation angle in [0, pi] carrying `q_a` onto `q_b`.

    The magnitude of the attitude error, without the axis `axis_angle_from_quat`
    also returns -- which is what a scalar penalty term and a scalar success
    gate both want.

    abs() on the scalar part handles the q/-q double cover: q and -q are the
    same rotation, but their w components differ in sign and would give angles
    2*pi apart. It also folds the far half-turn onto the near one, so a 270 deg
    error reads as the 90 deg rotation it actually is, and bounds the result at
    pi.

    Neither input is normalised here. Every caller holds an attitude that its
    own dynamics keep unit-norm, and normalising would hide a state that had
    stopped being one.
    """
    q_err = quat_multiply(quat_conjugate(q_a), q_b)
    w = jnp.clip(jnp.abs(q_err[0]), -1.0, 1.0)
    return 2.0 * jnp.arccos(w)
