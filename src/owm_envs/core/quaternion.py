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
# These are thin array-level wrappers rather than direct astrojax use at the call
# sites: the dynamics and policies operate on a flat 13-element state vector and
# slice `state[6:10]`, so keeping the object boundary here confines the dependency
# to one module and leaves the RK4 integrand a pure array-to-array function.
#
# Note: astrojax's `Quaternion.to_rotation_matrix()` returns the transpose of this
# module's q_bw convention (its multiplication and conjugate match ours exactly,
# but its rotation-matrix conversion does not). `quat_to_rotmat` transposes it back
# so callers see the same body -> world matrix as before the swap.
#
# `omega_matrix_body` and `quat_derivative_from_omega_body` are implemented locally
# because astrojax provides no attitude kinematics -- there is no equivalent of
# q_dot = 0.5 * Omega(omega) * q. That is the one function the integrator calls
# every step.
# --------------------------------------------------------------------------------------


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
