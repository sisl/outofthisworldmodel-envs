import jax.numpy as jnp
import numpy as np
import pytest

from owm_envs.core.integrator import Integrator
from owm_envs.core.quaternion import (
    quat_conjugate,
    quat_derivative_from_omega_body,
    quat_multiply,
    quat_normalize,
    quat_to_rotmat,
    rotate_body_to_world,
)

IDENTITY = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)


def test_normalize_returns_unit_norm():
    q = jnp.array([2.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    assert np.isclose(float(jnp.linalg.norm(quat_normalize(q))), 1.0, atol=1e-6)


def test_multiply_by_identity_is_noop():
    q = quat_normalize(jnp.array([0.5, 0.5, 0.5, 0.5], dtype=jnp.float32))
    np.testing.assert_allclose(np.asarray(quat_multiply(q, IDENTITY)), np.asarray(q), atol=1e-6)


def test_multiply_by_conjugate_gives_identity():
    q = quat_normalize(jnp.array([0.1, -0.4, 0.7, 0.2], dtype=jnp.float32))
    got = quat_multiply(q, quat_conjugate(q))
    np.testing.assert_allclose(np.asarray(got), np.asarray(IDENTITY), atol=1e-6)


def test_identity_rotmat_is_eye():
    np.testing.assert_allclose(np.asarray(quat_to_rotmat(IDENTITY)), np.eye(3), atol=1e-6)


def test_rotmat_is_orthonormal():
    q = quat_normalize(jnp.array([0.3, 0.1, -0.5, 0.8], dtype=jnp.float32))
    r = np.asarray(quat_to_rotmat(q))
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(r), 1.0, atol=1e-5)


def test_90deg_rotation_about_z_maps_x_to_y():
    # q = [cos(45deg), 0, 0, sin(45deg)] rotates +x onto +y
    s = float(np.sin(np.pi / 4))
    q = jnp.array([float(np.cos(np.pi / 4)), 0.0, 0.0, s], dtype=jnp.float32)
    v = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    got = rotate_body_to_world(q, v)
    np.testing.assert_allclose(np.asarray(got), np.array([0.0, 1.0, 0.0]), atol=1e-6)


def test_quat_derivative_is_orthogonal_to_quat():
    # d/dt of a unit quaternion must stay tangent to the unit sphere: <q, q_dot> == 0
    q = quat_normalize(jnp.array([0.2, 0.3, -0.1, 0.9], dtype=jnp.float32))
    omega = jnp.array([0.3, -0.2, 0.5], dtype=jnp.float32)
    q_dot = quat_derivative_from_omega_body(q, omega)
    assert abs(float(jnp.dot(q, q_dot))) < 1e-6


def test_zero_omega_gives_zero_derivative():
    q = quat_normalize(jnp.array([0.2, 0.3, -0.1, 0.9], dtype=jnp.float32))
    q_dot = quat_derivative_from_omega_body(q, jnp.zeros(3, dtype=jnp.float32))
    np.testing.assert_allclose(np.asarray(q_dot), np.zeros(4), atol=1e-7)


@pytest.mark.parametrize(
    "axis_index, omega_body",
    [
        (1, (1.5, 0.0, 0.0)),  # rotation about body x -> quaternion x component
        (3, (0.0, 0.0, 1.5)),  # rotation about body z -> quaternion z component
    ],
)
def test_single_axis_rotation_matches_analytic_solution(axis_index, omega_body):
    # Neither test_quat_derivative_is_orthogonal_to_quat (only requires
    # skew-symmetry) nor test_zero_omega_gives_zero_derivative (omega=0 makes
    # the Omega matrix all-zero) would catch a p/q/r axis permutation in
    # omega_matrix_body. Integrating a known single-axis rotation and checking
    # against the closed-form solution pins the axis mapping itself.
    omega = jnp.array(omega_body, dtype=jnp.float32)
    Omega = float(jnp.linalg.norm(omega))
    dt = 0.01
    steps = 50
    t = dt * steps

    def f(q, _):
        return quat_derivative_from_omega_body(q, omega)

    integrator = Integrator(dt)
    q = IDENTITY
    dummy_u = jnp.zeros(1, dtype=jnp.float32)
    for _ in range(steps):
        q = integrator.rk4(f, q, dummy_u)

    expected = np.zeros(4, dtype=np.float32)
    expected[0] = np.cos(Omega * t / 2)
    expected[axis_index] = np.sin(Omega * t / 2)
    np.testing.assert_allclose(np.asarray(q), expected, atol=1e-5)
