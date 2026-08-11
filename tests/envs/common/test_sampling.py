import jax
import jax.numpy as jnp
import numpy as np

from owm_envs.envs.common.sampling import sample_small_rotation, sample_vector_in_ball


def test_ball_zero_radius_is_zero():
    v = sample_vector_in_ball(jax.random.PRNGKey(0), jnp.float32(0.0))
    np.testing.assert_array_equal(np.asarray(v), np.zeros(3))


def test_ball_stays_inside_and_fills_volume():
    keys = jax.random.split(jax.random.PRNGKey(1), 2000)
    vs = jax.vmap(lambda k: sample_vector_in_ball(k, jnp.float32(2.0)))(keys)
    norms = np.linalg.norm(np.asarray(vs), axis=1)
    assert norms.max() <= 2.0 + 1e-6
    # Uniform-by-volume: median norm at 2 * (1/2)^(1/3) ~= 1.587, not 1.0.
    assert abs(np.median(norms) - 2.0 * 0.5 ** (1 / 3)) < 0.05


def test_rotation_zero_angle_is_identity():
    q = sample_small_rotation(jax.random.PRNGKey(2), jnp.float32(0.0))
    np.testing.assert_allclose(np.asarray(q), [1.0, 0.0, 0.0, 0.0], atol=1e-7)


def test_rotation_angle_bounded():
    keys = jax.random.split(jax.random.PRNGKey(3), 500)
    qs = jax.vmap(lambda k: sample_small_rotation(k, jnp.float32(0.2)))(keys)
    angles = 2.0 * np.arccos(np.clip(np.abs(np.asarray(qs)[:, 0]), -1, 1))
    assert angles.max() <= 0.2 + 1e-5
