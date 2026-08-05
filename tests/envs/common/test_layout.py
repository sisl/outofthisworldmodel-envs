import jax.numpy as jnp
import numpy as np

from owm_envs.envs.common.layout import ISS_LAYOUT, VIEW_DIM, StateLayout


def test_iss_layout_slices_match_current_convention():
    assert ISS_LAYOUT.state_dim == 13
    assert ISS_LAYOUT.pos == slice(0, 3)
    assert ISS_LAYOUT.vel == slice(3, 6)
    assert ISS_LAYOUT.quat == slice(6, 10)
    assert ISS_LAYOUT.omega == slice(10, 13)
    assert ISS_LAYOUT.epoch is None
    assert ISS_LAYOUT.chief is None
    assert len(ISS_LAYOUT.labels) == 13


def test_iss_view_is_identity():
    state = jnp.arange(13.0)
    np.testing.assert_array_equal(np.asarray(ISS_LAYOUT.view(state)), np.arange(13.0))


def test_offset_view_slices_prefix():
    layout = StateLayout(
        state_dim=15,
        pos=slice(2, 5), vel=slice(5, 8), quat=slice(8, 12), omega=slice(12, 15),
        epoch=slice(0, 2),
        labels=("jd", "sec") + ISS_LAYOUT.labels,
    )
    state = jnp.arange(15.0)
    view = np.asarray(layout.view(state))
    assert view.shape == (VIEW_DIM,)
    np.testing.assert_array_equal(view, np.arange(2.0, 15.0))


def test_view_is_batched_friendly():
    state = jnp.arange(26.0).reshape(2, 13)
    out = np.asarray(ISS_LAYOUT.view(state))
    assert out.shape == (2, 13)
