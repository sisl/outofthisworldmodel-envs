import jax.numpy as jnp
import numpy as np
import pytest

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


def test_iss_slice_view_is_identity():
    state = jnp.arange(13.0)
    np.testing.assert_array_equal(np.asarray(ISS_LAYOUT.slice_view(state)), np.arange(13.0))


def test_offset_slice_view_slices_prefix():
    layout = StateLayout(
        state_dim=15,
        pos=slice(2, 5), vel=slice(5, 8), quat=slice(8, 12), omega=slice(12, 15),
        epoch=slice(0, 2),
        labels=("jd", "sec") + ISS_LAYOUT.labels,
    )
    state = jnp.arange(15.0)
    view = np.asarray(layout.slice_view(state))
    assert view.shape == (VIEW_DIM,)
    np.testing.assert_array_equal(view, np.arange(2.0, 15.0))


def test_slice_view_is_batched_friendly():
    state = jnp.arange(26.0).reshape(2, 13)
    out = np.asarray(ISS_LAYOUT.slice_view(state))
    assert out.shape == (2, 13)


def test_label_count_must_match_state_dim():
    with pytest.raises(ValueError, match="labels has 13 entries for state_dim 15"):
        StateLayout(
            state_dim=15,
            pos=slice(2, 5), vel=slice(5, 8), quat=slice(8, 12), omega=slice(12, 15),
            epoch=slice(0, 2),
        )


def test_non_contiguous_slices_are_rejected():
    with pytest.raises(ValueError, match="must be contiguous"):
        StateLayout(
            state_dim=15,
            pos=slice(0, 3), vel=slice(3, 6), quat=slice(8, 12), omega=slice(12, 15),
            labels=ISS_LAYOUT.labels + ("pad_a", "pad_b"),
        )


def test_wrong_pos_width_is_rejected():
    with pytest.raises(ValueError, match="pos must have width 3"):
        StateLayout(
            state_dim=12,
            pos=slice(0, 2), vel=slice(2, 5), quat=slice(5, 9), omega=slice(9, 12),
            labels=ISS_LAYOUT.labels[:-1],
        )


def test_wrong_quat_width_is_rejected():
    with pytest.raises(ValueError, match="quat must have width 4"):
        StateLayout(
            state_dim=12,
            pos=slice(0, 3), vel=slice(3, 6), quat=slice(6, 9), omega=slice(9, 12),
            labels=ISS_LAYOUT.labels[:-1],
        )
