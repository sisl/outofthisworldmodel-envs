import jax.numpy as jnp
import numpy as np

from owm_envs.envs.iss_hcw.config import HCW_LAYOUT, HCWConfig


def test_config_roundtrips_through_toml(tmp_path):
    original = HCWConfig(dt=0.02, max_steps=500)
    path = tmp_path / "run_config.toml"
    original.to_toml(path)
    assert HCWConfig.from_toml(path) == original


def test_orbit_defaults_to_iss_like_sma():
    assert HCWConfig().orbit.sma_m == 6_795_000.0


def test_hcw_layout_state_dim_is_15():
    assert HCW_LAYOUT.state_dim == 15


def test_hcw_layout_slice_view_drops_the_epoch_prefix():
    view = np.asarray(HCW_LAYOUT.slice_view(jnp.arange(15.0)))
    np.testing.assert_array_equal(view, np.arange(2.0, 15.0))
