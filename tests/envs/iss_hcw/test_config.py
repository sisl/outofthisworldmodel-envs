from pathlib import Path

import jax.numpy as jnp
import numpy as np

from owm_envs.envs.common.config import PhysicsConfig
from owm_envs.envs.iss_hcw.config import HCW_LAYOUT, HCWConfig

CONFIGS = Path(__file__).resolve().parents[3] / "configs"


def test_the_shipped_config_disperses_through_the_orbit_section():
    # The committed config is what an iss-hcw dataset is generated from, and
    # its start dispersions have to live in `orbit`: `HCWDynamics.reset`
    # never reads the `physics` start radius, so the copy of that field the
    # file inherited from iss_default.toml set nothing at all.
    cfg = HCWConfig.from_toml(CONFIGS / "iss_hcw_default.toml")
    assert cfg.orbit.start_radius_range_m == (80.0, 120.0)
    assert cfg.physics.start_radius_range_m == PhysicsConfig().start_radius_range_m


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
