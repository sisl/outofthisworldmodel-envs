import pytest
from pydantic import ValidationError

from owm_envs.envs.iss_numerical.config import (
    NUM_LAYOUT,
    OBS_MODE_DIM,
    BallisticConfig,
    NumericalConfig,
    PerturbationsConfig,
)


def test_config_roundtrips_through_toml(tmp_path):
    original = NumericalConfig(dt=0.02, max_steps=500)
    path = tmp_path / "run_config.toml"
    original.to_toml(path)
    assert NumericalConfig.from_toml(path) == original


def test_observation_mode_defaults_to_relative():
    assert NumericalConfig().observation.mode == "relative"


def test_orbit_defaults_to_iss_like_sma():
    assert NumericalConfig().orbit.sma_m == 6_795_000.0


@pytest.mark.parametrize("degree", [1, 7, -1])
def test_zonal_max_degree_rejects_unsupported_values(degree):
    with pytest.raises(ValidationError):
        PerturbationsConfig(zonal_max_degree=degree)


@pytest.mark.parametrize("degree", [0, 2, 3, 4, 5, 6])
def test_zonal_max_degree_accepts_supported_values(degree):
    assert PerturbationsConfig(zonal_max_degree=degree).zonal_max_degree == degree


def test_ballistic_config_rejects_non_positive_area():
    with pytest.raises(ValidationError):
        BallisticConfig(area_m2=0.0)


def test_num_layout_state_dim_is_21():
    assert NUM_LAYOUT.state_dim == 21


def test_num_layout_slices_are_contiguous_and_sized():
    assert NUM_LAYOUT.epoch == slice(0, 2)
    assert NUM_LAYOUT.chief == slice(2, 8)
    assert NUM_LAYOUT.pos == slice(8, 11)
    assert NUM_LAYOUT.vel == slice(11, 14)
    assert NUM_LAYOUT.quat == slice(14, 18)
    assert NUM_LAYOUT.omega == slice(18, 21)
    assert len(NUM_LAYOUT.labels) == 21


def test_obs_mode_dim_covers_every_mode():
    assert OBS_MODE_DIM == {
        "absolute": 21,
        "chaser_absolute": 21,
        "chief_absolute": 21,
        "relative": 15,
    }
