"""Run-directory env resolution and view extraction behind scripts/run_view.py.

The scripts directory is not a package; the module is imported by path so the
resolution both diagnostics scripts trust is covered by the suite rather than
only by running them against a real run.
"""
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from astrojax.constants import GM_EARTH

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_view import observed_views, run_env, state_views  # noqa: E402

from owm_envs.envs import ENV_REGISTRY  # noqa: E402
from owm_envs.envs.iss.config import ISSConfig  # noqa: E402
from owm_envs.envs.iss_numerical.config import NumericalConfig  # noqa: E402
from owm_envs.envs.iss_numerical.dynamics import chaser_state_from_view  # noqa: E402


def _run_dir(tmp_path: Path, env: str | None = None, cfg=None) -> Path:
    if env is not None:
        (tmp_path / "dataset_card.json").write_text(json.dumps({"env": env}))
    (cfg if cfg is not None else ISSConfig()).to_yaml(tmp_path / "env_config.yaml")
    return tmp_path


def _numerical_state(view: np.ndarray) -> np.ndarray:
    """A valid 21D iss-numerical state whose relative view is `view`."""
    sma = 6.795e6
    chief = jnp.asarray([sma, 0.0, 0.0, 0.0, float(np.sqrt(GM_EARTH / sma)), 0.0],
                        jnp.float64)
    epoch = jnp.asarray([2460000.5, 0.0], jnp.float64)
    chaser = chaser_state_from_view(chief, jnp.asarray(view, jnp.float64))
    return np.asarray(jnp.concatenate([epoch, chief, chaser]))


def test_run_env_resolves_the_env_the_card_records(tmp_path):
    run = _run_dir(tmp_path, env="iss-numerical", cfg=NumericalConfig())
    spec, cfg = run_env(run)
    assert spec.name == "iss-numerical"
    assert isinstance(cfg, NumericalConfig)


def test_run_env_reads_a_run_with_no_card_as_iss(tmp_path):
    # Runs from before the card recorded an env at all are iss runs: nothing
    # else existed when they were written.
    spec, cfg = run_env(_run_dir(tmp_path))
    assert spec.name == "iss"
    assert isinstance(cfg, ISSConfig)


def test_run_env_refuses_an_env_this_build_does_not_register(tmp_path):
    # Unlike publishing, which falls back to iss to avoid losing data, a
    # diagnostic read through the wrong env's view would return wrong numbers
    # with a confident face.
    run = _run_dir(tmp_path, env="iss-warp")
    with pytest.raises(SystemExit, match="iss-warp"):
        run_env(run)


def test_observed_views_are_the_first_13_dims_when_the_state_is_recorded():
    # iss records its state directly; anything past dim 13 is the goal block.
    obs = np.arange(50.0).reshape(2, 25)
    views = observed_views(ENV_REGISTRY["iss"], ISSConfig(), obs)
    np.testing.assert_array_equal(views, obs[:, :13])


def test_observed_views_strip_the_epoch_prefix_under_relative_mode():
    obs = np.arange(54.0).reshape(2, 27)
    views = observed_views(ENV_REGISTRY["iss-numerical"], NumericalConfig(), obs)
    np.testing.assert_array_equal(views, obs[:, 2:15])


def test_observed_views_refuse_a_mode_that_does_not_record_the_view():
    # The absolute modes store ECI magnitudes at float32, whose ~0.5 m grain
    # would swamp a metre-level relative residual; there is no honest way to
    # read the measured view back out of them.
    cfg = NumericalConfig(observation={"mode": "absolute"})
    with pytest.raises(SystemExit, match="absolute"):
        observed_views(ENV_REGISTRY["iss-numerical"], cfg, np.zeros((1, 21)))


def test_state_views_are_the_state_itself_for_iss():
    states = np.arange(26.0).reshape(2, 13)
    np.testing.assert_array_equal(state_views(ENV_REGISTRY["iss"], states), states)


def test_state_views_recover_the_relative_view_of_a_numerical_state():
    view = np.array([10.0, -24.0, 5.0, 0.1, -0.05, 0.02,
                     1.0, 0.0, 0.0, 0.0, 0.001, -0.002, 0.0005])
    states = _numerical_state(view)[None, :]
    out = state_views(ENV_REGISTRY["iss-numerical"], states)
    np.testing.assert_allclose(out[0], view, atol=1e-4)
