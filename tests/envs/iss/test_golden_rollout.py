"""Bit-identity pin for the iss env across the env-suite refactor.

The golden files were generated ONCE from commit ef78210 (the
pre-refactor state on main that this branch was rebased onto) and
committed. The tests then assert byte equality forever after.

There are two, covering disjoint code paths:

- ``golden_rollout.npz`` -- the default config: no sensor noise, bare 13D
  observation.
- ``golden_rollout_noisy.npz`` -- the "noncooperative" sensor-noise preset
  plus the goal-error augment (obs 13 -> 25), which exercises the sensing
  reassembly and observation-augment code the refactor rewrote and the
  nominal case never touches.

This test module did not exist on that pre-refactor main, so regeneration
cannot import anything from this file (or any other file in this
worktree) -- the recipes below are fully self-contained scripts using
pre-refactor imports (``owm_envs.envs.iss.policies`` rather than
``owm_envs.envs.common.policies``), run with PYTHONPATH pointed at an
archive of that commit's src/ tree so no refactored code is on the
import path. They archive the SHA rather than ``origin/main``: main is a
moving branch, and once it advances the recipe would silently rebuild
against different source and stop reproducing these files.

    mkdir -p /tmp/owm-golden-main
    git archive ef78210 src | tar -x -C /tmp/owm-golden-main
    PYTHONPATH=/tmp/owm-golden-main/src uv run python -c "
import owm_envs
assert '/tmp/owm-golden-main' in owm_envs.__file__, owm_envs.__file__

import numpy as np
from owm_envs.drivers.scan_driver import ScanDriver
from owm_envs.drivers.types import RolloutSpec
from owm_envs.envs.iss.policies import PolicyConfig
from owm_envs.envs.iss.config import ISSConfig

cfg = ISSConfig(max_steps=50)
driver = ScanDriver(cfg, PolicyConfig(type='union'), num_envs=4)
b = driver.generate(RolloutSpec(num_episodes=6, max_steps=50, seed=123))
np.savez('golden_rollout.npz', observations=b.observations, actions=b.actions,
         rewards=b.rewards, lengths=b.lengths, true_state=b.true_state)
"

and, for the noisy case (note that on pre-refactor main ObservationConfig
lives in ``owm_envs.envs.iss.config`` and PRESETS in
``owm_envs.envs.iss.sensing``, not their envs.common homes):

    mkdir -p /tmp/owm-golden-noisy
    git archive ef78210 src | tar -x -C /tmp/owm-golden-noisy
    PYTHONPATH=/tmp/owm-golden-noisy/src uv run python -c "
import owm_envs
assert '/tmp/owm-golden-noisy' in owm_envs.__file__, owm_envs.__file__

import numpy as np
from owm_envs.drivers.scan_driver import ScanDriver
from owm_envs.drivers.types import RolloutSpec
from owm_envs.envs.iss.policies import PolicyConfig
from owm_envs.envs.iss.config import ISSConfig, ObservationConfig
from owm_envs.envs.iss.sensing import PRESETS

cfg = ISSConfig(max_steps=50, sensor_noise=PRESETS['noncooperative'],
                observation=ObservationConfig(goal_error=True))
driver = ScanDriver(cfg, PolicyConfig(type='union'), num_envs=4)
b = driver.generate(RolloutSpec(num_episodes=6, max_steps=50, seed=123))
np.savez('golden_rollout_noisy.npz', observations=b.observations, actions=b.actions,
         rewards=b.rewards, lengths=b.lengths, true_state=b.true_state)
"

Then copy the resulting .npz files into this directory.
"""
from pathlib import Path

import numpy as np

from owm_envs.drivers.scan_driver import ScanDriver
from owm_envs.drivers.types import RolloutSpec
from owm_envs.envs.common.config import ObservationConfig
from owm_envs.envs.common.policies import PolicyConfig
from owm_envs.envs.common.sensing import PRESETS
from owm_envs.envs.iss.config import ISSConfig

GOLDEN = Path(__file__).parent / "golden_rollout.npz"
GOLDEN_NOISY = Path(__file__).parent / "golden_rollout_noisy.npz"


def _rollout(cfg: ISSConfig):
    policy_cfg = PolicyConfig(type="union")
    driver = ScanDriver(cfg, policy_cfg, num_envs=4)
    return driver.generate(RolloutSpec(num_episodes=6, max_steps=50, seed=123))


def _assert_matches(batch, golden_path: Path) -> None:
    g = np.load(golden_path)
    np.testing.assert_array_equal(batch.observations, g["observations"])
    np.testing.assert_array_equal(batch.actions, g["actions"])
    np.testing.assert_array_equal(batch.rewards, g["rewards"])
    np.testing.assert_array_equal(batch.lengths, g["lengths"])
    np.testing.assert_array_equal(batch.true_state, g["true_state"])


def test_iss_rollout_bit_identical_to_golden():
    assert GOLDEN.exists(), "generate golden_rollout.npz on main first (see module docstring)"
    _assert_matches(_rollout(ISSConfig(max_steps=50)), GOLDEN)


def test_iss_noisy_rollout_bit_identical_to_golden():
    assert GOLDEN_NOISY.exists(), (
        "generate golden_rollout_noisy.npz on main first (see module docstring)"
    )
    cfg = ISSConfig(
        max_steps=50,
        sensor_noise=PRESETS["noncooperative"],
        observation=ObservationConfig(goal_error=True),
    )
    _assert_matches(_rollout(cfg), GOLDEN_NOISY)
