"""Bit-identity pin for the iss env.

The tests assert a current rollout is byte-equal to two stored ``.npz``
fixtures, covering disjoint code paths:

- ``golden_rollout.npz`` -- the default config: no sensor noise, bare 13D
  observation.
- ``golden_rollout_noisy.npz`` -- the "noncooperative" sensor-noise preset
  plus the goal-error augment (obs 13 -> 25), which exercises the sensing
  reassembly and observation-augment code the nominal case never touches.

Provenance. The files were first generated from commit ef78210, the
pre-refactor state the env-suite refactor was measured against, and have
been re-pinned once since. Two separate commits matter to that re-pin and
they are not the same one:

- What INVALIDATED the previous files: the docking-reward rewrite, landed
  across 07a8ec2 (the weights and ``RewardShapingConfig`` the reward
  reads) and 94ac4ed (``docking_reward`` itself, from a sum of squared
  errors onto normalised Huber terms). That rewrite changed the reward's
  scale by six orders of magnitude -- the first step of episode 0 went
  from -504320.75 to -0.9927034, and the whole fixture from
  ``|r| <= 792241`` to ``|r| <= 1.14141``.
- What PRODUCED the bytes now in these files: the working tree at
  772e827, which was HEAD when they were regenerated. That commit is a
  config revert and contains no reward change of its own; it is named
  here only to identify the source tree.

Only ``rewards`` changed at that re-pin. ``observations``, ``actions``,
``lengths`` and ``true_state`` all carried over BIT-IDENTICALLY from the
ef78210 files, which is what makes the re-pin safe rather than a
trajectory being quietly rewritten underneath a stale expectation: no
control law reads the reward, so the trajectory, the actions and the
underlying dynamics state are independent of it, and here they measurably
were. Any future re-pin should establish the same property first --
compare the four non-reward arrays against the committed files BEFORE
overwriting them, because overwriting destroys the evidence.

Regenerating. Run against the CURRENT working tree, from the repository
root. Do not replay an archived commit's ``src`` -- an older tree would
reproduce that tree's reward, which is the thing a re-pin exists to move
away from. Then update the provenance above with BOTH: the new
``git rev-parse HEAD`` as the tree that produced the bytes, and,
separately, the commit whose behaviour change made the re-pin necessary.

    uv run python -c "
import numpy as np
from owm_envs.drivers.scan_driver import ScanDriver
from owm_envs.drivers.types import RolloutSpec
from owm_envs.envs.common.policies import PolicyConfig
from owm_envs.envs.iss.config import ISSConfig

cfg = ISSConfig(max_steps=50)
driver = ScanDriver(cfg, PolicyConfig(type='union'), num_envs=4)
b = driver.generate(RolloutSpec(num_episodes=6, max_steps=50, seed=123))
np.savez('tests/envs/iss/golden_rollout.npz', observations=b.observations, actions=b.actions,
         rewards=b.rewards, lengths=b.lengths, true_state=b.true_state)
"

and, for the noisy case:

    uv run python -c "
import numpy as np
from owm_envs.drivers.scan_driver import ScanDriver
from owm_envs.drivers.types import RolloutSpec
from owm_envs.envs.common.config import ObservationConfig
from owm_envs.envs.common.policies import PolicyConfig
from owm_envs.envs.common.sensing import PRESETS
from owm_envs.envs.iss.config import ISSConfig

cfg = ISSConfig(max_steps=50, sensor_noise=PRESETS['noncooperative'],
                observation=ObservationConfig(goal_error=True))
driver = ScanDriver(cfg, PolicyConfig(type='union'), num_envs=4)
b = driver.generate(RolloutSpec(num_episodes=6, max_steps=50, seed=123))
np.savez('tests/envs/iss/golden_rollout_noisy.npz', observations=b.observations, actions=b.actions,
         rewards=b.rewards, lengths=b.lengths, true_state=b.true_state)
"

Both mirror ``_rollout`` and the two tests below exactly, but are written
standalone so a regeneration never imports the module whose expectations
it is about to rewrite.
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
    assert GOLDEN.exists(), "generate golden_rollout.npz first (see module docstring)"
    _assert_matches(_rollout(ISSConfig(max_steps=50)), GOLDEN)


def test_iss_noisy_rollout_bit_identical_to_golden():
    assert GOLDEN_NOISY.exists(), (
        "generate golden_rollout_noisy.npz first (see module docstring)"
    )
    cfg = ISSConfig(
        max_steps=50,
        sensor_noise=PRESETS["noncooperative"],
        observation=ObservationConfig(goal_error=True),
    )
    _assert_matches(_rollout(cfg), GOLDEN_NOISY)
