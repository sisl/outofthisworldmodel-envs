"""The shipped docking recipe, configs/iss/gen/default.yaml.

This file is maintained alongside the PORTS table -- the pinned-pose escape
hatch for an unknown port name exists for users' own configs, not for this
one. So these tests assert it still loads AND still agrees with the table:
a port whose pose moves must break here, in the repo, rather than silently
in someone's run.
"""

from pathlib import Path

import numpy as np
import pytest

from owm_envs.datasets.stats import GenerationConfig
from owm_envs.envs.common.docking_ports import PORT_NAMES, PORTS_BY_NAME, port_pose

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "iss" / "gen" / "default.yaml"

# Approaches held out of training so validation measures them: the two
# zenith-facing corridors, and the Unity nadir berth ISS_base.glb freed.
HELD_OUT = ("harmony_zenith_cbm", "poisk_zenith", "unity_nadir_cbm")
TRAIN_PORTS = tuple(name for name in PORT_NAMES if name not in HELD_OUT)


@pytest.fixture(scope="module")
def gen() -> GenerationConfig:
    return GenerationConfig.from_yaml(CONFIG_PATH)


def test_the_shipped_config_loads(gen):
    assert set(gen.splits) == {"train", "val"}


def test_sizes_and_seeds_mirror_the_built_in_defaults(gen):
    assert (gen.splits["train"].num_episodes, gen.splits["train"].seed) == (64, 0)
    assert (gen.splits["val"].num_episodes, gen.splits["val"].seed) == (8, 1)


def test_training_holds_the_zenith_approaches_out(gen):
    train = gen.splits["train"].policy
    assert train.type == "union"
    assert tuple(p.name for p in train.dock.ports) == TRAIN_PORTS
    assert not set(HELD_OUT) & {p.name for p in train.dock.ports}


def test_validation_covers_every_port_including_the_held_out_ones(gen):
    val = gen.splits["val"].policy
    assert val.type == "dock"
    assert tuple(p.name for p in val.dock.ports) == PORT_NAMES
    assert set(HELD_OUT) <= {p.name for p in val.dock.ports}


def test_every_pinned_pose_still_matches_the_current_table(gen):
    # Loading already rejects a mismatch, so reaching this assertion means the
    # poses agree -- but assert it directly, so the failure names the port and
    # this test does not silently become a duplicate of test_the_shipped_config_loads.
    for split in gen.splits.values():
        for port in split.policy.dock.ports:
            position, quaternion = port_pose(PORTS_BY_NAME[port.name])
            np.testing.assert_allclose(port.position, position, atol=1e-5, err_msg=port.name)
            np.testing.assert_allclose(port.quaternion, quaternion, atol=1e-5, err_msg=port.name)


def test_the_shipped_config_is_written_in_pinned_form():
    # Not just "it parses": the point of shipping it pinned is that the file
    # itself carries the poses, so a name-only file would defeat the exercise.
    text = CONFIG_PATH.read_text()
    assert "position:" in text and "quaternion:" in text
