"""Pinned docking-port entries: normalisation, verification, and what they buy.

A config records ports so a run can be reproduced. A bare name is only a
reference into the PORTS table, so it stops meaning the same thing the moment
the table is revised -- these tests pin down that a name resolves to a pose at
load, that the pose is what gets serialised, and that a stale pinned pose is a
loud error rather than a silent disagreement.
"""

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from owm_envs.envs.iss.config import ISSConfig
from owm_envs.envs.iss.docking_ports import PORT_NAMES, PORTS_BY_NAME, port_pose
from owm_envs.envs.iss.policies import (
    DockParams,
    DockPort,
    PolicyConfig,
    dock_target_table,
)

CFG = ISSConfig()


def pinned(name: str) -> dict:
    position, quaternion = port_pose(PORTS_BY_NAME[name])
    return {
        "name": name,
        "position": [float(v) for v in position],
        "quaternion": [float(v) for v in quaternion],
    }


def test_bare_names_normalise_to_pinned_entries():
    params = DockParams(ports=("poisk_zenith", "zvezda_aft"))
    assert [p.name for p in params.ports] == ["poisk_zenith", "zvezda_aft"]
    for port in params.ports:
        assert isinstance(port, DockPort)
        position, quaternion = port_pose(PORTS_BY_NAME[port.name])
        np.testing.assert_allclose(port.position, position, atol=1e-6)
        np.testing.assert_allclose(port.quaternion, quaternion, atol=1e-6)


def test_all_still_expands_to_every_port():
    params = DockParams(ports=("all",))
    assert tuple(p.name for p in params.ports) == PORT_NAMES


def test_all_cannot_be_mixed_with_other_entries():
    with pytest.raises(ValidationError, match="cannot be mixed"):
        DockParams(ports=("all", pinned("zvezda_aft")))


def test_serialised_config_carries_the_poses_it_used():
    # The reproducibility property: the as-run record is not a list of names
    # whose meaning depends on which version of the table reads it back.
    text = PolicyConfig(type="dock", dock=DockParams(ports=("poisk_zenith",))).to_yaml()
    entries = yaml.safe_load(text)["dock"]["ports"]
    position, quaternion = port_pose(PORTS_BY_NAME["poisk_zenith"])
    assert [e["name"] for e in entries] == ["poisk_zenith"]
    np.testing.assert_allclose(entries[0]["position"], position, atol=1e-6)
    np.testing.assert_allclose(entries[0]["quaternion"], quaternion, atol=1e-6)


def test_pinned_entries_round_trip_through_yaml():
    original = PolicyConfig(type="dock", dock=DockParams(ports=("all",)))
    reloaded = PolicyConfig.model_validate_json(original.model_dump_json())
    assert reloaded == original


def test_pinned_pose_matching_the_table_loads():
    params = DockParams(ports=(pinned("rassvet_nadir"),))
    assert [p.name for p in params.ports] == ["rassvet_nadir"]


def test_pinned_pose_disagreeing_with_the_table_is_a_load_error():
    stale = pinned("harmony_nadir_cbm")
    stale["position"] = [stale["position"][0] + 0.5, *stale["position"][1:]]
    with pytest.raises(ValidationError, match="harmony_nadir_cbm"):
        DockParams(ports=(stale,))


def test_pinned_quaternion_disagreeing_with_the_table_is_a_load_error():
    stale = pinned("zvezda_aft")
    stale["quaternion"] = [0.0, 0.0, 0.0, 1.0]
    with pytest.raises(ValidationError, match="pinned quaternion .* 'zvezda_aft'"):
        DockParams(ports=(stale,))


def test_a_pinned_port_the_table_no_longer_knows_still_loads_and_is_flown_to():
    # The escape hatch: a config that outlives a table revision keeps its own
    # pose rather than failing to load or silently retargeting.
    retired = {
        "name": "retired_berth",
        "position": [11.0, -22.0, 33.0],
        "quaternion": [1.0, 0.0, 0.0, 0.0],
    }
    params = DockParams(ports=(retired,))
    table = np.asarray(dock_target_table(CFG, params))
    assert table.shape == (1, 7)
    np.testing.assert_allclose(table[0], [11.0, -22.0, 33.0, 1.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_mixed_bare_and_pinned_entries_keep_config_order():
    params = DockParams(ports=("zvezda_aft", pinned("poisk_zenith")))
    assert [p.name for p in params.ports] == ["zvezda_aft", "poisk_zenith"]
    table = np.asarray(dock_target_table(CFG, params))
    for row, name in zip(table, ("zvezda_aft", "poisk_zenith")):
        position, quaternion = port_pose(PORTS_BY_NAME[name])
        np.testing.assert_allclose(row, [*position, *quaternion], atol=1e-5)


def test_duplicates_are_rejected_across_bare_and_pinned_forms():
    with pytest.raises(ValidationError, match="duplicate docking port"):
        DockParams(ports=("poisk_zenith", pinned("poisk_zenith")))
