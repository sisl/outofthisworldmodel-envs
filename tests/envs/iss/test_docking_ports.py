import jax.numpy as jnp
import numpy as np
import pytest

from owm_envs.envs.iss.config import ISSConfig, PhysicsConfig, load_collision_boxes
from owm_envs.envs.iss.docking_ports import (
    PORT_NAMES,
    PORTS,
    PORTS_BY_NAME,
    dock_targets,
    port_pose,
    resolve_port_names,
)
CFG = ISSConfig()
CHASER_RADIUS = PhysicsConfig().dragon_collision_radius_m


def clearance(position):
    centers, half_extents = load_collision_boxes(CFG.physics.collision_boxes_path)
    gap = np.maximum(np.abs(np.asarray(position) - centers) - half_extents, 0.0)
    return float(np.linalg.norm(gap, axis=1).min())


def test_every_port_goal_is_reachable_without_collision():
    # A goal the chaser cannot occupy is not a goal: the collision test expands
    # every hull box by the chaser radius, so a clearance under that radius
    # means the dock pose is inside the expanded hull and the episode ends on
    # contact the moment it arrives.
    for port in PORTS:
        position, _ = port_pose(port)
        assert clearance(position) > CHASER_RADIUS, port.name


def test_recorded_clearances_match_the_shipped_hull():
    for port in PORTS:
        position, _ = port_pose(port)
        assert clearance(position) == pytest.approx(port.clearance_m, abs=0.01), port.name


def test_pma2_pose_reproduces_the_shipped_dock_attitude():
    # The nadir up-hint convention is anchored on DockConfig: if this drifts,
    # every derived port attitude has silently changed convention too.
    _, quat = port_pose(PORTS_BY_NAME["harmony_fwd_pma2"])
    np.testing.assert_allclose(np.abs(quat), np.abs(np.asarray(CFG.dock.quaternion)), atol=1e-6)


def test_port_normals_are_unit_and_axis_aligned():
    for port in PORTS:
        normal = np.asarray(port.normal)
        assert np.linalg.norm(normal) == pytest.approx(1.0)
        assert sorted(np.abs(normal)) == [0.0, 0.0, 1.0]


def test_body_z_points_down_the_inbound_corridor():
    from owm_envs.core.quaternion import quat_to_rotmat

    for port in PORTS:
        position, quat = port_pose(port)
        body_z = np.asarray(quat_to_rotmat(jnp.asarray(quat, jnp.float32)))[:, 2]
        to_port = np.asarray(port.interface) - position
        np.testing.assert_allclose(body_z, to_port / np.linalg.norm(to_port), atol=1e-6)


def test_resolve_port_names_expands_all_and_rejects_unknown():
    assert resolve_port_names(("all",)) == PORT_NAMES
    assert resolve_port_names(("poisk_zenith",)) == ("poisk_zenith",)
    with pytest.raises(ValueError, match="unknown docking port"):
        resolve_port_names(("harmony_fwd_pma2", "not_a_port"))
    with pytest.raises(ValueError, match="empty"):
        resolve_port_names(())


def test_resolve_port_names_rejects_duplicates():
    # A repeated name would silently double that port's share of the uniform
    # per-episode draw, so it is an error rather than a deduplication.
    with pytest.raises(ValueError, match=r"duplicate docking port\(s\) \['poisk_zenith'\]"):
        resolve_port_names(("poisk_zenith", "harmony_fwd_pma2", "poisk_zenith"))


def test_dock_targets_rows_are_position_then_quaternion():
    table = dock_targets(("all",))
    assert table.shape == (len(PORT_NAMES), 7)
    for row, name in zip(table, PORT_NAMES):
        position, quat = port_pose(PORTS_BY_NAME[name])
        np.testing.assert_allclose(row[0:3], position, atol=1e-5)
        np.testing.assert_allclose(row[3:7], quat, atol=1e-5)
