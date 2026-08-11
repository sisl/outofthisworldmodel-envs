"""The variant collision hulls must block the port their asset occupies.

Three render assets berth a visiting vehicle at a station port, and the
shipped 313-box hull describes the bare station: flown against it, a chaser
passes through the vehicle the renderer draws and reports a clean dock. Each
committed collision_boxes_<variant>.yaml is that hull plus boxes covering the
vehicle, written by scripts/write_variant_collision_boxes.py.

Five things are checked, and only the first is a drift guard. That the
committed hulls still equal what the writer produces; that each variant
config names its own asset beside its own hull, since a config that draws a
Dragon while colliding against a Cygnus is the failure this pairing exists to
prevent; that the occupied port is now genuinely blocked, which is the whole
reason the hulls exist; that no part of the berthed vehicle's surface is out
of reach of the hull, since a chaser that flies through an uncovered solar
array registers nothing and the goal-pose check above sees only the one pose;
and that the other seven ports keep exactly their shipped clearances, which is
what confines a hull to its vehicle. The last would fail if the writer's
cluster selection had pulled in re-tessellated station structure alongside the
vehicle -- geometry the shipped hull already approximates, whose inclusion
would silently change collision behaviour at ports that berth nothing.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from owm_envs.envs.common.config import load_collision_boxes
from owm_envs.envs.common.docking_ports import PORTS, PORTS_BY_NAME, port_pose
from owm_envs.envs.common.events import EventChecker
from owm_envs.envs.iss_numerical.config import NumericalConfig

REPO = Path(__file__).resolve().parents[1]
NUMERICAL = REPO / "configs" / "iss-numerical"
RESOURCES = REPO / "src" / "owm_envs" / "envs" / "common" / "resources"

# The writers are scripts under scripts/, not modules of the installed
# package, so this guard has to reach them by path to compare the committed
# files against the very code that produced them.
sys.path.insert(0, str(REPO / "scripts"))

import write_variant_collision_boxes as wvcb  # noqa: E402

VARIANTS = tuple(wvcb.OCCUPIED_PORT)


@pytest.fixture(scope="module")
def generated() -> dict[str, str]:
    """What the writer produces today, for every variant."""
    return {variant: wvcb.variant_hull(variant) for variant in VARIANTS}


def clearance(position: np.ndarray, boxes) -> float:
    """Distance from `position` to the nearest surface of `boxes`."""
    centers, half_extents = boxes
    gap = np.maximum(np.abs(np.asarray(position) - centers) - half_extents, 0.0)
    return float(np.linalg.norm(gap, axis=1).min())


@pytest.fixture(scope="module")
def base_boxes():
    return load_collision_boxes("collision_boxes.yaml")


# Radius around the occupied port that holds the whole berthed vehicle. The
# writer's new-geometry set also contains re-tessellated station structure
# 20-35 m away, which belongs to the shipped station boxes, not to a vehicle.
VEHICLE_RADIUS = 15.0


def uncovered_area(triangles: np.ndarray, boxes, radius: float) -> tuple[float, float]:
    """Area of `triangles` further than `radius` from every box, and their total.

    Each triangle is sampled at six barycentric points and contributes the
    fraction of them that are out of reach, so a partly covered triangle
    counts partly. `radius` is the chaser radius the collision test expands
    the boxes by, so anything nearer than that is already caught.
    """
    centers, half_extents = boxes
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(edge_a, edge_b), axis=1)

    weights = np.array(
        [
            [2 / 3, 1 / 6, 1 / 6],
            [1 / 6, 2 / 3, 1 / 6],
            [1 / 6, 1 / 6, 2 / 3],
            [1 / 3, 1 / 3, 1 / 3],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
        ]
    )
    points = np.einsum("kj,tjd->tkd", weights, triangles).reshape(-1, 3)
    far = np.empty(len(points), dtype=bool)
    for start in range(0, len(points), 4096):
        chunk = points[start : start + 4096]
        gap = np.maximum(np.abs(chunk[:, None, :] - centers[None]) - half_extents[None], 0.0)
        far[start : start + 4096] = np.linalg.norm(gap, axis=2).min(axis=1) > radius

    fraction = far.reshape(len(triangles), -1).mean(axis=1)
    return float((area * fraction).sum()), float(area.sum())


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_hulls_are_byte_stable_under_regeneration(variant, generated):
    # Rerunning the writer must be a no-op on an up-to-date checkout;
    # otherwise every regeneration shows a spurious diff and a committed hull
    # can drift from the asset it describes without anything noticing.
    path = RESOURCES / wvcb.hull_filename(variant)
    assert path.read_text() == generated[variant]


@pytest.mark.parametrize("variant", VARIANTS)
def test_each_hull_is_the_shipped_station_plus_a_vehicle(variant):
    # Self-containment: collision_boxes_path names one file, so a variant hull
    # has to carry the whole station as well as the vehicle. Compared as the
    # leading boxes rather than as a set, so the station geometry is not just
    # present but unmodified.
    shipped = yaml.safe_load((RESOURCES / "collision_boxes.yaml").read_text())
    boxes = yaml.safe_load((RESOURCES / wvcb.hull_filename(variant)).read_text())
    assert boxes[: len(shipped)] == shipped
    assert len(boxes) > len(shipped)


@pytest.mark.parametrize(
    "variant, asset, hull",
    [
        ("dragon", "ISS_dragon.glb", "collision_boxes_dragon.yaml"),
        ("cygnus", "ISS_cygnus.glb", "collision_boxes_cygnus.yaml"),
        ("soyuz", "ISS_soyuz.glb", "collision_boxes_soyuz.yaml"),
    ],
)
def test_each_variant_config_pairs_its_asset_with_its_own_hull(variant, asset, hull):
    # Spelled out rather than derived from the writer's table: the pairing is
    # what these configs exist for, and a writer that assembled the wrong pair
    # would take the committed files with it. A config naming ISS_dragon.glb
    # beside collision_boxes_cygnus.yaml renders one visiting vehicle and
    # collides against another.
    cfg = NumericalConfig.from_toml(NUMERICAL / "env" / f"coop_{variant}.toml")
    assert cfg.render == {"iss_asset": asset}
    assert cfg.physics.collision_boxes_path == hull


@pytest.mark.parametrize(
    "variant, port_name",
    [
        ("dragon", "harmony_fwd_pma2"),
        ("cygnus", "unity_nadir_cbm"),
        ("soyuz", "zvezda_aft"),
    ],
)
def test_the_occupied_port_is_blocked_under_its_variant_hull(variant, port_name):
    # The property the whole change exists for, through the real EventChecker
    # rather than a distance formula: a chaser holding the occupied port's
    # goal pose is in collision with the berthed vehicle, where against the
    # bare-station hull the same pose is clear.
    assert wvcb.OCCUPIED_PORT[variant] == port_name
    position, _ = port_pose(PORTS_BY_NAME[port_name])
    at_rest = np.asarray(position, dtype=np.float32)

    variant_cfg = NumericalConfig.from_toml(NUMERICAL / "env" / f"coop_{variant}.toml")
    base_cfg = NumericalConfig.from_toml(NUMERICAL / "env" / "coop_goal.toml")
    assert bool(EventChecker(variant_cfg).collision(at_rest, at_rest))
    assert not bool(EventChecker(base_cfg).collision(at_rest, at_rest))


@pytest.mark.parametrize("variant", VARIANTS)
def test_no_part_of_the_berthed_vehicle_is_out_of_reach_of_its_hull(variant):
    # Coverage of the whole vehicle surface, not just the goal pose: every
    # point of it must lie within a chaser radius of some box, or a chaser can
    # fly through that part -- a solar array wing, say -- and report nothing.
    # Measured on the vehicle's own triangles, area-weighted, against the
    # committed hull and the same radius the EventChecker expands boxes by.
    cfg = NumericalConfig.from_toml(NUMERICAL / "env" / f"coop_{variant}.toml")
    radius = cfg.physics.dragon_collision_radius_m

    port = np.asarray(PORTS_BY_NAME[wvcb.OCCUPIED_PORT[variant]].interface, dtype=np.float64)
    triangles = wvcb.new_triangles(variant)
    vehicle = triangles[np.linalg.norm(triangles.mean(axis=1) - port, axis=1) < VEHICLE_RADIUS]
    assert len(vehicle) > 0

    boxes = load_collision_boxes(wvcb.hull_filename(variant))
    beyond, total = uncovered_area(vehicle, boxes, radius)
    assert total > 0.0
    assert beyond == 0.0, f"{beyond:.2f} m^2 of {total:.1f} m^2 beyond {radius} m"


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_unoccupied_ports_keep_their_shipped_clearances(variant, base_boxes):
    # Exactly equal, not merely close: every added box belongs to the vehicle
    # at one port, so no other port's nearest surface may move at all. A
    # difference here means the writer's cluster selection reached beyond the
    # vehicle into station structure.
    variant_boxes = load_collision_boxes(wvcb.hull_filename(variant))
    occupied = wvcb.OCCUPIED_PORT[variant]
    others = [port for port in PORTS if port.name != occupied]
    assert len(others) == 7
    for port in others:
        position, _ = port_pose(port)
        assert clearance(position, variant_boxes) == clearance(position, base_boxes), port.name
