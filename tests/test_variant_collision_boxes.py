"""The variant collision hulls must block the port their asset occupies.

Three render assets berth a visiting vehicle at a station port, and the
shipped 313-box hull describes the bare station: flown against it, a chaser
passes through the vehicle the renderer draws and reports a clean dock. Each
committed collision_boxes_<variant>.yaml is that hull plus boxes covering the
vehicle, written by scripts/write_variant_collision_boxes.py.

Six things are checked, and only the first is a drift guard. That the
committed hulls still equal what the writer produces; that each variant
config names its own asset beside its own hull, since a config that draws a
Dragon while colliding against a Cygnus is the failure this pairing exists to
prevent; that the asset it names resolves to a packaged file, since `render`
is an unvalidated dict and a renamed or absent GLB would otherwise surface
only at render time; that the occupied port is now genuinely blocked, which is
the whole reason the hulls exist; that no part of the berthed vehicle's
surface is out of reach of the hull, since a chaser that flies through an
uncovered solar array registers nothing and the goal-pose check above sees
only the one pose; and that the other seven ports keep exactly their shipped
clearances, which is what confines a hull to its vehicle. The last would fail
if the writer's cluster selection had pulled in re-tessellated station
structure alongside the vehicle -- geometry the shipped hull already
approximates, whose inclusion would silently change collision behaviour at
ports that berth nothing.

One further check is on the writer rather than on a hull: that its sampling
lattice leaves no cell unmarked that a triangle enters by more than a sample
spacing, on the triangle shapes -- slivers, degenerate edges, faces metres
across -- a lattice sized by area steps over.
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
from owm_envs.render import asset_path

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

# Triangles chosen for the shapes that defeat an area-sized sampling lattice:
# a 50 m x 1 mm sliver, a 10:1 right triangle, three collinear points spanning
# 30 m with no area at all, and a 30 m equilateral face large enough that any
# fixed cap on the lattice size would bite. Shifted so that no vertex lands on
# a cell boundary and the three planar ones sit near the middle of a cell in
# z, where a flat triangle reaches a cell's interior rather than its face.
ADVERSARIAL_TRIANGLES = np.array(
    [
        [[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [0.0, 1e-3, 0.0]],
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [15.0, 7.5, 3.0], [30.0, 15.0, 6.0]],
        [[0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [15.0, 25.981, 0.0]],
    ]
) + np.array([0.137, 0.229, 0.263])


def occupied_cells(points: np.ndarray, depth: float = 0.0) -> set[tuple[int, ...]]:
    """Indices of the VOXEL cells `points` fall in, on a grid through the origin.

    A point nearer than `depth` to any face of its cell is dropped. No lattice
    of pitch `depth` can be relied on to land inside a cell a surface merely
    grazes at a corner, so that is what separates the cells a triangle passes
    through from the ones it only clips.
    """
    index = np.floor(points / wvcb.VOXEL).astype(int)
    if depth > 0.0:
        offset = points - index * wvcb.VOXEL
        index = index[np.minimum(offset, wvcb.VOXEL - offset).min(axis=1) >= depth]
    return set(map(tuple, index.tolist()))


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


@pytest.mark.parametrize("index", range(len(ADVERSARIAL_TRIANGLES)))
def test_sampling_marks_the_voxels_a_triangle_enters_a_sample_deep(index):
    # The hull is built from the cells the sampled points land in, so a
    # lattice coarser than VOXEL lets a triangle cross a cell nothing marks --
    # an unfilled hole in the vehicle, and one that grows with the triangle's
    # aspect ratio rather than its area. Ground truth is 200k points drawn
    # uniformly over the triangle.
    #
    # The predicate is deliberately "enters a cell by more than
    # SAMPLE_SPACING from every face", not "touches the cell at all". The
    # latter is unsatisfiable by any finite lattice: a triangle can clip a
    # cell corner in less space than the lattice pitch, and asserting it
    # fails on the 30 m equilateral below even against a correct sampler.
    # Deeper than the pitch is the provable guarantee -- a lattice of pitch p
    # always holds a sample inside a cell a surface penetrates further than p
    # from every face. So this is not a hedge, and strengthening it to bare
    # containment would only make the test unsatisfiable.
    #
    # It costs no detection power either. Sizing the lattice from sqrt(area)
    # rather than the longest edge leaves 99 of 100, 9 of 34, 10 of 12 and
    # 1321 of 1603 cells unmarked across the four triangles below.
    triangle = ADVERSARIAL_TRIANGLES[index : index + 1]
    marked = occupied_cells(wvcb.sample_triangles(triangle, wvcb.SAMPLE_SPACING))

    uv = np.random.default_rng(0).random((200_000, 2))
    folded = uv.sum(axis=1) > 1.0
    uv[folded] = 1.0 - uv[folded]
    weights = np.column_stack([1.0 - uv.sum(axis=1), uv])
    reached = occupied_cells(weights @ triangle[0], depth=wvcb.SAMPLE_SPACING)

    # Guards against a vacuous pass: a triangle lying near a cell face is
    # filtered away entirely and would then assert over an empty set.
    assert len(reached) > 10
    missed = reached - marked
    assert not missed, f"{len(missed)} of {len(reached)} cells reached but unmarked"


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


@pytest.mark.parametrize("variant", VARIANTS)
def test_each_variant_config_names_a_render_asset_that_is_packaged(variant):
    # NumericalConfig.render is dict[str, Any] with no validation, so the
    # asset name reaches the renderer unchecked: a renamed, deleted or
    # unfetched git-lfs GLB passes every other check in this file and fails
    # only when a frame is drawn. asset_path resolves within the packaged
    # resources and raises when the file is absent, so this needs no GPU.
    cfg = NumericalConfig.from_toml(NUMERICAL / "env" / f"coop_{variant}.toml")
    path = asset_path("international-space-station", cfg.render["iss_asset"])
    assert path.is_file()


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
