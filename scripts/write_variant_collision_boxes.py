"""Write the committed collision hulls for the ISS variant render assets.

Three render assets berth a visiting vehicle at a station port --
ISS_dragon.glb at harmony_fwd_pma2, ISS_cygnus.glb at unity_nadir_cbm,
ISS_soyuz.glb at zvezda_aft -- and the shipped 313-box hull describes the
bare station, so a chaser flown against it passes straight through the
vehicle the renderer draws. This writes one hull per variant into
src/owm_envs/envs/common/resources/, each the shipped 313 boxes plus boxes
covering that variant's vehicle, so a config's collision_boxes_path names a
single self-contained file.

The vehicle is found geometrically, not by module name: a variant adds nodes
for re-tessellated station structure alongside the vehicle's own, so the nodes
a name-based diff reports are not confined to the vehicle. Instead, take the
triangles of the variant whose every vertex is further than NEW_VERTEX_RADIUS
from every vertex of the base asset, and mark the voxels their surfaces run
through, sampling each triangle's interior at SAMPLE_SPACING -- fine enough
that a voxel goes unmarked only where a triangle clips it by less than that
spacing. Marking surfaces rather than corners is what makes solar arrays --
quads whose corners are metres apart -- occupy a connected run of cells
instead of a few scattered ones.

Those cells are clustered at VOXEL connectivity and only the cluster
nearest the occupied port is kept. The other clusters are re-tessellated
station structure the shipped hull already approximates, 20-35 m from the
port; keeping them would change collision behaviour at ports that have no
visiting vehicle. That cluster's cells are merged into axis-aligned boxes by
greedy meshing, which never covers empty space.

Reads scipy (cKDTree, ndimage.label), which is not a declared dependency of
this package but arrives transitively with jax.

Rerun after changing an ISS_<variant>.glb asset, the PORTS table's interface
points, or the constants below; tests/test_variant_collision_boxes.py fails
when the committed hulls drift from what this writes.

Usage:
    uv run python scripts/write_variant_collision_boxes.py
"""

from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage
from scipy.spatial import cKDTree

from owm_envs.envs.common.config import (
    DEFAULT_COLLISION_BOXES_FILENAME,
    default_collision_boxes_path,
)
from owm_envs.envs.common.docking_ports import PORTS_BY_NAME
from owm_envs.render.iss_frame import ISS_RECENTRE_OFFSET

# A sibling script, not a package module: it reads the glTF document directly
# so it needs no render extra and no GPU, which is what makes it usable here.
from check_iss_asset import UPRIGHT, module_vertices, node_matrix, read_glb

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "src/owm_envs/render/resources/international-space-station"
RESOURCES = REPO / "src/owm_envs/envs/common/resources"

# Edge length of the voxel grid the vehicle is discretised on, in metres, and
# the connectivity radius the new geometry is clustered at.
VOXEL = 0.5
# Spacing of the points sampled across each triangle. Below the voxel edge, so
# a triangle cannot step over a cell it passes through.
SAMPLE_SPACING = VOXEL / 3.0
# A vertex of the variant this far from every base-asset vertex, in metres, is
# geometry the base asset does not have.
NEW_VERTEX_RADIUS = 0.25

# glTF primitive mode for triangle lists, and the index accessor component
# types the spec allows.
_TRIANGLES = 4
_INDEX_DTYPE = {5121: "<u1", 5123: "<u2", 5125: "<u4"}

# Which port each variant berths its visiting vehicle at. The port names the
# interface point the cluster search measures against, so this table and the
# render asset are the only per-variant inputs.
OCCUPIED_PORT = {
    "dragon": "harmony_fwd_pma2",
    "cygnus": "unity_nadir_cbm",
    "soyuz": "zvezda_aft",
}


def hull_filename(variant: str) -> str:
    """Name of `variant`'s hull file within the package resources directory."""
    stem = Path(DEFAULT_COLLISION_BOXES_FILENAME).stem
    return f"{stem}_{variant}.yaml"


def _positions(document: dict, binary: bytes, index: int) -> np.ndarray:
    """Accessor `index` read as (N, 3) float32 vertex positions."""
    accessor = document["accessors"][index]
    view = document["bufferViews"][accessor["bufferView"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride", 12)
    count = accessor["count"]
    raw = np.frombuffer(binary, dtype=np.uint8, count=(count - 1) * stride + 12, offset=start)
    rows = np.lib.stride_tricks.as_strided(raw, shape=(count, 12), strides=(stride, 1)).copy()
    return rows.view("<f4").reshape(count, 3).astype(np.float64)


def _indices(document: dict, binary: bytes, index: int) -> np.ndarray:
    """Accessor `index` read as a flat array of vertex indices."""
    accessor = document["accessors"][index]
    view = document["bufferViews"][accessor["bufferView"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    dtype = np.dtype(_INDEX_DTYPE[accessor["componentType"]])
    return np.frombuffer(binary, dtype=dtype, count=accessor["count"], offset=start).astype(int)


def world_triangles(path: Path) -> np.ndarray:
    """(T, 3, 3) world-frame triangles of every mesh in the asset.

    The world frame is the one `ISSScene` renders and `module_vertices`
    reports: upright rotation, then recentring by ISS_RECENTRE_OFFSET.
    """
    document, binary = read_glb(path)
    nodes = document.get("nodes", [])
    triangles: list[np.ndarray] = []

    def visit(index: int, parent_matrix: np.ndarray) -> None:
        node = nodes[index]
        matrix = parent_matrix @ node_matrix(node)
        if "mesh" in node:
            for primitive in document["meshes"][node["mesh"]]["primitives"]:
                if primitive.get("mode", _TRIANGLES) != _TRIANGLES:
                    continue
                points = _positions(document, binary, primitive["attributes"]["POSITION"])
                homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
                world = (homogeneous @ matrix.T)[:, :3]
                if "indices" in primitive:
                    order = _indices(document, binary, primitive["indices"])
                else:
                    order = np.arange(len(world))
                whole = (len(order) // 3) * 3
                triangles.append(world[order[:whole]].reshape(-1, 3, 3))
        for child in node.get("children", []):
            visit(child, matrix)

    for scene in document.get("scenes", []):
        for root in scene.get("nodes", []):
            visit(root, np.eye(4))

    if not triangles:
        return np.zeros((0, 3, 3))
    offset = np.asarray(ISS_RECENTRE_OFFSET, dtype=np.float64)
    return np.concatenate(triangles) @ UPRIGHT.T - offset


def sample_triangles(triangles: np.ndarray, spacing: float) -> np.ndarray:
    """Points covering each triangle at roughly `spacing`, plus its corners.

    The lattice is sized from the triangle's longest edge, so no two adjacent
    samples are further apart than `spacing` whatever the triangle's shape: a
    cell the triangle enters by more than `spacing` from every face always
    holds a sample, though one it merely clips at a corner may not. Sizing
    from the square root of the area instead would undersample a sliver by the
    square root of its aspect ratio, and a zero-area triangle entirely.

    Sample count is quadratic in the longest edge, so a triangle metres across
    costs thousands of points whatever its area -- the shipped assets stay
    under 30 samples an edge. There is no cap, because a cap is what lets a
    long triangle step over a voxel.

    Triangles are grouped by how many samples they need along an edge so the
    barycentric lattice is built once per group rather than once per triangle.
    """
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    longest = np.maximum.reduce(
        [
            np.linalg.norm(b - a, axis=1),
            np.linalg.norm(c - b, axis=1),
            np.linalg.norm(a - c, axis=1),
        ]
    )
    per_edge = np.maximum(np.ceil(longest / spacing).astype(int) + 1, 2)

    points = [triangles.reshape(-1, 3)]
    for n in np.unique(per_edge):
        step = np.linspace(0.0, 1.0, n)
        u, v = np.meshgrid(step, step, indexing="ij")
        inside = (u + v) <= 1.0
        u, v = u[inside], v[inside]
        weights = np.stack([1.0 - u - v, u, v], axis=1)
        group = triangles[per_edge == n]
        points.append(np.einsum("kj,tjd->tkd", weights, group).reshape(-1, 3))
    return np.concatenate(points)


def new_triangles(variant: str) -> np.ndarray:
    """Triangles of `variant` whose every vertex is new to the base asset.

    New means further than NEW_VERTEX_RADIUS from every base-asset vertex. All
    three vertices must be new, so a triangle straddling the seam between the
    station and its visiting vehicle stays with the station.
    """
    base = np.vstack(list(module_vertices(ASSETS / "ISS_base.glb").values()))
    triangles = world_triangles(ASSETS / f"ISS_{variant}.glb")
    distance, _ = cKDTree(base).query(triangles.reshape(-1, 3), k=1)
    return triangles[(distance.reshape(-1, 3) > NEW_VERTEX_RADIUS).all(axis=1)]


def vehicle_voxels(points: np.ndarray, port: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Occupied voxel indices of the cluster nearest `port`, and the grid origin."""
    origin = points.min(axis=0)
    index = np.floor((points - origin) / VOXEL).astype(int)
    grid = np.zeros(index.max(axis=0) + 3, dtype=bool)
    grid[index[:, 0], index[:, 1], index[:, 2]] = True

    labels, count = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    best, best_distance = None, np.inf
    for label in range(1, count + 1):
        cells = np.argwhere(labels == label)
        centre = (cells.mean(axis=0) + 0.5) * VOXEL + origin
        distance = float(np.linalg.norm(centre - port))
        if distance < best_distance:
            best, best_distance = cells, distance
    return best, origin


def greedy_boxes(cells: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Merge occupied cells into axis-aligned boxes, largest-first per seed.

    Grows each seed along x, then y, then z while every cell of the candidate
    slab is occupied and unclaimed, which is standard greedy meshing: it is not
    the minimal decomposition, but it is stable and never covers empty space.
    """
    occupied = {tuple(c) for c in cells}
    remaining = set(occupied)
    boxes: list[tuple[np.ndarray, np.ndarray]] = []
    while remaining:
        x0, y0, z0 = min(remaining)
        x1 = x0
        while (x1 + 1, y0, z0) in remaining:
            x1 += 1
        y1 = y0
        while all((x, y1 + 1, z0) in remaining for x in range(x0, x1 + 1)):
            y1 += 1
        z1 = z0
        while all(
            (x, y, z1 + 1) in remaining
            for x in range(x0, x1 + 1)
            for y in range(y0, y1 + 1)
        ):
            z1 += 1
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                for z in range(z0, z1 + 1):
                    remaining.discard((x, y, z))
        boxes.append((np.array([x0, y0, z0]), np.array([x1, y1, z1])))
    return boxes


def to_world(boxes: list[tuple[np.ndarray, np.ndarray]], origin: np.ndarray) -> list[dict]:
    """Voxel-index box corners as {center, size} dicts in world metres."""
    out = []
    for lo, hi in boxes:
        low = lo * VOXEL + origin
        high = (hi + 1) * VOXEL + origin
        centre = (low + high) / 2.0
        size = high - low
        out.append(
            {
                "center": [round(float(v), 4) for v in centre],
                "size": [round(float(v), 4) for v in size],
            }
        )
    return out


def vehicle_boxes(variant: str) -> list[dict]:
    """Collision boxes covering the vehicle `variant` berths at its port."""
    port = np.asarray(PORTS_BY_NAME[OCCUPIED_PORT[variant]].interface, dtype=np.float64)
    marks = sample_triangles(new_triangles(variant), SAMPLE_SPACING)
    cells, origin = vehicle_voxels(marks, port)
    return to_world(greedy_boxes(cells), origin)


def variant_hull(variant: str) -> str:
    """The shipped station hull plus `variant`'s vehicle, as YAML text.

    The station boxes are carried through as loaded rather than recomputed, so
    a variant hull differs from the shipped one only by the appended vehicle.
    """
    base = yaml.safe_load(Path(default_collision_boxes_path()).read_text())
    return yaml.safe_dump(
        base + vehicle_boxes(variant), sort_keys=False, default_flow_style=None
    )


def main() -> None:
    for variant in OCCUPIED_PORT:
        path = RESOURCES / hull_filename(variant)
        text = variant_hull(variant)
        path.write_text(text)
        boxes = yaml.safe_load(text)
        print(f"wrote {path} ({len(boxes)} boxes)")


if __name__ == "__main__":
    main()
