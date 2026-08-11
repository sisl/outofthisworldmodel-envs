"""Write the committed collision hulls for the ISS variant render assets.

Three render assets berth a visiting vehicle at a station port --
ISS_dragon.glb at harmony_fwd_pma2, ISS_cygnus.glb at unity_nadir_cbm,
ISS_soyuz.glb at zvezda_aft -- and the shipped 313-box hull describes the
bare station, so a chaser flown against it passes straight through the
vehicle the renderer draws. This writes one hull per variant into
src/owm_envs/envs/common/resources/, each the shipped 313 boxes plus boxes
covering that variant's vehicle, so a config's collision_boxes_path names a
single self-contained file.

The vehicle is found geometrically, not by module name: modules are renamed
and re-tessellated between assets, so a name-based diff spans the whole
station rather than the vehicle. Instead, take the vertices of the variant
with no vertex of the base asset within NEW_VERTEX_RADIUS, cluster them at
VOXEL connectivity, and keep only the cluster nearest the occupied port. The
other clusters are re-tessellated station structure the shipped hull already
approximates, 20-35 m from the port; keeping them would change collision
behaviour at ports that have no visiting vehicle. That cluster is then
voxelised at VOXEL and its occupied cells merged into axis-aligned boxes by
greedy meshing, which never covers empty space.

Reads scipy (cKDTree, ndimage.label), which is not a declared dependency of
this package but arrives transitively with jax.

Rerun after changing an ISS_<variant>.glb asset, the PORTS table's interface
points, or the constants below; tests/test_variant_collision_boxes.py fails
when the committed hulls drift from what this writes.

Usage:
    uv run --extra datasets python scripts/write_variant_collision_boxes.py
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

# A sibling script, not a package module: it reads the glTF document directly
# so it needs no render extra and no GPU, which is what makes it usable here.
from check_iss_asset import module_vertices

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "src/owm_envs/render/resources/international-space-station"
RESOURCES = REPO / "src/owm_envs/envs/common/resources"

# Edge length of the voxel grid the vehicle is discretised on, in metres, and
# the connectivity radius the new geometry is clustered at.
VOXEL = 0.5
# A vertex of the variant this far from every base-asset vertex, in metres, is
# geometry the base asset does not have.
NEW_VERTEX_RADIUS = 0.25

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


def new_geometry(variant: str) -> np.ndarray:
    """Vertices of `variant` with no base-asset vertex within NEW_VERTEX_RADIUS."""
    base = np.vstack(list(module_vertices(ASSETS / "ISS_base.glb").values()))
    points = np.vstack(list(module_vertices(ASSETS / f"ISS_{variant}.glb").values()))
    distance, _ = cKDTree(base).query(points, k=1)
    return points[distance > NEW_VERTEX_RADIUS]


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
    cells, origin = vehicle_voxels(new_geometry(variant), port)
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
