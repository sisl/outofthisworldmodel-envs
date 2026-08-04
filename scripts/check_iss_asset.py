"""Check that an ISS asset variant places the station where the reference does.

The station's world placement is a constant of the environment: the collision
hull, the dock poses and the recentre offset in `render/iss_frame` are all
authored against it. A variant asset (other spacecraft berthed at other
ports) is only usable if every module it shares with the reference sits at
exactly the same coordinates, so swapping the file cannot move the station.

Works on the glTF document directly -- vertex positions grouped by their
nearest named ancestor node -- so it needs no render extra and no GPU. The
pygfx scene graph drops glTF node names, which is why this does not go
through `render/loaders`. Positions are reported in the world frame
`ISSScene` renders: upright rotation (asset +Y -> world +Z), then recentring
by `ISS_RECENTRE_OFFSET`.

    uv run scripts/check_iss_asset.py path/to/variant.glb

Exits non-zero if any module shared with the reference moved or changed
geometry. Modules only in one asset (added or removed visiting vehicles) are
reported with their world bounding box but are not failures.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from owm_envs.render import asset_path
from owm_envs.render.iss_frame import ISS_RECENTRE_OFFSET, UPRIGHT_EULER_XYZ

app = typer.Typer(add_completion=False)

_JSON_CHUNK = 0x4E4F534A
_BIN_CHUNK = 0x004E4942
_FLOAT32 = 5126


def _upright_matrix(euler_xyz: tuple[float, float, float]) -> np.ndarray:
    """Rotation matrix for an xyz euler triple, matching `la.quat_from_euler`.

    pylinalg's lowercase "xyz" order is extrinsic, so the composed matrix is
    R_z @ R_y @ R_x. Derived here in plain numpy rather than through pylinalg
    because this tool must run without the render extra installed.
    """
    ax, ay, az = (float(a) for a in euler_xyz)
    rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(ax), -np.sin(ax)],
                   [0.0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0.0, np.sin(ay)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(ay), 0.0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0.0],
                   [np.sin(az), np.cos(az), 0.0],
                   [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


# The upright rotation ISSScene applies: asset +Y -> world +Z.
_UPRIGHT = _upright_matrix(UPRIGHT_EULER_XYZ)


def read_glb(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    magic, version, _ = struct.unpack_from("<III", raw, 0)
    if magic != 0x46546C67 or version != 2:
        raise ValueError(f"{path} is not a glTF 2.0 binary file")

    offset, document, binary = 12, None, b""
    while offset < len(raw):
        length, kind = struct.unpack_from("<II", raw, offset)
        payload = raw[offset + 8 : offset + 8 + length]
        if kind == _JSON_CHUNK:
            document = json.loads(payload.decode("utf-8"))
        elif kind == _BIN_CHUNK:
            binary = payload
        offset += 8 + length
    if document is None:
        raise ValueError(f"{path} has no JSON chunk")
    return document, binary


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T
    matrix = np.eye(4)
    x, y, z, w = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    matrix[:3, :3] = rotation * np.asarray(node.get("scale", (1.0, 1.0, 1.0)))[None, :]
    matrix[:3, 3] = node.get("translation", (0.0, 0.0, 0.0))
    return matrix


def _positions(document: dict, binary: bytes, mesh_index: int) -> np.ndarray:
    chunks = []
    for primitive in document["meshes"][mesh_index]["primitives"]:
        accessor = document["accessors"][primitive["attributes"]["POSITION"]]
        if accessor["componentType"] != _FLOAT32 or accessor["type"] != "VEC3":
            raise ValueError("POSITION accessor is not float32 VEC3")
        view = document["bufferViews"][accessor["bufferView"]]
        start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        stride = view.get("byteStride", 12)
        count = accessor["count"]
        raw = np.frombuffer(binary, dtype=np.uint8, count=(count - 1) * stride + 12, offset=start)
        rows = np.lib.stride_tricks.as_strided(raw, shape=(count, 12), strides=(stride, 1)).copy()
        chunks.append(rows.view("<f4").reshape(count, 3).astype(np.float64))
    return np.concatenate(chunks) if chunks else np.zeros((0, 3))


def module_vertices(path: Path) -> dict[str, np.ndarray]:
    """World-frame vertices per module, keyed by the nearest named ancestor node."""
    document, binary = read_glb(path)
    nodes = document.get("nodes", [])
    modules: dict[str, list[np.ndarray]] = {}

    def visit(index: int, parent_matrix: np.ndarray, owner: str) -> None:
        node = nodes[index]
        matrix = parent_matrix @ _node_matrix(node)
        name = str(node.get("name", "")).strip()
        if name:
            owner = name
        if "mesh" in node:
            points = _positions(document, binary, node["mesh"])
            homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
            modules.setdefault(owner or "<unnamed>", []).append((homogeneous @ matrix.T)[:, :3])
        for child in node.get("children", []):
            visit(child, matrix, owner)

    for scene in document.get("scenes", []):
        for root in scene.get("nodes", []):
            visit(root, np.eye(4), "")

    offset = np.asarray(ISS_RECENTRE_OFFSET, dtype=np.float64)
    return {
        name: np.concatenate(chunks) @ _UPRIGHT.T - offset
        for name, chunks in modules.items()
    }


def _bbox(points: np.ndarray) -> str:
    low, high = points.min(axis=0), points.max(axis=0)
    return (
        f"[{low[0]:7.2f},{high[0]:7.2f}] x "
        f"[{low[1]:7.2f},{high[1]:7.2f}] x "
        f"[{low[2]:7.2f},{high[2]:7.2f}] m"
    )


@app.command()
def main(
    candidate: Path = typer.Argument(..., help="Variant GLB to check."),
    reference: Optional[Path] = typer.Option(
        None, help="Baseline GLB; default is the shipped ISS asset."
    ),
    tolerance: float = typer.Option(
        1e-6, help="Maximum allowed displacement of a shared module, in metres."
    ),
) -> None:
    baseline = reference or asset_path("international-space-station", "ISS_stationary.glb")
    ref, cand = module_vertices(Path(baseline)), module_vertices(candidate)

    removed = sorted(set(ref) - set(cand))
    added = sorted(set(cand) - set(ref))
    shared = sorted(set(ref) & set(cand))
    typer.echo(f"modules: {len(ref)} reference, {len(cand)} candidate, {len(shared)} shared")
    for label, names, table in (("removed", removed, ref), ("added", added, cand)):
        for name in names:
            typer.echo(f"  {label}: {name}  {_bbox(table[name])}")

    failures = []
    worst = 0.0
    for name in shared:
        if ref[name].shape != cand[name].shape:
            failures.append(f"  changed geometry: {name} "
                            f"({ref[name].shape[0]} -> {cand[name].shape[0]} verts)")
            continue
        displacement = float(np.abs(ref[name] - cand[name]).max())
        worst = max(worst, displacement)
        if displacement > tolerance:
            failures.append(f"  moved {displacement:.6f} m: {name}")

    typer.echo(f"max displacement over unchanged shared modules: {worst:.9f} m")
    if failures:
        typer.echo("FAIL: the station would not render where the reference does:")
        for line in failures:
            typer.echo(line)
        raise typer.Exit(1)
    typer.echo("OK: every shared module is identical; the station renders at the same world location.")


if __name__ == "__main__":
    app()
