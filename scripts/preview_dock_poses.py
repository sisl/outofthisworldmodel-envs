"""Render every candidate docking pose so a goal list can be checked by eye.

For each port in `owm_envs.envs.common.docking_ports.PORTS` (or a subset) this
builds the goal state the environment would hold at that port and renders four
panels of the same scene the dataset pipeline uses:

  chaser view    DRAGON_FPV from the goal pose, i.e. what the onboard camera sees
  corridor A/B   two orthographic views 90 degrees apart, both taken from
                 directions perpendicular to the approach normal, so the
                 approach axis lies in the image plane
  context        the fixed ISS_ISO shot, identical across poses for comparison

The overlay draws the world axes at the origin (x red, y green, z blue), a
sphere at every port interface, a sphere at the active goal, and a shaft from
the goal down the corridor to its interface. Per-pose PNGs, a contact sheet and
a TOML block for `DockConfig` are written to `--out`.

    uv run --extra render scripts/preview_dock_poses.py --out logs/dock_preview
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pygfx as gfx
import pylinalg as la
import typer
from PIL import Image, ImageDraw

from owm_envs.envs.common.config import load_collision_boxes
from owm_envs.envs.common.docking_ports import (
    PORTS,
    PORTS_BY_NAME,
    DockingPort,
    port_pose,
    resolve_port_names,
)
from owm_envs.envs.iss.config import ISSConfig
from owm_envs.render.iss_scene import RenderConfig
from owm_envs.render.renderer import ISSRenderer
from owm_envs.render.view import CameraView

app = typer.Typer(add_completion=False)

GOAL_COLOR = (1.0, 0.82, 0.10, 1.0)
PORT_COLOR = (0.20, 0.72, 1.0, 1.0)
AXIS_COLORS = ((1.0, 0.25, 0.25, 1.0), (0.30, 1.0, 0.35, 1.0), (0.35, 0.55, 1.0, 1.0))
AXIS_LENGTH = 22.0


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-9)


def goal_state(position: np.ndarray, quat: np.ndarray) -> np.ndarray:
    state = np.zeros(13, dtype=np.float32)
    state[0:3] = position
    state[6:10] = quat
    return state


def clearance(position: np.ndarray, centers: np.ndarray, half_extents: np.ndarray) -> float:
    gap = np.maximum(np.abs(position[None, :] - centers) - half_extents, 0.0)
    return float(np.linalg.norm(gap, axis=1).min())


def _arrow(start: np.ndarray, end: np.ndarray, color, radius: float) -> gfx.Group:
    direction = end - start
    length = float(np.linalg.norm(direction))
    material = gfx.MeshBasicMaterial(color=color)

    group = gfx.Group()
    shaft = gfx.Mesh(gfx.cylinder_geometry(radius, radius, length, 16, 1), material)
    shaft.local.position = (0.0, 0.0, 0.5 * length)
    head = gfx.Mesh(gfx.cone_geometry(2.6 * radius, 6.0 * radius, 16, 1), material)
    head.local.position = (0.0, 0.0, length)
    group.add(shaft)
    group.add(head)

    group.local.position = tuple(start.tolist())
    group.local.rotation = la.quat_from_vecs(
        np.array([0.0, 0.0, 1.0], dtype=np.float32), _unit(direction).astype(np.float32)
    )
    return group


def _sphere(centre: np.ndarray, radius: float, color) -> gfx.Mesh:
    mesh = gfx.Mesh(
        gfx.sphere_geometry(radius=radius, width_segments=24, height_segments=12),
        gfx.MeshBasicMaterial(color=color),
    )
    mesh.local.position = tuple(np.asarray(centre, dtype=float).tolist())
    return mesh


def build_overlay(port: DockingPort, position: np.ndarray, scale: float) -> gfx.Group:
    interface = np.asarray(port.interface, dtype=float)
    overlay = gfx.Group()

    origin = np.zeros(3)
    for axis in range(3):
        end = origin.copy()
        end[axis] = AXIS_LENGTH
        overlay.add(_arrow(origin, end, AXIS_COLORS[axis], 0.16 * scale))
    overlay.add(_sphere(origin, 0.5 * scale, (1.0, 0.35, 1.0, 1.0)))

    for other in PORTS:
        overlay.add(_sphere(np.asarray(other.interface, dtype=float), 0.55 * scale, PORT_COLOR))

    overlay.add(_sphere(position, 1.1 * scale, GOAL_COLOR))
    overlay.add(_arrow(position, interface, GOAL_COLOR, 0.28 * scale))
    return overlay


@contextmanager
def station_only(scene: gfx.Scene, extent_limit: float = 1.0e4):
    """Hide Earth, the Moon and the Sun for the duration of the block.

    Selected by world extent rather than by name: everything in this scene that
    is kilometres across is celestial, and everything that is metres across is
    the station or the chaser.
    """
    hidden = []
    for child in list(scene.children):
        box = child.get_world_bounding_box()
        if box is not None and float(np.max(box[1] - box[0])) > extent_limit and child.visible:
            child.visible = False
            hidden.append(child)
    try:
        yield
    finally:
        for child in hidden:
            child.visible = True


def corridor_views(position: np.ndarray, normal: np.ndarray, size: int) -> list[CameraView]:
    """Two orthographic cameras looking across the approach corridor, 90 deg apart."""
    reference = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    side = _unit(np.cross(normal, reference))
    up = _unit(np.cross(side, normal))

    target = 0.5 * position
    half_extent = 1.15 * max(float(np.linalg.norm(position)), 58.0)
    distance = 4.0 * half_extent

    views = []
    for name, eye_dir, cam_up in (("corridor A", side, up), ("corridor B", up, -side)):
        views.append(
            CameraView(
                name=name,
                camera_type="orthographic",
                position=target + distance * eye_dir,
                target=target,
                up=cam_up,
                ortho_half_extent=half_extent,
                width=size,
                height=size,
                near=1.0,
                far=2.0 * distance,
            )
        )
    return views


def compose(panels: list[tuple[str, np.ndarray]], caption: str) -> Image.Image:
    height = panels[0][1].shape[0]
    width = sum(frame.shape[1] for _, frame in panels)
    sheet = Image.new("RGB", (width, height + 54), (10, 10, 14))
    draw = ImageDraw.Draw(sheet)
    x = 0
    for title, frame in panels:
        sheet.paste(Image.fromarray(frame), (x, 0))
        draw.rectangle([x, 0, x + 2, height], fill=(10, 10, 14))
        draw.text((x + 10, 8), title, fill=(240, 240, 240))
        x += frame.shape[1]
    for i, line in enumerate(caption.split("\n")):
        draw.text((10, height + 10 + 17 * i), line, fill=(205, 205, 205))
    return sheet


@app.command()
def main(
    out: Path = typer.Option(Path("logs/dock_preview"), help="Output directory."),
    standoff: Optional[float] = typer.Option(None, help="Metres along the outward normal; default is each port's own standoff_m."),
    size: int = typer.Option(512, help="Panel edge in pixels."),
    ports: Optional[list[str]] = typer.Option(None, "--port", help="Port names, or 'all'; default is all."),
    include_shipped_dock: bool = typer.Option(True, help="Also render the pose from DockConfig."),
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if ports:
        try:
            names = resolve_port_names(tuple(ports))
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        selected = [PORTS_BY_NAME[name] for name in names]
    else:
        selected = list(PORTS)

    cfg = ISSConfig()
    centers, half_extents = load_collision_boxes(cfg.physics.collision_boxes_path)
    renderer = ISSRenderer(RenderConfig(image_width=size, image_height=size))
    scene = renderer._iss_scene.scene

    rows: list[tuple[str, DockingPort, np.ndarray, np.ndarray, float]] = []
    if include_shipped_dock:
        rows.append(
            (
                "shipped_dock_config",
                PORTS_BY_NAME["harmony_fwd_pma2"],
                np.asarray(cfg.dock.position, dtype=float),
                np.asarray(cfg.dock.quaternion, dtype=float),
                float("nan"),
            )
        )
    for port in selected:
        position, quat = port_pose(port, standoff)
        rows.append((port.name, port, position, quat, port.standoff_m if standoff is None else standoff))

    sheets: list[Image.Image] = []
    for name, port, position, quat, used_standoff in rows:
        normal = _unit(np.asarray(port.normal, dtype=float))
        views = corridor_views(position, normal, size)
        scale = 0.022 * views[0].ortho_half_extent

        state = goal_state(position, quat)
        # The goal marker sits on the chaser itself, so the onboard view is
        # rendered before the overlay goes in.
        panels = [("chaser view", renderer.render(state, None, view="DRAGON_FPV"))]

        overlay = build_overlay(port, position, scale)
        scene.add(overlay)
        with station_only(scene):
            panels += [(v.name, renderer.render_view(state, v)) for v in views]
        panels.append(("context", renderer.render(state, None, view="ISS_ISO")))
        scene.remove(overlay)

        gap = clearance(position, centers, half_extents)
        caption = (
            f"{name}   mechanism={port.mechanism}   module={port.module}\n"
            f"position=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})   "
            f"quaternion=({quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f})   "
            f"interface=({port.interface[0]:.3f}, {port.interface[1]:.3f}, {port.interface[2]:.3f})   "
            f"normal=({port.normal[0]:+.0f}, {port.normal[1]:+.0f}, {port.normal[2]:+.0f})   "
            f"standoff={used_standoff:.2f} m   "
            f"clearance={gap:.2f} m vs chaser radius {cfg.physics.dragon_collision_radius_m:.2f} m"
        )
        sheet = compose(panels, caption)
        sheet.save(out / f"{name}.png")
        sheets.append(sheet)
        typer.echo(f"{name:22s} clearance {gap:6.2f} m -> {out / (name + '.png')}")

    contact = Image.new(
        "RGB", (sheets[0].width // 3, sum(s.height // 3 for s in sheets)), (10, 10, 14)
    )
    y = 0
    for sheet in sheets:
        small = sheet.resize((sheet.width // 3, sheet.height // 3), Image.LANCZOS)
        contact.paste(small, (0, y))
        y += small.height
    contact.save(out / "contact_sheet.png")

    lines = ["# generated by scripts/preview_dock_poses.py", ""]
    for name, _, position, quat, _ in rows:
        lines += [
            f"# {name}",
            "[dock]",
            f"position = [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]",
            f"quaternion = [{quat[0]:.7f}, {quat[1]:.7f}, {quat[2]:.7f}, {quat[3]:.7f}]",
            "",
        ]
    (out / "dock_targets.toml").write_text("\n".join(lines))
    renderer.close()


if __name__ == "__main__":
    app()
