"""Camera views: a declarative CameraView config and a pygfx camera builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pygfx as gfx
import pylinalg as la

CameraType = Literal["perspective", "orthographic"]


@dataclass(frozen=True)
class CameraView:
    name: str
    camera_type: CameraType
    position: np.ndarray  # (3,)
    target: np.ndarray  # (3,)
    up: np.ndarray  # (3,)
    fov_y_deg: float = 60.0
    ortho_half_extent: float = 10.0
    width: int = 512
    height: int = 512
    # To allow defining the near/far planes if needed
    near: float | None = None
    far: float | None = None


def _quat_look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)

    f = target - eye
    f_norm = np.linalg.norm(f)
    up_norm = np.linalg.norm(up)

    if f_norm < 1e-8 or up_norm < 1e-8:
        return la.quat_from_euler((0.0, 0.0, 0.0), order="XYZ")

    f /= f_norm
    up /= up_norm

    z_axis = -f
    x_axis = np.cross(up, z_axis)
    x_norm = np.linalg.norm(x_axis)

    if x_norm < 1e-8:
        if abs(z_axis[2]) < 0.9:
            fallback_up = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            fallback_up = np.array([0.0, 1.0, 0.0], dtype=float)
        x_axis = np.cross(fallback_up, z_axis)
        x_norm = np.linalg.norm(x_axis)

    x_axis /= max(x_norm, 1e-8)
    y_axis = np.cross(z_axis, x_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    quat = la.quat_from_mat(R)
    return quat


def make_camera(view: CameraView) -> gfx.Camera:
    if view.camera_type == "perspective":
        cam = gfx.PerspectiveCamera(
            fov=view.fov_y_deg,
            aspect=view.width / view.height,
        )
    elif view.camera_type == "orthographic":
        e = float(view.ortho_half_extent)
        width_world = 2.0 * e
        height_world = 2.0 * e
        cam = gfx.OrthographicCamera(width_world, height_world, maintain_aspect=False)
    else:
        raise ValueError(f"unknown camera_type: {view.camera_type!r}")

    if view.near is not None or view.far is not None:
        # Keep sensible defaults if only one is provided
        near = float(view.near) if view.near is not None else 0.01
        far = float(view.far) if view.far is not None else 1e3
        # Pygfx uses depth_range for clipping
        cam.depth_range = (near, far)

    eye = np.asarray(view.position, dtype=float)
    target = np.asarray(view.target, dtype=float)
    up = np.asarray(view.up, dtype=float)

    cam.local.position = tuple(eye.tolist())
    quat = _quat_look_at(eye, target, up)
    cam.local.rotation = quat
    return cam
