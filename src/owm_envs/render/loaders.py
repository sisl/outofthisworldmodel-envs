"""Asset loaders: GLB scenes and cubemap skyboxes."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pygfx as gfx


def load_cubemap_from_faces(
    directory: str | Path,
    *,
    ext: str = "png",
    use_rgba: bool = False,
) -> gfx.Texture:
    """
    Load a skybox cubemap from 6 images in `directory` named:
      px.ext, nx.ext, py.ext, ny.ext, pz.ext, nz.ext

    Returns
    -------
    gfx.Texture
        Cubemap texture suitable for gfx.BackgroundSkyboxMaterial(map=tex)
    """
    d = Path(directory)
    names = ["px", "nx", "py", "ny", "pz", "nz"]  # expected cubemap face order
    faces = []

    for name in names:
        path = d / f"{name}.{ext}"

        # If you can't find it, raise an error
        if not path.exists():
            raise FileNotFoundError(f"Cubemap face not found: {path}")

        im = iio.imread(path)

        if im.ndim != 3:
            raise ValueError(f"{path} expected HxWxC, got {im.shape}")

        if use_rgba:
            if im.shape[2] == 3:
                alpha = 255 if im.dtype == np.uint8 else 1.0
                im = np.concatenate([im, np.full((*im.shape[:2], 1), alpha, dtype=im.dtype)], axis=2)
            im = im[..., :4]
        else:
            im = im[..., :3]

        faces.append(im)

    # Validate all faces same square size
    h, w = faces[0].shape[:2]
    if h != w:
        raise ValueError(f"Cubemap faces must be square, got {h}x{w} for {names[0]}")

    for name, im in zip(names, faces):
        if im.shape[:2] != (h, w):
            raise ValueError(f"Face {name} shape mismatch: got {im.shape[:2]}, expected {(h, w)}")

    arr = np.stack(faces, axis=0)  # (6, N, N, C)
    tex = gfx.Texture(arr, dim=2, size=(w, h, 6))
    return tex


def load_glb_scene(
    path: str | Path,
    *,
    scale: float = 1.0,
    cast_shadow: bool = False,
    receive_shadow: bool = False,
) -> gfx.Group:
    """Load a GLB (or any scene file supported by gfx.utils.load.load_scene).

    Applies a uniform `scale` and recursively sets shadow flags on the result.
    """
    scene_obj = gfx.utils.load.load_scene(str(path))
    scene_obj.local.scale = (scale, scale, scale)
    set_shadow_flags_recursive(scene_obj, cast_shadow=cast_shadow, receive_shadow=receive_shadow)
    return scene_obj


def set_shadow_flags_recursive(
    obj: gfx.WorldObject,
    *,
    cast_shadow: bool = True,
    receive_shadow: bool = True,
) -> None:
    """
    Best-effort: recursively set cast_shadow / receive_shadow flags on meshes.
    """
    if hasattr(obj, "cast_shadow"):
        try:
            obj.cast_shadow = cast_shadow  # type: ignore[attr-defined]
        except Exception:
            pass
    if hasattr(obj, "receive_shadow"):
        try:
            obj.receive_shadow = receive_shadow  # type: ignore[attr-defined]
        except Exception:
            pass

    for child in getattr(obj, "children", []) or []:
        set_shadow_flags_recursive(child, cast_shadow=cast_shadow, receive_shadow=receive_shadow)
