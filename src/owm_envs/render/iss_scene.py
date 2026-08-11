"""The ISS docking scene: station, capsule, Earth, Moon, and starfield.

Builds a static pygfx scene graph once and re-poses the Dragon capsule from
the simulation state on every call to `ISSScene.update`. A caller with real
ephemeris can pass `Lighting` to that same call and move the sun, Earth and
Moon with it; without it they stay where the config put them. Camera views and
the force/torque debug overlays are built on top of this scene elsewhere.

The station asset is the ISS as it stood between February and May 2015: CATS
was installed on the JEM exposed facility in January 2015 and the PMM was
relocated off Unity nadir that May, and the model shows both. It therefore
predates BEAM (2016), PMA-3's move to Harmony zenith (2017), the Bishop
airlock (2020), Nauka and Prichal (2021) and the iROSA arrays (2021), and it
still carries Pirs, which was deorbited in 2021. Two later JEM payloads,
ECOSTRESS (2018) and OCO-3 (2019), are fitted anachronistically. Anything
that reads a port or an airlock off this geometry is reading the 2015
configuration.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import imageio.v3 as iio
import jax.numpy as jnp
import numpy as np
import pygfx as gfx
import pylinalg as la
from PIL import Image
from pydantic import model_validator
from pygfx.renderers.wgpu import get_shared

from owm_envs.core.models import ConfigModel
from owm_envs.core.quaternion import quat_to_rotmat
from owm_envs.render import asset_path
from owm_envs.render.atmosphere import atmosphere_shell
from owm_envs.render.earth import MAP_WIDTHS, earth_texture_path
from owm_envs.render.inputs import Lighting
from owm_envs.render.iss_frame import ISS_RECENTRE_OFFSET, UPRIGHT_EULER_XYZ
from owm_envs.render.loaders import load_cubemap_from_faces, load_glb_scene


class RenderConfig(ConfigModel):
    """Graphics settings for the ISS docking scene."""

    image_width: int = 512
    image_height: int = 512

    # Filename under resources/international-space-station. A variant with
    # different visiting vehicles must place the station identically; check a
    # new asset with scripts/check_iss_asset.py before pointing this at it.
    iss_asset: str = "ISS_base.glb"
    iss_recentre_offset: tuple[float, float, float] = ISS_RECENTRE_OFFSET

    earth_radius_m: float = 6_378_137.0
    iss_altitude_m: float = 420_000.0
    # Geographic lon/lat of the sub-satellite point: the globe is oriented so
    # this surface point lies directly under the scene origin.
    earth_subpoint_lon_deg: float = -122.1697
    earth_subpoint_lat_deg: float = 37.4275
    show_earth_glow: bool = True
    # Brightness of the atmospheric limb, and how sharply it fades with
    # altitude -- `earth_glow_falloff` stands in for a scale height, so larger
    # is a tighter, brighter rim.
    #
    # `earth_atmosphere_scale` is where the air ends, as a multiple of the
    # surface radius. 1.02 puts it 128 km up, a little above the Karman line
    # and close to the depth of the blue band the limb actually shows. It has
    # to stay BELOW the station's own radius: the shader shades the air still
    # in front of the camera, so a shell the camera flies inside stops being a
    # rim at the horizon and becomes a wash over the whole sky.
    earth_glow_strength: float = 0.9
    earth_glow_falloff: float = 3.0
    earth_atmosphere_scale: float = 1.020
    show_earth_clouds: bool = True
    earth_cloud_altitude_m: float = 12_000.0
    earth_cloud_opacity: float = 0.8
    # Relief shading. `earth_bump_strength` scales the height gradients the
    # normal map is built from; `earth_normal_scale` scales its effect in the
    # shader, so terrain relief can be tuned without re-deriving the map.
    show_earth_bump: bool = True
    earth_bump_strength: float = 8.0
    earth_normal_scale: float = 2.5

    moon_radius_m: float = 1_737_400.0
    moon_asset_radius_units: float = 1.2718640565872192
    earth_moon_distance_m: float = 384_400_000.0
    moon_direction_from_earth_world: tuple[float, float, float] = (0.0, 1.0, 0.0)

    # Exposure multiplier on the starfield, in linear light: 1.0 is the map at
    # full strength, 0.0 a black sky. The map is a survey composite, exposed to
    # show the Milky Way rather than as a camera pointed at a sunlit station
    # would see it -- that camera stops down for the Earth and the hull, and
    # the sky behind them goes nearly black. 0.3 keeps the band faintly legible
    # without competing with the lit scene.
    star_brightness: float = 0.3

    directional_light_intensity: float = 8.0
    sun_direction_world: tuple[float, float, float] = (1.0, -0.3, 0.5)
    sun_visual_distance_m: float = 1_000_000.0
    sun_angular_diameter_deg: float = 0.53

    # Near planes are what bound how far a camera can see, not the far planes:
    # see `_MAX_DEPTH_RANGE_RATIO` in `owm_envs.render.renderer`. The FPV
    # camera is squeezed from both sides. Measured over the limb framing at
    # 128x128, not derived:
    #
    #   near   surface   atmosphere   hull kept
    #   0.02     11.1%       3.6%      100%  (reference)
    #   0.20    100%       100%         98.0%
    #   0.30    100%       100%         98.1%
    #   0.40    100%       100%         98.1%
    #   0.45    100%       100%         87.1%
    #   0.50    100%       100%         65.3%
    #
    # Below ~0.20 m the depth range stops reaching Earth's limb at 2.35e6 m and
    # the planet is cut off just inside it; above ~0.40 m it eats the capsule's
    # own nose cone, whose nearest surface inside this camera's frustum is
    # 0.4026 m away. 0.3 m sits in the middle of that window: 1.5x clear of the
    # planet bound, 1.34x clear of the hull.
    #
    # The window used to be far tighter -- effectively the single point 0.40 --
    # because the atmospheric glow shells reach much further out than the
    # surface does and were being clipped away below that. Comparing their
    # depth on `<=` instead of `<` (see `_GLOW_QUEUE`) costs nothing and
    # retires that constraint, leaving only the surface and the hull, which
    # are a factor of two apart.
    #
    # The STATION is not the binding constraint: the closest an FPV camera came
    # to ISS geometry over a full run of docking episodes was 2.19 m, 6x clear.
    # This near plane serves ISS_FPV too, which is unaffected either way -- it
    # is mounted on the zenith side looking away from the station, with no ISS
    # geometry inside its frustum at all, and its far plane still clears the
    # limb 2.5x.
    scene_camera_near_m: float = 5.0
    scene_camera_far_m: float = 1_000_000.0
    fpv_camera_near_m: float = 0.3
    fpv_camera_far_m: float = 1_000_000.0

    dragon_iso_offset_world: tuple[float, float, float] = (31.5, -31.5, 24.5)
    dragon_iso_fov_y_deg: float = 28.0
    dragon_top_height_m: float = 57.75
    dragon_top_fov_y_deg: float = 24.0
    dragon_fpv_offset_body: tuple[float, float, float] = (0.0, 1.75, 1.55)
    dragon_fpv_target_distance_m: float = 500.0
    dragon_fpv_fov_y_deg: float = 82.0

    iss_iso_position_world: tuple[float, float, float] = (340.0, -340.0, 265.0)
    iss_iso_fov_y_deg: float = 18.2
    iss_top_height_m: float = 440.0
    iss_top_fov_y_deg: float = 22.4
    iss_fpv_position_iss: tuple[float, float, float] = (0.225, 0.6, 16.2)
    iss_fpv_forward_iss: tuple[float, float, float] = (0.0, 0.0, 1.0)
    iss_fpv_up_iss: tuple[float, float, float] = (0.0, 1.0, 0.0)
    iss_fpv_target_distance_m: float = 500.0
    iss_fpv_fov_y_deg: float = 82.0

    @model_validator(mode="after")
    def _atmosphere_stays_below_the_station(self) -> RenderConfig:
        """The limb shader shades the air still in front of the camera, so a
        camera inside the shell sees a wash over the whole sky instead of a rim
        at the horizon. The scene only ever puts a camera at the station's own
        radius, so the shell has to end below it."""
        outer = self.earth_atmosphere_scale * self.earth_radius_m
        station = self.earth_radius_m + self.iss_altitude_m
        if self.show_earth_glow and outer >= station:
            raise ValueError(
                f"earth_atmosphere_scale {self.earth_atmosphere_scale} puts the top "
                f"of the atmosphere at {outer:.4g} m, at or above the station's "
                f"{station:.4g} m; the limb would wash over the whole sky"
            )
        return self


def _unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n >= eps else np.zeros_like(v)


def _strip_embedded_extras(scene_obj: gfx.WorldObject) -> gfx.Group:
    """Loaded GLB scenes carry whatever camera/light/background nodes the
    exporter embedded alongside the actual geometry. Keep only the geometry
    so it can be parented into this scene's own lighting and camera setup."""
    group = gfx.Group()
    for child in list(getattr(scene_obj, "children", []) or []):
        if isinstance(child, (gfx.Camera, gfx.Light, gfx.Background)):
            continue
        group.add(child)
    return group


def _collect_meshes(obj: gfx.WorldObject) -> list[gfx.Mesh]:
    meshes: list[gfx.Mesh] = []
    for child in getattr(obj, "children", []) or []:
        if isinstance(child, gfx.Mesh):
            meshes.append(child)
        meshes.extend(_collect_meshes(child))
    return meshes


def _mesh_positions(mesh: gfx.Mesh) -> np.ndarray | None:
    geom = getattr(mesh, "geometry", None)
    pos = getattr(geom, "positions", None) if geom is not None else None
    data = getattr(pos, "data", None) if pos is not None else None
    if data is None:
        return None
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        return None
    return arr


def _hide_large_flat_helper_meshes(
    obj: gfx.WorldObject, *, min_planar_extent: float = 8.0, max_thickness: float = 0.05
) -> None:
    """Some source meshes embed a large flat helper plane (e.g. a clipping or
    reference panel) alongside the real geometry. Hide anything that is
    nearly flat (thin along one axis) and large in the other two."""
    for mesh in _collect_meshes(obj):
        arr = _mesh_positions(mesh)
        if arr is None:
            continue
        extent = np.sort(arr.max(axis=0) - arr.min(axis=0))
        if extent[0] <= max_thickness and extent[1] >= min_planar_extent and extent[2] >= min_planar_extent:
            mesh.visible = False


def _visible_geometry_center(obj: gfx.WorldObject) -> np.ndarray | None:
    """Average vertex position across all visible meshes, in `obj`'s local
    frame. Loaded assets are re-centered on this point so the group origin --
    which the simulation state positions and rotates about -- sits at the
    mesh's own center rather than wherever the exporter placed it."""
    total = np.zeros(3, dtype=np.float64)
    count = 0

    def visit(node: gfx.WorldObject, parent_matrix: np.ndarray) -> None:
        nonlocal total, count
        matrix = parent_matrix @ np.asarray(node.local.matrix, dtype=np.float32)
        if isinstance(node, gfx.Mesh) and bool(getattr(node, "visible", True)):
            arr = _mesh_positions(node)
            if arr is not None:
                homogeneous = np.concatenate([arr.astype(np.float64), np.ones((arr.shape[0], 1))], axis=1)
                world = (matrix.astype(np.float64) @ homogeneous.T).T[:, :3]
                total += world.sum(axis=0)
                count += world.shape[0]
        for child in getattr(node, "children", []) or []:
            visit(child, matrix)

    visit(obj, np.eye(4, dtype=np.float32))
    if count == 0:
        return None
    return (total / count).astype(np.float32)


# Pillow warns above 89.5 Mpx and refuses outright above twice that, on the
# assumption that a file that big is a decompression bomb aimed at whoever
# opens it. The Earth maps are first-party assets baked by
# `owm_envs.render.downsample` to the widths in `MAP_WIDTHS` -- the widest is
# 16384x8192 = 134 Mpx -- so the cap is raised to exactly what those can be
# rather than removed: an asset that outgrows the widths this package itself
# produces should still trip the check.
_MAX_MAP_PIXELS = max(MAP_WIDTHS.values()) * (max(MAP_WIDTHS.values()) // 2)


def _read_map(path: Path) -> np.ndarray:
    """Read one of the Earth maps, whatever its pixel count.

    Pillow's cap is a process-wide global, so raising it here raises it for
    everything in the process; `None` means a caller has already lifted it
    entirely and must not be walked back.
    """
    if Image.MAX_IMAGE_PIXELS is not None:
        Image.MAX_IMAGE_PIXELS = max(Image.MAX_IMAGE_PIXELS, _MAX_MAP_PIXELS)
    return np.asarray(iio.imread(path))


# Draw order for the Earth's cloud deck, and for what has to be behind it.
#
# The deck is a shell 12 km above the surface, which at orbital viewing
# distances is far below what a float32 depth buffer can tell apart: one depth
# step is `d**2 / (near * 2**24)`, about 21 km at the 420 km nadir range and
# 650 km at the limb. Depth-testing the shell against the surface is therefore
# rounding noise -- it discarded ~94% of the deck and flipped which pixels
# survived as the camera moved. No near plane fixes it either: resolving 12 km
# at the limb would need a near plane of ~27 m, which would clip the station
# away during a dock.
#
# So the deck is composited by draw order instead, which does not depend on
# precision at all. Geometry puts it strictly in front of everything below it
# and strictly behind the station, so it is drawn between the two with no depth
# test and no depth write of its own. pygfx sorts on
# `(material.render_queue, object.render_order, distance)`; the station and
# capsule keep pygfx's default queue of 2600 and so paint over the deck
# normally, using the depth the surface wrote.
#
# "Strictly behind the station" is an assumption about this scene rather than
# something the sort key can check: it holds because the deck tops out at
# `earth_cloud_altitude_m` (12 km) while everything the simulation flies stays
# near the station's 420 km. A capsule flown down through the deck, or a config
# that raised the deck above the station, would be drawn on the wrong side of
# it -- both are outside what this environment produces.
_DISTANT_QUEUE = 2000  # Earth's surface, the Sun, the Moon -- all below the deck
_CLOUD_QUEUE = 2100  # the deck, over them
_GLOW_QUEUE = 2200  # the atmospheric limb, over the deck

# The rim's colour: a Rayleigh-scattered blue, brighter and less saturated than
# the sky from the ground because the path this shades is a limb path through
# the whole atmosphere rather than a vertical one.
EARTH_GLOW_COLOR = (0.56, 0.87, 1.0, 1.0)

# The fill light is a fixed fraction of the key light, which is what makes it
# a fill and not a second sun: it keeps the shadowed side readable rather than
# black. Held here because per-frame eclipse dimming has to scale both by the
# same factor -- a fill left at full strength would light a station the sun no
# longer reaches. The one term that does survive an eclipse is the ambient,
# which is deliberate rather than an omission here: see `_build_ambient_light`.
_FILL_INTENSITY_FRACTION = 0.35

# Photosphere map for the sun disc, under resources/sun. Equirectangular, so
# it lands on `sphere_geometry`'s UVs without reprojection.
SUN_TEXTURE = "sun_2k.jpg"


def _max_texture_size() -> int:
    return int(get_shared().device.limits["max-texture-dimension-2d"])


def _fit_to_device_limit(arr: np.ndarray, path: Path) -> np.ndarray:
    """Halve an image until it fits the adapter's 2D texture limit.

    The full-globe maps are 16384 px wide, which plenty of adapters refuse:
    8192 is a common `max-texture-dimension-2d`. Creating the texture anyway
    fails at draw time with a wgpu validation error rather than anything a
    caller can act on, so shrink here and say so.
    """
    limit = _max_texture_size()
    height, width = arr.shape[:2]
    if max(width, height) <= limit:
        return arr

    factor = 2
    while max(width // factor, height // factor) > limit:
        factor *= 2
    size = (max(width // factor, 1), max(height // factor, 1))
    warnings.warn(
        f"{path.name} is {width}x{height}, above this device's {limit} px texture limit; "
        f"downscaled to {size[0]}x{size[1]}"
    )
    return np.asarray(Image.fromarray(arr).resize(size, Image.LANCZOS))


def _load_rgb_texture(path: Path) -> gfx.Texture:
    arr = _fit_to_device_limit(_read_map(path), path)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    arr = np.ascontiguousarray(arr[..., :3].astype(np.uint8))
    return gfx.Texture(arr, dim=2, colorspace="srgb", generate_mipmaps=True)


def _load_cloud_texture(path: Path, *, opacity: float) -> gfx.Texture:
    """The cloud map is stored as plain RGB; its brightness is coverage, so
    turn that into the alpha channel of a white RGBA texture."""
    arr = _fit_to_device_limit(_read_map(path), path)
    brightness = arr[..., :3].astype(np.float32).mean(axis=-1) if arr.ndim == 3 else arr.astype(np.float32)
    alpha = np.clip(np.rint(brightness * float(opacity)), 0.0, 255.0).astype(np.uint8)
    rgb = np.full(alpha.shape + (3,), 255, dtype=np.uint8)
    rgba = np.ascontiguousarray(np.concatenate([rgb, alpha[..., None]], axis=-1))
    return gfx.Texture(rgba, dim=2, colorspace="srgb", generate_mipmaps=True)


def _normal_map_from_height(height: np.ndarray, strength: float) -> np.ndarray:
    """Equirectangular height field (H, W) in [0, 1] -> tangent-space normals
    (H, W, 3) in [-1, 1].

    Longitude is periodic: the first and last columns are the same meridian,
    so the x-gradient is taken across the wrap. Letting `np.gradient` fall back
    to a one-sided difference there tilts the antimeridian's normals by up to
    48 degrees wherever it crosses relief, which lights as a seam.

    The y component is +dh/dv, not -dh/dv: pygfx's `getTangentFrame` returns
    `mat3x3f(T, -B, N)` -- a negated bitangent, the glTF flipped-green
    convention -- so the shader already subtracts this channel.
    """
    h = np.asarray(height, dtype=np.float32)
    dy = np.gradient(h, axis=0)
    dx = np.gradient(np.pad(h, ((0, 0), (1, 1)), mode="wrap"), axis=1)[:, 1:-1]
    normals = np.dstack([-dx * strength, dy * strength, np.ones_like(h)])
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals.astype(np.float32)


def _earth_normal_map(path: Path, *, show: bool, strength: float) -> gfx.TextureMap | None:
    """Earth's bump map as a shader-ready tangent-space normal map.

    Returns None when relief shading is off or the map is absent -- a missing
    texture costs some surface detail, which is no reason to refuse to render.
    """
    if not show:
        return None
    if not path.exists():
        warnings.warn(f"earth bump map missing at {path}; rendering without a normal map")
        return None

    arr = _fit_to_device_limit(_read_map(path), path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    normals = _normal_map_from_height(arr.astype(np.float32) / 255.0, strength)
    # mesh.wgsl decodes the sample as `2 * s - 1`, so encode the other way and
    # keep the texture linear: an sRGB transfer curve would bend every normal.
    encoded = np.rint((normals * 0.5 + 0.5) * 255.0).astype(np.uint8)
    return _texture_map(gfx.Texture(encoded, dim=2, colorspace="physical", generate_mipmaps=True))


def _texture_map(texture: gfx.Texture) -> gfx.TextureMap:
    return gfx.TextureMap(texture, filter="linear", wrap="repeat")


# ECEF axes (+X through the prime meridian at the equator, +Z north) onto the
# axes an untilted globe geometry is built on (+Z through the prime meridian,
# +Y north). Applied before the subpoint tilt in `_globe_frame_from_ecef`, and
# undone in `ISSScene._apply_earth_surface_rotation` so a frame carrying real
# ephemeris can pose the globe in ECEF terms without knowing this convention.
_GLOBE_FROM_ECEF_AXES = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])


def _globe_frame_from_ecef(subpoint_lon_deg: float, subpoint_lat_deg: float) -> np.ndarray:
    """R mapping ECEF axes onto the globe geometry's own axes, for a globe
    built with `(subpoint_lon_deg, subpoint_lat_deg)` on local +Z."""
    # Spin the subpoint's meridian onto lon 0 (about +Y), then its parallel
    # down to the equator (about +X); +Z then points at the subpoint.
    cos_lon0, sin_lon0 = np.cos(np.deg2rad(subpoint_lon_deg)), np.sin(np.deg2rad(subpoint_lon_deg))
    cos_lat0, sin_lat0 = np.cos(np.deg2rad(subpoint_lat_deg)), np.sin(np.deg2rad(subpoint_lat_deg))
    about_y = np.array([[cos_lon0, 0.0, -sin_lon0], [0.0, 1.0, 0.0], [sin_lon0, 0.0, cos_lon0]])
    about_x = np.array([[1.0, 0.0, 0.0], [0.0, cos_lat0, -sin_lat0], [0.0, sin_lat0, cos_lat0]])
    return about_x @ about_y @ _GLOBE_FROM_ECEF_AXES


def _earth_globe_geometry(
    *,
    radius: float,
    subpoint_lon_deg: float,
    subpoint_lat_deg: float,
    width_segments: int = 512,
    height_segments: int = 256,
) -> gfx.Geometry:
    """A full sphere carrying equirectangular UVs, rotated so the geographic
    point `(subpoint_lon_deg, subpoint_lat_deg)` sits on local +Z with north
    at +Y -- the axis the scene places directly under its origin."""
    u = np.linspace(0.0, 1.0, width_segments + 1, dtype=np.float32)
    v = np.linspace(0.0, 1.0, height_segments + 1, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    lon = (uu - 0.5) * (2.0 * np.pi)
    lat = (0.5 - vv) * np.pi
    cos_lat, sin_lat = np.cos(lat), np.sin(lat)
    unit_ecef = np.stack(
        [cos_lat * np.cos(lon), cos_lat * np.sin(lon), sin_lat], axis=-1
    ).reshape(-1, 3)

    frame = _globe_frame_from_ecef(subpoint_lon_deg, subpoint_lat_deg)
    normals = (unit_ecef @ frame.T).astype(np.float32)
    positions = (float(radius) * normals).astype(np.float32)
    texcoords = np.stack([uu, vv], axis=-1).reshape(-1, 2).astype(np.float32)

    cols = width_segments + 1
    rows = np.arange(height_segments, dtype=np.uint32)[:, None] * cols
    columns = np.arange(width_segments, dtype=np.uint32)[None, :]
    a = rows + columns
    b, c, d = a + 1, a + cols, a + cols + 1
    indices = np.stack([a, c, b, b, c, d], axis=-1).reshape(-1, 3)

    return gfx.Geometry(positions=positions, normals=normals, texcoords=texcoords, indices=indices)


class ISSScene:
    """The pygfx scene graph for the ISS docking environment: the station,
    the Dragon capsule, Earth, the Moon, and the starfield background."""

    def __init__(self, cfg: RenderConfig, *, download_textures: bool = True) -> None:
        self.cfg = cfg

        # How the globe's baked geometry relates to ECEF, so a frame carrying
        # `Lighting.earth_rotation_world` -- which speaks ECEF -- can pose it
        # without knowing where the configured subpoint put the texture.
        self._globe_frame_from_ecef = _globe_frame_from_ecef(
            cfg.earth_subpoint_lon_deg, cfg.earth_subpoint_lat_deg
        )

        # Resolved once and held: these calls may fetch and downsample a
        # multi-gigabyte source, and a per-frame lookup would also repeat a
        # failed network round-trip on every frame. `download_textures` is
        # cleared by a render worker, whose parent resolved all three before
        # the pool started -- a worker that fetched its own would restore the
        # race that resolution removed.
        self._earth_color_path = earth_texture_path("color", allow_download=download_textures)
        self._earth_clouds_path = earth_texture_path("clouds", allow_download=download_textures)
        self._earth_bump_path = earth_texture_path("bump", allow_download=download_textures)

        # Loaded assets face +Y; rotate them onto this environment's body +Z
        # so the capsule's nose and the station's long axis agree with the
        # simulation's convention.
        self._upright_quat = la.quat_from_euler(np.array(UPRIGHT_EULER_XYZ, dtype=np.float32), order="xyz")

        self.background = gfx.Scene()
        starmap = load_cubemap_from_faces(asset_path("nasa_starmap_2020"), ext="png")
        star_material = gfx.BackgroundSkyboxMaterial(map=starmap)
        # The skybox shader ignores `material.color` -- it samples the cubemap
        # straight into the output -- but it does scale by `opacity`, and the
        # background blends over a buffer already cleared to black, in linear
        # light. So opacity is the exposure multiplier, exactly: measured, 0.5
        # and 0.3 come back as 0.506x and 0.299x the linear radiance. Scaling
        # the texels instead would cost a copy of the cubemap and a gamma
        # round-trip for the same result.
        star_material.opacity = cfg.star_brightness
        self.background.add(gfx.Background(None, star_material))

        self._earth_surface_group = gfx.Group()

        self.scene = gfx.Scene()
        self.iss = self._load_iss_group()
        self.scene.add(self.iss)
        # Held as attributes, not just added and forgotten: `update` moves
        # every one of them when it is handed a `Lighting`. Earth is built
        # first because the moon is placed relative to its centre.
        self._earth_group = self._load_earth_group()
        # The moon is 3.9e8 m out, past what a near plane close enough to
        # render a dock can express in a float32 depth buffer, so it gets its
        # own scene and its own pass -- see `ISSRenderer._draw`. It carries its
        # own key light because a scene is what pygfx gathers lights from, and
        # a body at lunar distance wants the sun and nothing else anyway: the
        # ambient and fill that keep an eclipsed STATION readable have no
        # business brightening the moon, which is not in the Earth's shadow
        # when the station is.
        self.distant = gfx.Scene()
        self._moon_group = self._load_moon_group()
        self._moon_light = self._build_moon_light()
        self.distant.add(self._moon_group)
        self.distant.add(self._moon_light)
        self._sun = self._build_sun_sphere()
        ambient_light = self._build_ambient_light()
        self._directional_light = self._build_directional_light()
        self._fill_light = self._build_fill_directional_light()
        self.scene.add(self._earth_group)
        self.scene.add(self._sun)
        self.scene.add(ambient_light)
        self.scene.add(self._directional_light)
        self.scene.add(self._fill_light)

        # What `update(lighting=None)` means, held as values rather than as
        # "whatever is there now". A renderer outlives the frame -- one is
        # shared across every episode of a split, and the same process may
        # render an env that supplies ephemeris and one that does not -- so a
        # frame without ephemeris has to show this scene's static
        # configuration, not wherever the last lit frame left the sun. Every
        # placement `_apply_lighting` writes except the globe's attitude, whose
        # static value is the identity the geometry is already baked at.
        self._static_lighting = {
            "sun_position": tuple(self._sun.local.position),
            "key_position": tuple(self._directional_light.local.position),
            "key_intensity": self._directional_light.intensity,
            "fill_position": tuple(self._fill_light.local.position),
            "fill_intensity": self._fill_light.intensity,
            "earth_position": tuple(self._earth_group.local.position),
            "moon_position": tuple(self._moon_group.local.position),
            "moon_light_position": tuple(self._moon_light.local.position),
        }
        self._lit = False

        self.dragon = self._load_dragon_group()
        self.scene.add(self.dragon)

    def _load_dragon_group(self) -> gfx.Group:
        scene_obj = load_glb_scene(asset_path("spacex-dragon-capsule", "spacex_dragon_2_exterior.glb"))
        asset_group = _strip_embedded_extras(scene_obj)
        asset_group.local.rotation = self._upright_quat
        _hide_large_flat_helper_meshes(asset_group)

        center = _visible_geometry_center(asset_group)
        if center is not None:
            asset_group.local.position = tuple((-center).tolist())

        dragon_group = gfx.Group()
        dragon_group.add(asset_group)

        # The source mesh has no flat base; cap it so the capsule doesn't
        # read as hollow from below.
        bottom_cap = gfx.Mesh(
            gfx.cylinder_geometry(1.875, 1.875, 0.0375, 48, 1),
            gfx.MeshBasicMaterial(color=(0.5, 0.5, 0.5, 1.0)),
        )
        bottom_cap.local.position = (0.0, 0.0, -4.8)
        dragon_group.add(bottom_cap)

        return dragon_group

    def _load_iss_group(self) -> gfx.Group:
        scene_obj = load_glb_scene(asset_path("international-space-station", self.cfg.iss_asset))
        iss_group = _strip_embedded_extras(scene_obj)
        iss_group.local.rotation = self._upright_quat
        iss_group.local.position = tuple(-np.asarray(self.cfg.iss_recentre_offset, dtype=np.float32))
        return iss_group

    def _load_earth_group(self) -> gfx.Group:
        cfg = self.cfg
        earth_group = gfx.Group()
        earth_group.add(self._earth_surface_group)

        earth_tex = _load_rgb_texture(self._earth_color_path)
        earth_geom = _earth_globe_geometry(
            radius=cfg.earth_radius_m,
            subpoint_lon_deg=cfg.earth_subpoint_lon_deg,
            subpoint_lat_deg=cfg.earth_subpoint_lat_deg,
        )
        earth_mat = gfx.MeshStandardMaterial(
            map=_texture_map(earth_tex),
            normal_map=_earth_normal_map(
                self._earth_bump_path, show=cfg.show_earth_bump, strength=cfg.earth_bump_strength
            ),
            normal_scale=(cfg.earth_normal_scale, cfg.earth_normal_scale),
            roughness=1.0,
            metalness=0.0,
            emissive=(0.06, 0.06, 0.08),
            emissive_intensity=1.0,
        )
        earth_mat.render_queue = _DISTANT_QUEUE
        earth_mesh = gfx.Mesh(earth_geom, earth_mat)
        self._earth_surface_group.add(earth_mesh)

        if cfg.show_earth_glow and cfg.earth_glow_strength > 0.0:
            self._add_earth_glow(earth_group)

        if cfg.show_earth_clouds:
            cloud_tex = _load_cloud_texture(self._earth_clouds_path, opacity=cfg.earth_cloud_opacity)
            cloud_geom = _earth_globe_geometry(
                radius=cfg.earth_radius_m + cfg.earth_cloud_altitude_m,
                subpoint_lon_deg=cfg.earth_subpoint_lon_deg,
                subpoint_lat_deg=cfg.earth_subpoint_lat_deg,
            )
            cloud_mat = gfx.MeshBasicMaterial(map=_texture_map(cloud_tex))
            cloud_mat.alpha_mode = "blend"  # already implies depth_write=False
            cloud_mat.render_queue = _CLOUD_QUEUE
            cloud_mat.depth_test = False
            # Without a depth test the shell no longer hides its own far half,
            # and pygfx draws both sides of a mesh by default. Front faces are
            # exactly the near hemisphere seen from anywhere OUTSIDE the shell,
            # which is the only place this environment's cameras go: the deck
            # tops out 12 km up and the station orbits at 420 km. A camera
            # inside it would lose the deck rather than see it from below --
            # no static `side` is right for both faces of a surface, and that
            # vantage point is 400 km beneath the scene.
            cloud_mat.side = gfx.VisibleSide.front
            cloud_mesh = gfx.Mesh(cloud_geom, cloud_mat)
            self._earth_surface_group.add(cloud_mesh)

        earth_group.local.position = (0.0, 0.0, -(cfg.earth_radius_m + cfg.iss_altitude_m))
        self._apply_earth_surface_rotation(None)
        return earth_group

    def _add_earth_glow(self, earth_group: gfx.Group) -> None:
        """The atmospheric limb, as one screen-space pass.

        Parented to the globe, because `atmosphere.AtmosphereShader` reads the
        planet's centre off this object's own world transform -- so it follows
        when `_apply_lighting` moves the Earth to the chief's true altitude,
        with nothing per-frame to update.

        Drawn after the cloud deck and before the station, on `_GLOW_QUEUE`,
        and compared on `<=` for the same reason everything else at this range
        is: the rim reaches ~1e6 m, where a float32 depth rounds to exactly
        1.0 and would lose `<` against the cleared buffer. The station and the
        capsule are metres away, so their depths sit far below the rim's and
        occlude it normally.
        """
        cfg = self.cfg
        self._earth_glow = atmosphere_shell(
            surface_radius=cfg.earth_radius_m,
            outer_radius=cfg.earth_radius_m * cfg.earth_atmosphere_scale,
            color=EARTH_GLOW_COLOR,
            strength=cfg.earth_glow_strength,
            falloff=cfg.earth_glow_falloff,
            render_queue=_GLOW_QUEUE,
        )
        self._earth_glow.material.depth_compare = "<="
        earth_group.add(self._earth_glow)

    def _apply_earth_surface_rotation(self, earth_rotation_world: np.ndarray | None) -> None:
        """Point the globe at `earth_rotation_world` (ECEF axes -> world), or
        back to the configured subpoint when a frame carries no ephemeris.

        The surface and the cloud deck are the only children this reaches: the
        glow shells are spheres, so their orientation is not observable.
        """
        if earth_rotation_world is None:
            self._earth_surface_group.local.rotation = (0.0, 0.0, 0.0, 1.0)
            return
        matrix = np.asarray(earth_rotation_world, dtype=np.float64) @ self._globe_frame_from_ecef.T
        self._earth_surface_group.local.rotation = la.quat_from_mat(matrix)

    def _earth_center_world(self) -> np.ndarray:
        """Where the planet's centre currently sits. Read rather than
        recomputed from the config, so anything placed against it -- the moon
        -- follows when `update` moves the Earth to the chief's true altitude."""
        return np.asarray(self._earth_group.local.position, dtype=np.float32)

    def _load_moon_group(self) -> gfx.Group:
        cfg = self.cfg
        scene_obj = load_glb_scene(asset_path("moon", "moon_small.glb"))
        moon_group = _strip_embedded_extras(scene_obj)

        scale = cfg.moon_radius_m / max(cfg.moon_asset_radius_units, 1e-6)
        moon_group.local.scale = (scale, scale, scale)

        direction = _unit(np.array(cfg.moon_direction_from_earth_world, dtype=np.float32))
        moon_world = self._earth_center_world() + cfg.earth_moon_distance_m * direction
        moon_group.local.position = tuple(moon_world.tolist())
        for mesh in _collect_meshes(moon_group):
            mesh.material.render_queue = _DISTANT_QUEUE
            # Drawn between the starfield and the main scene, so everything in
            # that scene paints over it and occlusion by the Earth, the station
            # and the capsule comes out of draw order rather than out of a
            # depth comparison -- which could not work anyway, since the two
            # passes project depth through different near planes. Writing
            # depth here would leave values the main pass reads as nearer than
            # the Earth and hide the planet behind the moon.
            mesh.material.depth_write = False
            mesh.material.depth_test = False
            # With no depth test a sphere no longer hides its own far side, and
            # pygfx draws both faces by default. The moon is convex and the
            # camera is always outside it, so its front faces are exactly the
            # hemisphere that should be visible.
            mesh.material.side = gfx.VisibleSide.front
        return moon_group

    def _build_sun_sphere(self) -> gfx.Mesh:
        """The sun as a textured disc at `sun_visual_distance_m`.

        A proxy at a fictitious distance, not the sun where it really is: the
        radius is derived from that distance so the disc subtends
        `sun_angular_diameter_deg`, which is what the frame actually shows.
        `_apply_lighting` moves it along the sphere of that radius, never off
        it, so the angular size holds however the ephemeris points it.

        Basic-material, so nothing shades it: the photosphere emits rather than
        reflects, and a lit material would give it a terminator. The map is
        left at full white so its own colour comes through -- the texture is
        already the sun's, and tinting it again would apply that colour twice.
        """
        cfg = self.cfg
        distance = max(cfg.sun_visual_distance_m, 1.0)
        angular_radius_rad = 0.5 * np.deg2rad(cfg.sun_angular_diameter_deg)
        radius = max(distance * float(np.tan(angular_radius_rad)), 1.0)
        sun_mat = gfx.MeshBasicMaterial(
            map=_texture_map(_load_rgb_texture(asset_path("sun", SUN_TEXTURE))),
            color=(1.0, 1.0, 1.0, 1.0),
        )
        sun_mat.render_queue = _DISTANT_QUEUE
        # 64x32 rather than 32x16: the disc is a few pixels wide in the FPV
        # view but fills the frame of any narrow-field shot, where a 32-segment
        # silhouette reads as a polygon rather than a circle.
        sun = gfx.Mesh(
            gfx.sphere_geometry(radius=radius, width_segments=64, height_segments=32),
            sun_mat,
        )
        direction = _unit(np.array(cfg.sun_direction_world, dtype=np.float32))
        sun.local.position = tuple((direction * distance).tolist())
        return sun

    def _build_directional_light(self) -> gfx.DirectionalLight:
        cfg = self.cfg
        light = gfx.DirectionalLight("#ffffff", cfg.directional_light_intensity)
        direction = _unit(np.array(cfg.sun_direction_world, dtype=np.float32))
        light.local.position = tuple((direction * cfg.sun_visual_distance_m).tolist())
        light.cast_shadow = True
        light.shadow.map_size = (4096, 4096)
        light.shadow.camera.width = 600.0
        light.shadow.camera.height = 600.0
        light.shadow.camera.depth_range = (0.1, 2_000.0)
        light.shadow.bias = 0.0005
        return light

    def _build_moon_light(self) -> gfx.DirectionalLight:
        """The key light for the distant scene, held separately from the one
        that lights the station: pygfx gathers lights per scene, and this one
        is never dimmed for eclipse -- the moon is lit whether or not the
        station it is being viewed from is in the Earth's shadow."""
        light = gfx.DirectionalLight("#ffffff", self.cfg.directional_light_intensity)
        direction = _unit(np.array(self.cfg.sun_direction_world, dtype=np.float32))
        light.local.position = tuple((direction * self.cfg.sun_visual_distance_m).tolist())
        light.cast_shadow = False
        return light

    def _build_ambient_light(self) -> gfx.AmbientLight:
        # Held at full strength through an eclipse, unlike the key and fill
        # that `_apply_lighting` dims. A station in umbra is not in the dark:
        # the Earth fills half its sky and earthshine is what it is lit by.
        # This term stands in for that, and it is also the floor that keeps an
        # eclipsed frame from being a black image with nothing in it to learn
        # from. It is a flat approximation -- a real earthshine would swing
        # with the sunlit fraction of the disc below -- and dimming it with
        # the sun would be a worse one.
        return gfx.AmbientLight("#c7d8ff", 0.22)

    def _build_fill_directional_light(self) -> gfx.DirectionalLight:
        fill_direction = _unit(np.array([-0.55, 0.4, 0.9], dtype=np.float32))
        light = gfx.DirectionalLight(
            "#dfe9ff", _FILL_INTENSITY_FRACTION * self.cfg.directional_light_intensity
        )
        light.local.position = tuple((fill_direction * self.cfg.sun_visual_distance_m).tolist())
        light.cast_shadow = False
        return light

    def _apply_lighting(self, lighting: Lighting) -> None:
        """Point the sun, dim it for eclipse, and place Earth and the Moon.

        Everything here is per-frame ephemeris, replacing the static config
        placement the scene is built with.
        """
        cfg = self.cfg

        # At `sun_visual_distance_m` exactly, never further or nearer: the
        # disc's radius is baked from that distance in `_build_sun_sphere`, so
        # moving it along the sphere of that radius is what keeps its angular
        # diameter at `sun_angular_diameter_deg`. Nothing needs rescaling.
        sun_direction = _unit(np.asarray(lighting.sun_direction_world, dtype=np.float32))
        sun_world = sun_direction * cfg.sun_visual_distance_m
        self._sun.local.position = tuple(sun_world.tolist())

        # In umbra the disc is hidden, not merely unlit. Depth does not do
        # this for us: the proxy sits at `sun_visual_distance_m` (1e6 m) while
        # the Earth's surface along an oblique ray is further away than that,
        # so the disc draws IN FRONT of the night side. Measured over one ISS
        # orbit, 55 of 1115 samples -- 4.9% of the period, sun_dir_z ~ -0.47 --
        # would show the sun through the dark planet. Hiding it is exact
        # rather than a patch: the camera rides the chief, and this station
        # being in umbra is precisely the Earth standing between the two.
        self._sun.visible = bool(lighting.illumination > 0.0)

        # The shadow camera is not moved here: pygfx re-derives it from the
        # light's world position every frame (`LightShadow._update_matrix`).
        self._directional_light.local.position = tuple(sun_world.tolist())
        self._moon_light.local.position = tuple(sun_world.tolist())
        illumination = float(lighting.illumination)
        self._directional_light.intensity = illumination * cfg.directional_light_intensity
        self._fill_light.intensity = illumination * _FILL_INTENSITY_FRACTION * cfg.directional_light_intensity

        # The chief's true altitude, in place of the fixed `iss_altitude_m`.
        # The cloud deck and the glow shells are children of this group and
        # follow it; the moon is not, so it is placed off the new centre.
        self._earth_group.local.position = (0.0, 0.0, -float(lighting.chief_distance_m))
        moon_world = self._earth_center_world() + np.asarray(lighting.moon_vector_world, dtype=np.float32)
        self._moon_group.local.position = tuple(moon_world.tolist())

        # Which terrain is under the station. Most of what this rotation does
        # between frames is the chief's own motion around the planet, not the
        # planet's rotation: the world frame rides the chief.
        self._apply_earth_surface_rotation(lighting.earth_rotation_world)

    def _restore_static_lighting(self) -> None:
        """Undo `_apply_lighting`, back to the values built from the config."""
        static = self._static_lighting
        self._sun.local.position = static["sun_position"]
        self._sun.visible = True
        self._moon_light.local.position = static["moon_light_position"]
        self._directional_light.local.position = static["key_position"]
        self._directional_light.intensity = static["key_intensity"]
        self._fill_light.local.position = static["fill_position"]
        self._fill_light.intensity = static["fill_intensity"]
        self._earth_group.local.position = static["earth_position"]
        self._moon_group.local.position = static["moon_position"]
        self._apply_earth_surface_rotation(None)

    def update(
        self,
        state: np.ndarray,
        action: np.ndarray | None = None,
        lighting: Lighting | None = None,
    ) -> None:
        """Pose the Dragon capsule from a 13D state: 0:3 position, 6:10
        quaternion q_bw (body -> world), the rest unused here.

        `lighting` is this frame's ephemeris. `None` means the static
        configuration -- the sun direction, altitude and moon placement this
        scene was built with -- which is what every env but `iss-hcw` renders
        through. It is a statement about the frame, not an instruction to skip
        the lights: a scene a previous frame moved is put back. One renderer
        serves a whole split and can be handed frames from more than one env,
        so leaving them where they were would light an env that supplies no
        ephemeris with the last `iss-hcw` frame's sun.
        """
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        if s.shape[0] != 13:
            raise ValueError(f"expected a 13-element state, got shape {s.shape}")

        position = s[0:3]
        q_bw = s[6:10]
        rotation = np.asarray(quat_to_rotmat(jnp.asarray(q_bw)), dtype=np.float32)

        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = position
        self.dragon.local.matrix = matrix

        if lighting is not None:
            self._apply_lighting(lighting)
            self._lit = True
        elif self._lit:
            # Only when something moved them: a scene that has never been lit
            # is already at these values, and every frame of every env but
            # `iss-hcw` takes this branch.
            self._restore_static_lighting()
            self._lit = False


def iss_vertices_world(
    asset: str | Path,
    recentre: tuple[float, float, float] = ISS_RECENTRE_OFFSET,
) -> np.ndarray:
    """Every visible ISS vertex of `asset`, in world coordinates.

    Applies the same upright rotation and recentring `ISSScene` does, so the
    result is directly comparable with the dock poses in `envs/iss` and with
    the collision hull. Used by the world-frame pinning test in `tests/render/test_iss_scene.py`.
    """
    group = _strip_embedded_extras(load_glb_scene(Path(asset)))
    group.local.rotation = la.quat_from_euler(
        np.array(UPRIGHT_EULER_XYZ, dtype=np.float32), order="xyz"
    )
    group.local.position = tuple(-np.asarray(recentre, dtype=np.float32))

    collected: list[np.ndarray] = []

    def visit(node: gfx.WorldObject, parent_matrix: np.ndarray) -> None:
        matrix = parent_matrix @ np.asarray(node.local.matrix, dtype=np.float64)
        if isinstance(node, gfx.Mesh) and bool(getattr(node, "visible", True)):
            arr = _mesh_positions(node)
            if arr is not None:
                homogeneous = np.concatenate([arr.astype(np.float64), np.ones((arr.shape[0], 1))], axis=1)
                collected.append((matrix @ homogeneous.T).T[:, :3])
        for child in getattr(node, "children", []) or []:
            visit(child, matrix)

    visit(group, np.eye(4, dtype=np.float64))
    if not collected:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(collected, axis=0)
