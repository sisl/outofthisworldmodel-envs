"""The ISS docking scene: station, capsule, Earth, Moon, and starfield.

Builds a static pygfx scene graph once and re-poses the Dragon capsule from
the simulation state on every call to `ISSScene.update`. Camera views and the
force/torque debug overlays are built on top of this scene elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import jax.numpy as jnp
import numpy as np
import pygfx as gfx
import pylinalg as la
from astrojax.constants import OMEGA_EARTH

from owm_envs.core.models import ConfigModel
from owm_envs.core.quaternion import quat_to_rotmat
from owm_envs.render import asset_path
from owm_envs.render.earth import earth_texture_path
from owm_envs.render.loaders import load_cubemap_from_faces, load_glb_scene


class RenderConfig(ConfigModel):
    """Graphics settings for the ISS docking scene."""

    image_width: int = 512
    image_height: int = 512

    earth_radius_m: float = 6_378_137.0
    iss_altitude_m: float = 420_000.0
    # Geographic lon/lat placed directly under the ISS on the Earth patch.
    earth_patch_center_lon_deg: float = -122.1697
    earth_patch_center_lat_deg: float = 37.4275
    # Full angular width/height of the cropped Earth patch, in degrees.
    earth_patch_full_angle_deg: float = 50.0
    show_earth_glow: bool = True
    earth_glow_strength: float = 1.2
    show_earth_clouds: bool = True
    earth_cloud_altitude_m: float = 12_000.0
    earth_cloud_opacity: float = 0.8

    moon_radius_m: float = 1_737_400.0
    moon_asset_radius_units: float = 1.2718640565872192
    earth_moon_distance_m: float = 384_400_000.0
    moon_direction_from_earth_world: tuple[float, float, float] = (0.0, 1.0, 0.0)

    directional_light_intensity: float = 8.0
    sun_direction_world: tuple[float, float, float] = (1.0, -0.3, 0.5)
    sun_visual_distance_m: float = 1_000_000.0
    sun_angular_diameter_deg: float = 0.53
    # When True, the sun direction and Earth's spin angle are driven by
    # `orbit` (an `owm_envs.envs.iss.orbit.OrbitConfig` dump) at the sim time
    # passed to `ISSScene.update`, replacing the static `sun_direction_world`
    # above. Default off: rendering is byte-identical to a build with these
    # two fields absent.
    sun_from_epoch: bool = False
    orbit: dict | None = None

    scene_camera_near_m: float = 5.0
    scene_camera_far_m: float = 1_000_000.0
    fpv_camera_near_m: float = 0.05
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


def _unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n >= eps else np.zeros_like(v)


def _reference_orbit_from_config(cfg: RenderConfig) -> ReferenceOrbit | None:
    """A `ReferenceOrbit` built from `cfg.orbit`, or `None` when
    `cfg.sun_from_epoch` is off. The import is local so
    `owm_envs.render.iss_scene` stays importable without `owm_envs.envs.iss`
    when epoch-driven lighting isn't used."""
    if not cfg.sun_from_epoch:
        return None
    from owm_envs.envs.iss.orbit import OrbitConfig, ReferenceOrbit

    return ReferenceOrbit(OrbitConfig(**(cfg.orbit or {})))


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


def _load_rgb_texture(path: Path) -> gfx.Texture:
    arr = np.asarray(iio.imread(path))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    arr = np.ascontiguousarray(arr[..., :3].astype(np.uint8))
    return gfx.Texture(arr, dim=2, colorspace="srgb", generate_mipmaps=True)


def _load_cloud_texture(path: Path, *, opacity: float) -> gfx.Texture:
    """Cloud patches are baked as plain RGB; their brightness is coverage, so
    turn that into the alpha channel of a white RGBA texture."""
    arr = np.asarray(iio.imread(path))
    brightness = arr[..., :3].astype(np.float32).mean(axis=-1) if arr.ndim == 3 else arr.astype(np.float32)
    alpha = np.clip(np.rint(brightness * float(opacity)), 0.0, 255.0).astype(np.uint8)
    rgb = np.full(alpha.shape + (3,), 255, dtype=np.uint8)
    rgba = np.ascontiguousarray(np.concatenate([rgb, alpha[..., None]], axis=-1))
    return gfx.Texture(rgba, dim=2, colorspace="srgb", generate_mipmaps=True)


def _texture_map(texture: gfx.Texture) -> gfx.TextureMap:
    return gfx.TextureMap(texture, filter="linear", wrap="repeat")


def _earth_patch_geometry(*, radius: float, full_angle_deg: float, segments: int = 256) -> gfx.Geometry:
    """A curved rectangular patch of a sphere -- only the small region of
    Earth's surface visible from ISS altitude needs geometry, not a globe."""
    full_angle_rad = np.deg2rad(float(full_angle_deg))
    u = np.linspace(0.0, 1.0, segments + 1, dtype=np.float32)
    v = np.linspace(0.0, 1.0, segments + 1, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    lon = (uu - 0.5) * full_angle_rad
    lat = (0.5 - vv) * full_angle_rad
    cos_lat, sin_lat = np.cos(lat), np.sin(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)

    x = float(radius) * cos_lat * sin_lon
    y = float(radius) * sin_lat
    z = float(radius) * cos_lat * cos_lon
    positions = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
    normals = positions / max(float(radius), 1e-6)
    texcoords = np.stack([uu, vv], axis=-1).reshape(-1, 2).astype(np.float32)

    cols = segments + 1
    indices = np.empty((segments * segments * 2, 3), dtype=np.uint32)
    tri = 0
    for j in range(segments):
        row, next_row = j * cols, (j + 1) * cols
        for i in range(segments):
            a, b, c, d = row + i, row + i + 1, next_row + i, next_row + i + 1
            indices[tri] = (a, c, b)
            indices[tri + 1] = (b, c, d)
            tri += 2

    return gfx.Geometry(positions=positions, normals=normals, texcoords=texcoords, indices=indices)


class ISSScene:
    """The pygfx scene graph for the ISS docking environment: the station,
    the Dragon capsule, Earth, the Moon, and the starfield background."""

    def __init__(self, cfg: RenderConfig) -> None:
        self.cfg = cfg
        self._reference_orbit = _reference_orbit_from_config(cfg)

        # Earth patch local axes are east (+X), north (+Y), up (+Z); express
        # the planetary spin axis in that frame so the patch can be rotated
        # about Earth's true axis while staying anchored to its lon/lat.
        lat_rad = np.deg2rad(cfg.earth_patch_center_lat_deg)
        self._earth_spin_axis_local = _unit(
            np.array([0.0, np.cos(lat_rad), np.sin(lat_rad)], dtype=np.float32)
        )
        self._earth_spin_angle_rad = 0.0

        # Resolved once and held: the download mirror behind this call is
        # currently rate-limited, so a per-frame lookup would repeat a failed
        # network round-trip on every frame.
        self._earth_color_path = earth_texture_path("color")
        self._earth_clouds_path = earth_texture_path("clouds")

        # Loaded assets face +Y; rotate them onto this environment's body +Z
        # so the capsule's nose and the station's long axis agree with the
        # simulation's convention.
        self._upright_quat = la.quat_from_euler(np.array([np.pi / 2.0, 0.0, 0.0], dtype=np.float32), order="xyz")

        self.background = gfx.Scene()
        starmap = load_cubemap_from_faces(asset_path("nasa_starmap_2020"), ext="png")
        self.background.add(gfx.Background(None, gfx.BackgroundSkyboxMaterial(map=starmap)))

        self._earth_surface_group = gfx.Group()

        self.scene = gfx.Scene()
        self.scene.add(self._load_iss_group())
        self.scene.add(self._load_earth_group())
        self.scene.add(self._load_moon_group())
        # Held so `_apply_epoch_lighting` can re-point them per frame when
        # `cfg.sun_from_epoch` is on.
        self._sun = self._build_sun_sphere()
        self.scene.add(self._sun)
        self._directional_light = self._build_directional_light()
        self.scene.add(self._directional_light)
        self.scene.add(self._build_ambient_light())
        self.scene.add(self._build_fill_directional_light())

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
        scene_obj = load_glb_scene(asset_path("international-space-station", "ISS_stationary.glb"))
        iss_group = _strip_embedded_extras(scene_obj)
        iss_group.local.rotation = self._upright_quat

        center = _visible_geometry_center(iss_group)
        if center is not None:
            iss_group.local.position = tuple((-center).tolist())

        return iss_group

    def _load_earth_group(self) -> gfx.Group:
        cfg = self.cfg
        earth_group = gfx.Group()
        earth_group.add(self._earth_surface_group)

        earth_tex = _load_rgb_texture(self._earth_color_path)
        earth_geom = _earth_patch_geometry(radius=cfg.earth_radius_m, full_angle_deg=cfg.earth_patch_full_angle_deg)
        earth_mat = gfx.MeshStandardMaterial(
            map=_texture_map(earth_tex),
            roughness=1.0,
            metalness=0.0,
            emissive=(0.06, 0.06, 0.08),
            emissive_intensity=1.0,
        )
        earth_mesh = gfx.Mesh(earth_geom, earth_mat)
        self._earth_surface_group.add(earth_mesh)

        if cfg.show_earth_glow and cfg.earth_glow_strength > 0.0:
            self._add_earth_glow(earth_group, earth_mesh.local.rotation)

        if cfg.show_earth_clouds:
            cloud_tex = _load_cloud_texture(self._earth_clouds_path, opacity=cfg.earth_cloud_opacity)
            cloud_geom = _earth_patch_geometry(
                radius=cfg.earth_radius_m + cfg.earth_cloud_altitude_m,
                full_angle_deg=cfg.earth_patch_full_angle_deg,
            )
            cloud_mat = gfx.MeshBasicMaterial(map=_texture_map(cloud_tex))
            cloud_mat.alpha_mode = "blend"
            cloud_mesh = gfx.Mesh(cloud_geom, cloud_mat)
            self._earth_surface_group.add(cloud_mesh)

        earth_group.local.position = (0.0, 0.0, -(cfg.earth_radius_m + cfg.iss_altitude_m))
        self._apply_earth_surface_rotation()
        return earth_group

    def _add_earth_glow(self, earth_group: gfx.Group, surface_rotation) -> None:
        # Layered additive backface shells approximate a thin atmospheric rim
        # without a custom Fresnel shader.
        cfg = self.cfg
        color = (0.56, 0.87, 1.0, 1.0)
        shell_count = 64
        inner_scale, outer_scale = 1.003, 1.090
        t = np.linspace(0.0, 1.0, shell_count, dtype=np.float32)
        scales = inner_scale + (outer_scale - inner_scale) * np.power(t, 1.35)
        opacities = np.geomspace(0.020, 0.0008, shell_count).astype(np.float32)

        for scale, base_opacity in zip(scales, opacities):
            geom = gfx.sphere_geometry(radius=cfg.earth_radius_m * float(scale), width_segments=96, height_segments=48)
            mat = gfx.MeshBasicMaterial(color=color)
            mat.opacity = float(np.clip(base_opacity * cfg.earth_glow_strength, 0.0, 1.0))
            mat.alpha_mode = "add"
            mat.side = gfx.VisibleSide.back
            mat.depth_write = False
            shell = gfx.Mesh(geom, mat)
            shell.local.rotation = surface_rotation
            earth_group.add(shell)

    def _apply_earth_surface_rotation(self) -> None:
        self._earth_surface_group.local.rotation = la.quat_from_axis_angle(
            self._earth_spin_axis_local, self._earth_spin_angle_rad
        )

    def _load_moon_group(self) -> gfx.Group:
        cfg = self.cfg
        scene_obj = load_glb_scene(asset_path("moon", "moon_small.glb"))
        moon_group = _strip_embedded_extras(scene_obj)

        scale = cfg.moon_radius_m / max(cfg.moon_asset_radius_units, 1e-6)
        moon_group.local.scale = (scale, scale, scale)

        earth_center_world = np.array([0.0, 0.0, -(cfg.earth_radius_m + cfg.iss_altitude_m)], dtype=np.float32)
        direction = _unit(np.array(cfg.moon_direction_from_earth_world, dtype=np.float32))
        moon_world = earth_center_world + cfg.earth_moon_distance_m * direction
        moon_group.local.position = tuple(moon_world.tolist())
        return moon_group

    def _build_sun_sphere(self) -> gfx.Mesh:
        cfg = self.cfg
        distance = max(cfg.sun_visual_distance_m, 1.0)
        angular_radius_rad = 0.5 * np.deg2rad(cfg.sun_angular_diameter_deg)
        radius = max(distance * float(np.tan(angular_radius_rad)), 1.0)
        sun = gfx.Mesh(
            gfx.sphere_geometry(radius=radius, width_segments=32, height_segments=16),
            gfx.MeshBasicMaterial(color=(1.0, 0.96, 0.82, 1.0)),
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

    def _build_ambient_light(self) -> gfx.AmbientLight:
        return gfx.AmbientLight("#c7d8ff", 0.22)

    def _build_fill_directional_light(self) -> gfx.DirectionalLight:
        fill_direction = _unit(np.array([-0.55, 0.4, 0.9], dtype=np.float32))
        light = gfx.DirectionalLight("#dfe9ff", 0.35 * self.cfg.directional_light_intensity)
        light.local.position = tuple((fill_direction * self.cfg.sun_visual_distance_m).tolist())
        light.cast_shadow = False
        return light

    def update(
        self, state: np.ndarray, action: np.ndarray | None = None, t_offset_s: float = 0.0
    ) -> None:
        """Pose the Dragon capsule from a 13D state: 0:3 position, 6:10
        quaternion q_bw (body -> world), the rest unused here.

        `t_offset_s` is the simulation time, in seconds past the orbit
        epoch. It drives the sun direction and Earth's spin angle when
        `cfg.sun_from_epoch` is on; otherwise it is ignored and the static
        `cfg.sun_direction_world` lighting built in `__init__` stands.
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

        if self._reference_orbit is not None:
            self._apply_epoch_lighting(t_offset_s)

    def _apply_epoch_lighting(self, t_offset_s: float) -> None:
        """Point the sun and rotate the Earth's surface to match `t_offset_s`
        seconds past the orbit epoch, via `self._reference_orbit`."""
        direction = _unit(self._reference_orbit.sun_direction_world(t_offset_s))
        light_position = tuple((direction * self.cfg.sun_visual_distance_m).tolist())
        self._sun.local.position = light_position
        self._directional_light.local.position = light_position

        self._earth_spin_angle_rad = float(OMEGA_EARTH) * t_offset_s
        self._apply_earth_surface_rotation()
