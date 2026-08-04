"""The offscreen renderer: turns a simulation state into an RGB frame.

Renders the shared `ISSScene` graph to an off-screen wgpu texture and hands
back a plain `(H, W, 3)` uint8 array -- what a policy, a dataset writer, or a
notebook actually wants, rather than a canvas object or a raw RGBA texture.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Sequence

import jax.numpy as jnp
import numpy as np
import pygfx as gfx
import pylinalg as la
import wgpu
from pygfx.renderers import WgpuRenderer

from owm_envs.core.quaternion import quat_to_rotmat
from owm_envs.render.iss_scene import ISSScene, RenderConfig
from owm_envs.render.view import CameraView, make_camera

ViewName = Literal["DRAGON_ISO", "DRAGON_TOP", "DRAGON_FPV", "ISS_ISO", "ISS_TOP", "ISS_FPV"]

# Debug-overlay geometry. Small enough to be cheap to build and keep around,
# even though they default to hidden -- see `ISSRenderer.show_force_arrows`
# and `.show_torque_rings`.
_FORCE_ARROW_COLORS = (
    (1.0, 0.2, 0.2, 1.0),
    (0.2, 1.0, 0.2, 1.0),
    (0.2, 0.55, 1.0, 1.0),
)
_FORCE_ARROW_MIN_LENGTH = 0.25
_FORCE_ARROW_MAX_LENGTH = 6.0
_FORCE_ARROW_RADIUS = 0.06
_FORCE_ARROW_HEAD_RADIUS = 0.16
_FORCE_ARROW_HEAD_LENGTH = 0.45

_TORQUE_RING_COLORS = (
    (1.0, 0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0, 1.0),
    (0.0, 0.0, 1.0, 1.0),
)
_TORQUE_RING_OFFSET = 3.0
_TORQUE_RING_RADIUS_MIN = 0.05
_TORQUE_RING_RADIUS_MAX = 2.25
_TORQUE_RING_THICKNESS = 0.06
_TORQUE_RING_SEGMENTS = 32

# The overlays read a 6D body-frame action as [force_xyz, torque_xyz] and
# scale arrow length / ring radius against these. They mirror ISSConfig's
# default actuator limits (18 kN / 90 kN*m) purely to give the visualization
# a sensible default scale -- the renderer has no dependency on that config,
# so an action from a differently-tuned control config will just saturate
# the overlay rather than misrepresent it.
_DEFAULT_MAX_FORCE_N = 18_000.0
_DEFAULT_MAX_TORQUE_NM = 90_000.0


def _unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n >= eps else np.zeros_like(v)


def _dragon_fpv_pose_world(cfg: RenderConfig, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dragon's onboard camera position/forward/up in world coordinates.

    The mount point and look direction are fixed in the capsule's body frame,
    so they have to be rotated into world by the current attitude on every
    call.
    """
    s = np.asarray(state, dtype=np.float32).reshape(-1)
    pos = s[0:3]
    q_bw = s[6:10]
    rotation = np.asarray(quat_to_rotmat(jnp.asarray(q_bw)), dtype=np.float32)
    offset_body = np.array(cfg.dragon_fpv_offset_body, dtype=np.float32)
    fpv_pos = pos + rotation @ offset_body
    fpv_forward = _unit(rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float32))
    fpv_up = _unit(rotation @ np.array([0.0, 1.0, 0.0], dtype=np.float32))
    return fpv_pos, fpv_forward, fpv_up


# A perspective projection maps a point at distance d to a depth of
# `1 - near/d`, and the depth buffer is float32: once `near/d` falls below the
# spacing of the floats just under 1.0, that rounds to exactly 1.0, the
# fragment fails the depth test against the cleared buffer, and whatever was
# drawn behind it shows through. The usable depth range is therefore bounded
# by the float32 mantissa -- about `near * 2**24` -- NO MATTER WHAT `far` SAYS.
# Widening `far` past this point is inert: it is the near plane that has to
# move. (Measured on this scene: near=0.05 m clips at ~1.07e6 m however large
# `far` is; near=0.5 m at ~1.7e7 m.)
#
# This budget is what makes `RenderConfig.fpv_camera_near_m` a knife edge: one
# camera cannot comfortably hold both a hull 0.4 m away and a limb 2350 km
# away. The way out is to stop asking it to -- draw the planet and the station
# in two passes with the depth buffer CLEARED between them, so each gets the
# whole budget for its own scale. pygfx 0.15 has no public depth-only clear
# (`renderer.render(clear=True)` takes the colour with it, and the background
# pass shares this same buffer), so that means compositing the two passes
# ourselves rather than moving objects between the existing scenes. Worth
# doing; too structural to fold into a bug fix.
_MAX_DEPTH_RANGE_RATIO = 2.0**24


def _far_covering_the_scene(cfg: RenderConfig) -> float:
    """A far clip distance that reaches the far side of the Moon, the most
    distant thing in the scene.

    Earth and the Moon live in the same scene graph as the ISS and Dragon,
    tens to hundreds of millions of metres out -- a far plane sized for
    nearby station geometry would clip them out entirely. The Moon is past
    what `_MAX_DEPTH_RANGE_RATIO` allows at any near plane close enough to
    render a dock, so this is an upper bound on what a camera asks for, not
    a promise that it gets there.
    """
    return cfg.earth_moon_distance_m + cfg.earth_radius_m + cfg.iss_altitude_m + cfg.moon_radius_m


def _with_scene_far(view: CameraView, cfg: RenderConfig) -> CameraView:
    """The same camera, with its far plane pushed out to cover the scene.

    Widened to reach the scene, then capped at what the near plane can
    express -- including past a caller's own far, which the projection would
    round away regardless. The returned far is therefore one the camera really
    does clip at rather than a number that only looks generous.
    """
    near = view.near if view.near is not None else 0.01
    wanted = max(view.far or 0.0, _far_covering_the_scene(cfg))
    return dataclasses.replace(view, far=min(wanted, near * _MAX_DEPTH_RANGE_RATIO))


def _build_views(cfg: RenderConfig, state: np.ndarray) -> dict[ViewName, CameraView]:
    s = np.asarray(state, dtype=np.float32).reshape(-1)
    pos = s[0:3]
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    dragon_iso = CameraView(
        name="DRAGON_ISO",
        camera_type="perspective",
        position=pos + np.array(cfg.dragon_iso_offset_world, dtype=np.float32),
        target=pos,
        up=world_up,
        fov_y_deg=cfg.dragon_iso_fov_y_deg,
        width=cfg.image_width,
        height=cfg.image_height,
        near=cfg.scene_camera_near_m,
        far=cfg.scene_camera_far_m,
    )

    dragon_top = CameraView(
        name="DRAGON_TOP",
        camera_type="perspective",
        position=pos + np.array([0.0, 0.0, cfg.dragon_top_height_m], dtype=np.float32),
        target=pos,
        up=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        fov_y_deg=cfg.dragon_top_fov_y_deg,
        width=cfg.image_width,
        height=cfg.image_height,
        near=cfg.scene_camera_near_m,
        far=cfg.scene_camera_far_m,
    )

    fpv_pos, fpv_forward, fpv_up = _dragon_fpv_pose_world(cfg, s)
    dragon_fpv = CameraView(
        name="DRAGON_FPV",
        camera_type="perspective",
        position=fpv_pos,
        target=fpv_pos + cfg.dragon_fpv_target_distance_m * fpv_forward,
        up=fpv_up,
        fov_y_deg=cfg.dragon_fpv_fov_y_deg,
        width=cfg.image_width,
        height=cfg.image_height,
        near=cfg.fpv_camera_near_m,
        far=cfg.fpv_camera_far_m,
    )

    iss_iso = CameraView(
        name="ISS_ISO",
        camera_type="perspective",
        position=np.array(cfg.iss_iso_position_world, dtype=np.float32),
        target=np.zeros(3, dtype=np.float32),
        up=world_up,
        fov_y_deg=cfg.iss_iso_fov_y_deg,
        width=cfg.image_width,
        height=cfg.image_height,
        near=cfg.scene_camera_near_m,
        far=cfg.scene_camera_far_m,
    )

    iss_top = CameraView(
        name="ISS_TOP",
        camera_type="perspective",
        position=np.array([0.0, 0.0, cfg.iss_top_height_m], dtype=np.float32),
        target=np.zeros(3, dtype=np.float32),
        up=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        fov_y_deg=cfg.iss_top_fov_y_deg,
        width=cfg.image_width,
        height=cfg.image_height,
        near=cfg.scene_camera_near_m,
        far=cfg.scene_camera_far_m,
    )

    # The ISS itself never rotates in this environment -- only the Dragon
    # does -- so its FPV mount point and look direction are already
    # world-frame vectors; no attitude to rotate them through.
    iss_fpv_pos = np.array(cfg.iss_fpv_position_iss, dtype=np.float32)
    iss_fpv_forward = _unit(np.array(cfg.iss_fpv_forward_iss, dtype=np.float32))
    iss_fpv_up = _unit(np.array(cfg.iss_fpv_up_iss, dtype=np.float32))
    iss_fpv = CameraView(
        name="ISS_FPV",
        camera_type="perspective",
        position=iss_fpv_pos,
        target=iss_fpv_pos + cfg.iss_fpv_target_distance_m * iss_fpv_forward,
        up=iss_fpv_up,
        fov_y_deg=cfg.iss_fpv_fov_y_deg,
        width=cfg.image_width,
        height=cfg.image_height,
        near=cfg.fpv_camera_near_m,
        far=cfg.fpv_camera_far_m,
    )

    return {
        "DRAGON_ISO": dragon_iso,
        "DRAGON_TOP": dragon_top,
        "DRAGON_FPV": dragon_fpv,
        "ISS_ISO": iss_iso,
        "ISS_TOP": iss_top,
        "ISS_FPV": iss_fpv,
    }


class ISSRenderer:
    """Renders `ISSScene` states to RGB frames from any of six fixed views.

    Two views follow the Dragon capsule (isometric chase, top-down, and its
    own onboard camera); the other three mirror those from the ISS's point
    of view. `show_force_arrows` and `show_torque_rings` add debug overlays
    useful for eyeballing a policy's actions -- both default off, since a
    default render should show the scene, not the control signal.
    """

    def __init__(self, cfg: RenderConfig, *, download_textures: bool = True) -> None:
        self.cfg = cfg
        self._iss_scene = ISSScene(cfg, download_textures=download_textures)
        self._closed = False

        self.show_force_arrows = False
        self.show_torque_rings = False
        self._force_arrow_groups: list[gfx.Group] | None = None
        self._torque_rings: list[gfx.Mesh] | None = None

        texture = gfx.Texture(
            dim=2,
            size=(cfg.image_width, cfg.image_height, 1),
            format="rgba8unorm-srgb",
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        )
        self._renderer = WgpuRenderer(texture)
        # snapshot() otherwise returns a texture 2x the configured size --
        # pygfx defaults to a device pixel ratio of 2.
        self._renderer.pixel_ratio = 1

    def views(self, state: np.ndarray) -> dict[ViewName, CameraView]:
        return _build_views(self.cfg, state)

    def render(
        self,
        state: np.ndarray,
        action: np.ndarray | None = None,
        view: ViewName = "DRAGON_ISO",
    ) -> np.ndarray:
        return self.render_views(state, action, (view,))[view]

    def render_views(
        self,
        state: np.ndarray,
        action: np.ndarray | None = None,
        views: Sequence[ViewName] = ("DRAGON_ISO",),
    ) -> dict[ViewName, np.ndarray]:
        """Render several of the named views of one state, posed once.

        Posing the scene is per-state, not per-camera, so a caller that wants
        more than one view of the same frame should ask for them together:
        `render` in a loop would re-pose the capsule and rebuild the debug
        overlays once per view for no change in what is drawn.
        """
        if self._closed:
            raise RuntimeError("renderer is closed")

        if isinstance(views, str):
            # A bare string is iterable, so this would otherwise be reported as
            # an unknown view named "D".
            raise TypeError(f"views must be a sequence of view names, not {views!r}")

        all_views = _build_views(self.cfg, state)
        unknown = [view for view in views if view not in all_views]
        if unknown:
            raise ValueError(
                f"unknown view {unknown[0]!r}; expected one of {sorted(all_views)}"
            )

        self._iss_scene.update(state, action)
        self._update_debug_overlays(action)
        return {view: self._draw(all_views[view]) for view in views}

    def render_view(
        self,
        state: np.ndarray,
        view: CameraView,
        action: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render the posed scene through an arbitrary camera.

        The six named views go through here as well; pass a `CameraView` of your
        own to look at the scene from somewhere they do not cover.
        """
        if self._closed:
            raise RuntimeError("renderer is closed")

        self._iss_scene.update(state, action)
        self._update_debug_overlays(action)
        return self._draw(view)

    def _draw(self, view: CameraView) -> np.ndarray:
        """Draw the already-posed scene through one camera.

        The skybox is drawn on the camera as configured and the scene on a
        copy whose far plane covers the whole scene: the background writes no
        depth, so the two passes only have to agree on where the camera is,
        not on how deep it can see.
        """
        self._renderer.render(self._iss_scene.background, make_camera(view), clear=True)
        self._renderer.render(
            self._iss_scene.scene, make_camera(_with_scene_far(view, self.cfg)), clear=False
        )
        frame = np.asarray(self._renderer.snapshot())
        return np.ascontiguousarray(frame[..., :3])

    def close(self) -> None:
        """Release this renderer's resources.

        pygfx keeps a single WGPU device shared by every renderer in the
        process (`pygfx.renderers.wgpu.engine.shared.Shared`) and has no way
        to recreate it once gone, so this must not destroy `self._renderer.
        device` -- doing so would leave every other `ISSRenderer` in the
        process (present or future) unable to render. Dropping this
        instance's own references and letting them be garbage-collected is
        the correct release.
        """
        self._force_arrow_groups = None
        self._torque_rings = None
        self._iss_scene = None
        self._renderer = None
        self._closed = True

    # -- debug overlays -----------------------------------------------------

    def _ensure_debug_overlays(self) -> None:
        if self.show_force_arrows and self._force_arrow_groups is None:
            self._force_arrow_groups = [self._make_axis_arrow(c) for c in _FORCE_ARROW_COLORS]
            for group in self._force_arrow_groups:
                self._iss_scene.dragon.add(group)
        if self.show_torque_rings and self._torque_rings is None:
            self._torque_rings = self._make_torque_rings()
            for ring in self._torque_rings:
                self._iss_scene.dragon.add(ring)

    def _update_debug_overlays(self, action: np.ndarray | None) -> None:
        self._ensure_debug_overlays()

        if self._force_arrow_groups is not None:
            for group in self._force_arrow_groups:
                group.visible = self.show_force_arrows
        if self._torque_rings is not None:
            for ring in self._torque_rings:
                ring.visible = self.show_torque_rings

        if not (self.show_force_arrows or self.show_torque_rings):
            return

        a = np.zeros(6, dtype=np.float32) if action is None else np.asarray(action, dtype=np.float32).reshape(-1)
        if self.show_force_arrows:
            self._update_force_arrows(a)
        if self.show_torque_rings:
            self._update_torque_rings(a)

    def _make_axis_arrow(self, color: tuple[float, float, float, float]) -> gfx.Group:
        mat = gfx.MeshBasicMaterial(color=color)
        mat.alpha_mode = "blend"
        mat.opacity = 0.85

        group = gfx.Group()
        shaft = gfx.Mesh(gfx.cylinder_geometry(_FORCE_ARROW_RADIUS, _FORCE_ARROW_RADIUS, 1.0, 24, 1), mat)
        shaft.local.position = (0.0, 0.0, 0.5)
        head = gfx.Mesh(gfx.cone_geometry(_FORCE_ARROW_HEAD_RADIUS, _FORCE_ARROW_HEAD_LENGTH, 24, 1), mat)
        head.local.position = (0.0, 0.0, 1.0 + 0.5 * _FORCE_ARROW_HEAD_LENGTH)
        group.add(shaft)
        group.add(head)
        return group

    def _update_force_arrows(self, action: np.ndarray) -> None:
        force_body = action[:3] if action.size >= 3 else np.zeros(3, dtype=np.float32)
        axes = (
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )

        for group, axis, component in zip(self._force_arrow_groups, axes, force_body.tolist()):
            if abs(component) < 1e-5:
                group.visible = False
                continue

            group.visible = True
            direction = axis if component >= 0.0 else -axis
            norm = float(np.clip(abs(component) / _DEFAULT_MAX_FORCE_N, 0.0, 1.0))
            length = _FORCE_ARROW_MIN_LENGTH + (_FORCE_ARROW_MAX_LENGTH - _FORCE_ARROW_MIN_LENGTH) * norm
            group.local.rotation = la.quat_from_vecs(np.array([0.0, 0.0, 1.0], dtype=np.float32), direction)
            group.local.scale = (1.0, 1.0, max(length, 1e-6))

    def _make_torque_rings(self) -> list[gfx.Mesh]:
        geom = gfx.cylinder_geometry(1.0, 1.0, _TORQUE_RING_THICKNESS, _TORQUE_RING_SEGMENTS, 2)
        mats = [gfx.MeshBasicMaterial(color=c) for c in _TORQUE_RING_COLORS]
        for mat in mats:
            mat.alpha_mode = "blend"
            mat.opacity = 0.16

        off = _TORQUE_RING_OFFSET
        q_to_plus_x = la.quat_from_euler(np.array([0.0, -np.pi / 2.0, 0.0], dtype=np.float32), order="XYZ")
        q_to_minus_x = la.quat_from_euler(np.array([0.0, np.pi / 2.0, 0.0], dtype=np.float32), order="XYZ")
        q_to_plus_y = la.quat_from_euler(np.array([-np.pi / 2.0, 0.0, 0.0], dtype=np.float32), order="XYZ")
        q_to_minus_y = la.quat_from_euler(np.array([np.pi / 2.0, 0.0, 0.0], dtype=np.float32), order="XYZ")
        q_to_plus_z = la.quat_from_euler(np.array([0.0, 0.0, -np.pi / 2.0], dtype=np.float32), order="XYZ")
        q_to_minus_z = la.quat_from_euler(np.array([0.0, 0.0, np.pi / 2.0], dtype=np.float32), order="XYZ")

        placements = (
            ((off, 0.0, 0.0), q_to_plus_x, mats[0]),
            ((-off, 0.0, 0.0), q_to_minus_x, mats[0]),
            ((0.0, off, 0.0), q_to_plus_y, mats[1]),
            ((0.0, -off, 0.0), q_to_minus_y, mats[1]),
            ((0.0, 0.0, off), q_to_plus_z, mats[2]),
            ((0.0, 0.0, -off), q_to_minus_z, mats[2]),
        )

        rings = []
        for pos, quat, mat in placements:
            ring = gfx.Mesh(geom, mat)
            ring.local.position = pos
            ring.local.rotation = quat
            rings.append(ring)
        return rings

    def _update_torque_rings(self, action: np.ndarray) -> None:
        torque_body = action[3:6] if action.size >= 6 else np.zeros(3, dtype=np.float32)
        positive = np.clip(torque_body, 0.0, None)
        negative = np.clip(-torque_body, 0.0, None)

        span = _TORQUE_RING_RADIUS_MAX - _TORQUE_RING_RADIUS_MIN
        radius_pos = _TORQUE_RING_RADIUS_MIN + span * np.clip(positive / _DEFAULT_MAX_TORQUE_NM, 0.0, 1.0)
        radius_neg = _TORQUE_RING_RADIUS_MIN + span * np.clip(negative / _DEFAULT_MAX_TORQUE_NM, 0.0, 1.0)
        radii = (radius_pos[0], radius_neg[0], radius_pos[1], radius_neg[1], radius_pos[2], radius_neg[2])

        for ring, radius in zip(self._torque_rings, radii):
            r = max(float(radius), 1e-6)
            ring.local.scale = (r, r, 1.0)
