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
from owm_envs.render.inputs import RenderInputs
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
# default actuator limits (1.6 kN / 2 kN*m) purely to give the visualization
# a sensible default scale -- the renderer has no dependency on that config,
# so an action from a differently-tuned control config will just saturate
# the overlay rather than misrepresent it.
_DEFAULT_MAX_FORCE_N = 1600.0
_DEFAULT_MAX_TORQUE_NM = 2000.0


def _unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n >= eps else np.zeros_like(v)


def _pose_row(inputs: RenderInputs) -> np.ndarray:
    """The 13D row `_build_views` and `ISSScene.update` pose from.

    Both read the position out of 0:3 and the attitude out of 6:10 and nothing
    else, so the velocity and angular-rate slices are zeros: a `RenderInputs`
    carries no rates, and posing needs none.
    """
    return np.concatenate(
        [
            np.asarray(inputs.position_world, dtype=np.float32).reshape(3),
            np.zeros(3, dtype=np.float32),
            np.asarray(inputs.quaternion_bw, dtype=np.float32).reshape(4),
            np.zeros(3, dtype=np.float32),
        ]
    )


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
# camera cannot hold both a hull 0.4 m away and a moon 390,000 km away. The
# Moon is the object that could never fit -- reaching it needs near >= 23 m,
# which would clip the capsule's own nose cone -- so it is drawn in a pass of
# its own, against a near plane sized for lunar distance, and the main pass no
# longer has to reach it. See `ISSScene.distant` and `ISSRenderer._draw`.
#
# The Earth still shares the main pass with the station, which is a far milder
# ask: the limb at 2350 km sits comfortably inside what a 0.3 m near plane
# expresses.
_MAX_DEPTH_RANGE_RATIO = 2.0**24

# `earth_moon_distance_m` is the mean distance, but a frame rendered from
# ephemeris (`Lighting.moon_vector_world`) puts the Moon at the distance it
# really was: 405,500 km at apogee against the 384,400 km mean, 5.5% further
# out, and 5.8% at the extreme of the anomalistic swing. A constant factor
# rather than a clip distance derived from the frame's own Moon: this only has
# to bracket where the Moon can be, and making it per-frame would move the
# distant camera's planes with the ephemeris to buy nothing.
_MOON_APOGEE_MARGIN = 1.06

# How far in front of the nearest possible Moon the distant pass puts its near
# plane. The pass holds nothing else, so this plane exists only to keep the
# projected depth well clear of 1.0 -- at a tenth of the distance to the
# subject, depth lands near 0.9 and has bits to spare.
_DISTANT_NEAR_FRACTION = 0.1


def _distant_depth_range(cfg: RenderConfig) -> tuple[float, float]:
    """near/far bracketing every place the Moon can be, for the pass that
    draws only the Moon.

    The camera rides the chief, so the Moon's range is its geocentric distance
    give or take an orbital radius. Both ends are widened past that: nothing
    else is in this pass to be clipped by a generous far plane, and nothing in
    it writes depth, so the precision these planes buy is spent only on keeping
    the Moon's own fragments away from a depth of exactly 1.0.
    """
    nearest = cfg.earth_moon_distance_m / _MOON_APOGEE_MARGIN - (
        cfg.earth_radius_m + cfg.iss_altitude_m + cfg.moon_radius_m
    )
    farthest = cfg.earth_moon_distance_m * _MOON_APOGEE_MARGIN + (
        cfg.earth_radius_m + cfg.iss_altitude_m + cfg.moon_radius_m
    )
    return nearest * _DISTANT_NEAR_FRACTION, farthest


def _far_covering_the_scene(cfg: RenderConfig) -> float:
    """A far clip distance that reaches the far side of the Earth's glow
    shells, the most distant thing left in the main scene.

    Earth lives in the same scene graph as the ISS and Dragon, thousands of
    kilometres out, so a far plane sized for nearby station geometry would clip
    the planet out entirely. Its far side is what has to fit rather than its
    limb: a nearer plane would cut the globe through the middle, and the far
    half being hidden anyway is a fact about the depth test, not about the
    clip. The Moon is not counted -- it left this scene for `ISSScene.distant`
    precisely because no near plane close enough to render a dock could reach
    it.
    """
    outermost_glow = 1.09  # `ISSScene._add_earth_glow`'s outer shell scale
    return (
        cfg.earth_radius_m
        + cfg.iss_altitude_m
        + outermost_glow * cfg.earth_radius_m
        + cfg.sun_visual_distance_m
    )


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

    def views(self, inputs: RenderInputs) -> dict[ViewName, CameraView]:
        return _build_views(self.cfg, _pose_row(inputs))

    def render(
        self,
        inputs: RenderInputs,
        view: ViewName = "DRAGON_ISO",
    ) -> np.ndarray:
        return self.render_views(inputs, (view,))[view]

    def render_views(
        self,
        inputs: RenderInputs,
        views: Sequence[ViewName] = ("DRAGON_ISO",),
    ) -> dict[ViewName, np.ndarray]:
        """Render several of the named views of one frame, posed once.

        Posing the scene is per-frame, not per-camera, so a caller that wants
        more than one view of the same frame should ask for them together:
        `render` in a loop would re-pose the capsule and rebuild the debug
        overlays once per view for no change in what is drawn.

        `inputs` is whatever the source environment's render adapter made of
        its own state -- the renderer never sees that state, which is what
        lets it draw for an environment whose rows it could not read.
        """
        if self._closed:
            raise RuntimeError("renderer is closed")

        if isinstance(views, str):
            # A bare string is iterable, so this would otherwise be reported as
            # an unknown view named "D".
            raise TypeError(f"views must be a sequence of view names, not {views!r}")

        pose = _pose_row(inputs)
        all_views = _build_views(self.cfg, pose)
        unknown = [view for view in views if view not in all_views]
        if unknown:
            raise ValueError(
                f"unknown view {unknown[0]!r}; expected one of {sorted(all_views)}"
            )

        self._iss_scene.update(pose, inputs.action, lighting=inputs.lighting)
        self._update_debug_overlays(inputs.action)
        return {view: self._draw(all_views[view]) for view in views}

    def render_view(
        self,
        inputs: RenderInputs,
        view: CameraView,
    ) -> np.ndarray:
        """Render the posed scene through an arbitrary camera.

        The six named views go through here as well; pass a `CameraView` of your
        own to look at the scene from somewhere they do not cover.
        """
        if self._closed:
            raise RuntimeError("renderer is closed")

        self._iss_scene.update(_pose_row(inputs), inputs.action, lighting=inputs.lighting)
        self._update_debug_overlays(inputs.action)
        return self._draw(view)

    def _draw(self, view: CameraView) -> np.ndarray:
        """Draw the already-posed scene through one camera, in three passes
        from the back of the scene forward.

        Each pass shares the camera's position and orientation and differs only
        in how deep it can see, which is what lets one frame hold a capsule
        hull 0.4 m away and a moon 390,000 km away. None of the first two
        passes writes depth, so the three only have to agree on where the
        camera is:

          starfield  the camera as configured, clearing colour and depth
          distant    the Moon, against a near plane sized for lunar distance
          scene      Earth, station and capsule, far enough for the planet

        Ordering is what composites them: the main scene paints over the Moon,
        so the Earth, the station and the capsule occlude it without any depth
        comparison between passes -- which could not be meaningful anyway,
        since each pass projects depth through a different near plane.
        """
        self._renderer.render(self._iss_scene.background, make_camera(view), clear=True)
        distant_near, distant_far = _distant_depth_range(self.cfg)
        self._renderer.render(
            self._iss_scene.distant,
            make_camera(dataclasses.replace(view, near=distant_near, far=distant_far)),
            clear=False,
        )
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
