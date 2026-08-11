"""The atmospheric limb, as one screen-space pass instead of stacked shells.

The rim of air around the Earth is what a ray sees when it passes close to the
planet without hitting it, and how much it sees depends only on how close it
came: the impact parameter of the ray against the globe's centre. That is a
per-pixel quantity, so this draws it per pixel -- a full-screen pass that
intersects each view ray with the atmosphere and shades it from the chord it
cuts.

Stacking translucent shells is the way to get the same look out of stock
materials, and it is what this replaces. It cannot be made smooth: each shell
adds a constant over its whole silhouette, so N shells are N hard steps in
additive alpha, and the steps read as bands parallel to the horizon. Measured
on the limb, going from 64 shells to 256 cuts the high-frequency ripple 4.7x
(0.564 -> 0.119) and costs 46% more frame time (28.2 -> 41.1 ms), and 512 is
WORSE than 256 because each shell's opacity falls below the 8-bit blend
quantum. There is no shell count that removes the banding.

`AtmosphereMaterial` carries the geometry (both radii) and the look (colour,
strength, falloff) and `AtmosphereShader` is its wgpu render function. The
object is parented to the globe, so its world transform is where the planet
is and the shader needs no per-frame update of its own.
"""

from __future__ import annotations

import numpy as np
import pygfx as gfx
import wgpu
from pygfx.renderers.wgpu import Binding, BaseShader, register_wgpu_render_function

_ATMOSPHERE_WGSL = """
{$ include 'pygfx.std.wgsl' $}

struct VertexInput {
    @builtin(vertex_index) index : u32,
};

@vertex
fn vs_main(in: VertexInput) -> Varyings {
    // A screen-filling triangle strip, exactly as the background pass builds
    // one: this pass has no geometry of its own, because what it draws is a
    // property of each view ray rather than of any surface.
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
    );
    let pos = positions[i32(in.index)];
    var varyings: Varyings;
    varyings.position = vec4<f32>(pos, 0.0, 1.0);
    varyings.ray_ndc = vec2<f32>(pos * u_stdinfo.ndc_offset.xy + u_stdinfo.ndc_offset.zw);
    return varyings;
}

@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {
    // This pixel's view ray, in world coordinates.
    let cam_pos = u_stdinfo.cam_transform_inv[3].xyz;
    let ray_dir = normalize(ndc_to_world_pos(vec4<f32>(varyings.ray_ndc, 1.0, 1.0)) - cam_pos);

    // Where the globe is: this object is parented to it, so its own world
    // transform is the planet's centre.
    let centre = u_wobject.world_transform[3].xyz;
    let to_centre = centre - cam_pos;
    let along = dot(to_centre, ray_dir);
    if (along <= 0.0) { discard; }  // the planet is behind the camera

    let surface = u_material.surface_radius;
    let outer = u_material.outer_radius;
    // Impact parameter: how close this ray passes to the planet's centre.
    let impact_sq = max(dot(to_centre, to_centre) - along * along, 0.0);
    let impact = sqrt(impact_sq);

    // Outside the shell there is no air; inside the surface radius the ray
    // hits the ground, and the thin column of air in front of the lit surface
    // is not what this pass draws.
    if (impact >= outer || impact <= surface) { discard; }

    // How much air the ray crosses, as a fraction of the most any ray can --
    // the grazing ray, which just clears the surface. Entry is clamped to the
    // camera, so a camera that has descended inside the shell sees only the
    // air still in front of it rather than a full chord it is halfway through.
    let half_chord = sqrt(max(outer * outer - impact_sq, 0.0));
    let entry_t = max(along - half_chord, 0.0);
    let exit_t = along + half_chord;
    let max_chord = 2.0 * sqrt(outer * outer - surface * surface);
    let path = clamp((exit_t - entry_t) / max_chord, 0.0, 1.0);

    // Density falls off with altitude far faster than the chord alone
    // suggests, which is what keeps the rim a rim rather than a wash. The
    // exponent stands in for a scale height.
    let altitude = (impact - surface) / (outer - surface);
    let density = exp(-altitude * u_material.falloff);

    let intensity = path * density * u_material.strength;
    let physical_color = srgb2physical(u_material.color.rgb);
    let out_color = vec4<f32>(physical_color * intensity, intensity);

    var out: FragmentOutput;
    out.color = out_color;
    // The depth of the point where the ray enters the atmosphere, so whatever
    // is nearer -- the station, the capsule -- occludes the rim through the
    // ordinary depth test rather than through anything bespoke.
    let entry = cam_pos + ray_dir * entry_t;
    let entry_ndc = u_stdinfo.projection_transform * u_stdinfo.cam_transform * vec4<f32>(entry, 1.0);
    out.depth = entry_ndc.z / entry_ndc.w;
    $$ if write_pick
        out.pick = pick_pack(u32(u_wobject.global_id), 20);
    $$ endif
    return out;
}
"""


class AtmosphereMaterial(gfx.Material):
    """A planet's atmospheric limb, shaded per view ray.

    `surface_radius` and `outer_radius` are world-space radii about the object's
    own origin: rays passing between them are shaded, rays outside see nothing,
    and rays inside hit the ground and are left to the surface material.
    `falloff` is how sharply density drops with altitude -- larger is a tighter,
    brighter rim -- and `strength` scales the whole thing.
    """

    uniform_type = dict(
        gfx.Material.uniform_type,
        color="4xf4",
        surface_radius="f4",
        outer_radius="f4",
        strength="f4",
        falloff="f4",
    )

    def __init__(
        self,
        color=(0.56, 0.87, 1.0, 1.0),
        *,
        surface_radius: float,
        outer_radius: float,
        strength: float = 1.0,
        falloff: float = 4.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not outer_radius > surface_radius > 0.0:
            raise ValueError(
                f"need 0 < surface_radius < outer_radius, got {surface_radius} "
                f"and {outer_radius}"
            )
        self.color = color
        self.surface_radius = surface_radius
        self.outer_radius = outer_radius
        self.strength = strength
        self.falloff = falloff

    def _set_scalar(self, name: str, value: float) -> None:
        self.uniform_buffer.data[name] = float(value)
        self.uniform_buffer.update_full()

    @property
    def color(self) -> gfx.utils.color.Color:
        return gfx.utils.color.Color(self.uniform_buffer.data["color"])

    @color.setter
    def color(self, value) -> None:
        self.uniform_buffer.data["color"] = gfx.utils.color.Color(value)
        self.uniform_buffer.update_full()

    @property
    def surface_radius(self) -> float:
        return float(self.uniform_buffer.data["surface_radius"])

    @surface_radius.setter
    def surface_radius(self, value: float) -> None:
        self._set_scalar("surface_radius", value)

    @property
    def outer_radius(self) -> float:
        return float(self.uniform_buffer.data["outer_radius"])

    @outer_radius.setter
    def outer_radius(self, value: float) -> None:
        self._set_scalar("outer_radius", value)

    @property
    def strength(self) -> float:
        return float(self.uniform_buffer.data["strength"])

    @strength.setter
    def strength(self, value: float) -> None:
        self._set_scalar("strength", value)

    @property
    def falloff(self) -> float:
        return float(self.uniform_buffer.data["falloff"])

    @falloff.setter
    def falloff(self, value: float) -> None:
        self._set_scalar("falloff", value)


@register_wgpu_render_function(gfx.WorldObject, AtmosphereMaterial)
class AtmosphereShader(BaseShader):
    type = "render"

    def get_bindings(self, wobject, shared):
        bindings = {
            0: Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            1: Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            2: Binding("u_material", "buffer/uniform", wobject.material.uniform_buffer),
        }
        self.define_bindings(0, bindings)
        return {0: bindings}

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_strip,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        return {"indices": (4, 1)}

    def get_code(self):
        return _ATMOSPHERE_WGSL


def atmosphere_shell(
    *,
    surface_radius: float,
    outer_radius: float,
    color: tuple[float, float, float, float],
    strength: float,
    falloff: float,
    render_queue: int,
) -> gfx.WorldObject:
    """The object that draws the limb. Parent it to the globe: the shader reads
    the planet's centre off this object's world transform.

    It carries a geometry only because pygfx needs one to compute a bounding
    box -- the shader draws a screen-filling strip and never reads it. A single
    degenerate point keeps that box at the globe's own origin rather than
    claiming the whole scene.
    """
    obj = gfx.WorldObject(
        gfx.Geometry(positions=np.zeros((1, 3), dtype=np.float32)),
        AtmosphereMaterial(
            color,
            surface_radius=surface_radius,
            outer_radius=outer_radius,
            strength=strength,
            falloff=falloff,
        ),
    )
    obj.material.alpha_mode = "add"
    obj.material.depth_write = False
    obj.material.render_queue = render_queue
    return obj
