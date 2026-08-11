import numpy as np
import pytest

pytest.importorskip("pygfx", reason="rendering is an optional extra")
pytest.importorskip("trimesh", reason="GLB loading needs trimesh")

from owm_envs.render.atmosphere import AtmosphereMaterial  # noqa: E402
from owm_envs.render.iss_scene import ISSScene, RenderConfig  # noqa: E402
from owm_envs.render.renderer import ISSRenderer  # noqa: E402


def test_the_radii_have_to_bracket_a_shell():
    with pytest.raises(ValueError, match="surface_radius < outer_radius"):
        AtmosphereMaterial(surface_radius=6.4e6, outer_radius=6.4e6)
    with pytest.raises(ValueError, match="surface_radius < outer_radius"):
        AtmosphereMaterial(surface_radius=6.4e6, outer_radius=6.0e6)


def test_the_material_round_trips_its_uniforms():
    material = AtmosphereMaterial(
        (0.1, 0.2, 0.3, 1.0),
        surface_radius=6.0e6,
        outer_radius=6.2e6,
        strength=0.7,
        falloff=2.5,
    )
    assert material.surface_radius == pytest.approx(6.0e6)
    assert material.outer_radius == pytest.approx(6.2e6)
    assert material.strength == pytest.approx(0.7)
    assert material.falloff == pytest.approx(2.5)
    material.strength = 1.5
    assert material.strength == pytest.approx(1.5)


def test_the_shell_is_parented_to_the_globe_and_sized_from_the_config():
    """The shader reads the planet's centre off this object's own world
    transform, so it has to hang off the group that carries the globe -- that
    is what makes it follow when `_apply_lighting` moves the Earth to the
    chief's true altitude, with nothing per-frame to update."""
    cfg = RenderConfig(earth_atmosphere_scale=1.03, earth_glow_strength=0.5)
    scene = ISSScene(cfg)
    glow = scene._earth_glow

    assert glow.parent is scene._earth_group
    assert glow.material.surface_radius == pytest.approx(cfg.earth_radius_m)
    assert glow.material.outer_radius == pytest.approx(cfg.earth_radius_m * 1.03)
    assert glow.material.strength == pytest.approx(0.5)

    scene.update(
        np.zeros(13, dtype=np.float32),
        lighting=_lighting(chief_distance_m=7.0e6),
    )
    np.testing.assert_allclose(
        np.asarray(glow.world.position), [0.0, 0.0, -7.0e6], atol=1.0
    )


def _lighting(**overrides):
    from owm_envs.render.inputs import Lighting

    kwargs = {
        "sun_direction_world": np.array([0.0, 0.0, 1.0]),
        "illumination": 1.0,
        "chief_distance_m": 6_795_000.0,
        "moon_vector_world": np.array([0.0, 3.6e8, 0.0]),
        "earth_rotation_world": np.eye(3),
    }
    return Lighting(**(kwargs | overrides))


def test_an_atmosphere_reaching_past_the_station_is_rejected():
    """A shell the camera flies inside stops being a rim at the horizon and
    becomes a wash over the whole sky, because every ray is then inside the
    air. That is a property of the config rather than of any one frame, so it
    is refused where the config is built rather than left to be discovered in
    the pixels."""
    assert RenderConfig().earth_atmosphere_scale == pytest.approx(1.020)
    with pytest.raises(ValueError, match="wash over the whole sky"):
        RenderConfig(earth_atmosphere_scale=1.09)
    # The check is about the limb, so it does not fire when there is no limb.
    RenderConfig(earth_atmosphere_scale=1.09, show_earth_glow=False)


def test_the_limb_is_one_shaded_pass_not_a_stack():
    """Why the band cannot be striped.

    The reported artefact was stripes parallel to the horizon: a band built
    from N stacked shells adds a constant over each shell's whole silhouette,
    so its radial profile is a staircase of N steps rather than a curve. What
    rules that out is structural -- there is one object, shaded per view ray,
    with no silhouettes to step between. A pixel-level scan cannot say this
    more directly than the structure does: the band is a few tens of pixels
    across, its edges are hard discards, and averaging around a ring on a
    square pixel grid leaves a percent or two of sampling ripple of its own,
    which is the same size as the artefact being looked for. The eyeball
    version is `pocs/limb_artifacts.py`.

    What the pixels do still pin lives in `test_renderer.py`: that the band is
    a limb and not a wash over the disc, and that nothing is punched out of it.
    """
    scene = ISSScene(RenderConfig())
    glow = [
        child
        for child in scene._earth_group.children
        if child is not scene._earth_surface_group
    ]
    assert glow == [scene._earth_glow]
    assert isinstance(scene._earth_glow.material, AtmosphereMaterial)


def _inputs():
    from owm_envs.render.inputs import RenderInputs

    return RenderInputs(
        position_world=np.array([0.0, -120.0, 0.0], dtype=np.float32),
        quaternion_bw=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        lighting=_lighting(),
    )
