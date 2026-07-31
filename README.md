# outofthisworldmodel-envs
Training data environments for the Out of this World Model (OWM) world model

## Asset acknowledgements

The 3D assets under `src/owm_envs/render/resources/` are third-party works,
reproduced here with the provenance that is known. Their upstream **licence
terms were not recorded** when they were first collected, so the licence column
below reflects what is actually documented rather than an assertion of terms.

| Asset | Source | Licence |
|---|---|---|
| `ISS_stationary.glb` | NASA 3D Resources / science.nasa.gov | not recorded |
| `spacex_dragon_2_exterior.glb` | Sketchfab — "SpaceX Dragon 2 Exterior" | not recorded |
| `nasa_starmap_2020/` | NASA SVS #4851, cubemapped via jaxry/panorama-to-cubemap | not recorded |
| `moon/moon_small.glb` | Texture from NASA SVS #14959 (CGI Moon Kit), https://svs.gsfc.nasa.gov/14959/. **Mesh geometry source not recorded.** | not recorded |
| `earth/patches/earth_color_patch.jpg`, `earth_clouds_patch.jpg` | Baked by `scripts/bake_earth_patches.py` from high-resolution equirectangular sources whose origin is **one of several candidates and cannot be attributed to a specific one** | not recorded |

The Earth patches are derived works: each is a 50°×50° crop downsampled to
8192×8192 from a much larger source image, which is not redistributed here.

**Unresolved before public release.** Two gaps remain. The Earth source imagery
cannot be attributed to a specific origin, and while the Moon texture is now
attributed to NASA SVS #14959, the provenance of `moon_small.glb`'s mesh
geometry is still unrecorded. This is not cosmetic: Sketchfab models are
commonly CC-BY, which legally requires attribution, and NASA imagery carries its
own usage guidelines. Resolve both before this repository is made public or
redistributed.
