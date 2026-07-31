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
| `earth/patches/earth_color_patch.jpg`, `earth_clouds_patch.jpg` | Baked from high-resolution equirectangular imagery collected from the Earth sources listed below | not recorded |

**Earth imagery sources.** The high-resolution equirectangular maps the patches
are baked from were collected from these four:

- https://sketchfab.com/3d-models/earth-41fc80d85dfd480281f21b74b2de2faa
- https://science.nasa.gov/resource/earth-3d-model/
- https://www.cgtrader.com/items/6013252/download-page
- https://maps.drsys.eu/

Which specific source produced each shipped patch was not recorded per file, so
anyone assessing terms should check all four.

The Earth patches are derived works: each is a 50°×50° crop downsampled to
8192×8192 from a much larger source image. The source imagery itself is not
redistributed here — `scripts/bake_earth_patches.py` regenerates the patches
from a local copy.

**Before public release.** Licence terms were not recorded for any asset, and
two provenance gaps remain: which of the four Earth sources produced each patch,
and the mesh geometry of `moon_small.glb` (its texture is attributed to NASA SVS
#14959). This is not cosmetic — Sketchfab and CGTrader assets carry per-item
terms, some requiring attribution, and NASA imagery has its own usage
guidelines. Confirm terms for each asset before this repository is made public
or redistributed.
