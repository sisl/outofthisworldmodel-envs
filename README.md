# outofthisworldmodel-envs
Training data environments for the Out of this World Model (OWM) world model

## Asset acknowledgements

The 3D assets under `src/owm_envs/render/resources/` were carried over from the
`seamstress` project, whose `resources/links.txt` records source URLs but **no
licence terms**. They are reproduced here with the provenance that is known.

| Asset | Source | Licence |
|---|---|---|
| `ISS_stationary.glb` | NASA 3D Resources / science.nasa.gov | not recorded |
| `spacex_dragon_2_exterior.glb` | Sketchfab — "SpaceX Dragon 2 Exterior" | not recorded |
| `nasa_starmap_2020/` | NASA SVS #4851, cubemapped via jaxry/panorama-to-cubemap | not recorded |
| `earth_color_10K.tif`, `earth_clouds_8K.tif` | one of four candidate sources; **cannot be attributed to a specific one** | not recorded |
| `moon_small.glb` | **no source recorded** | unknown |

**Unresolved before public release.** The Earth textures cannot be attributed to
a specific source, and `moon_small.glb` has no recorded provenance at all. This
is not cosmetic: Sketchfab models are commonly CC-BY, which legally requires
attribution, and NASA imagery carries its own usage guidelines. Resolve both
rows before this repository is made public or redistributed.
