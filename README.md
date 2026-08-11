# outofthisworldmodel-envs
Training data environments for the Out of this World Model (OWM) world model

## Command-line usage

    owm-envs generate --out logs/run1 --split train:100000t:0 --split val:20000t:1
    owm-envs generate --out logs/run2 --split train:512:0 --noise noncooperative
    owm-envs list

`--split NAME:COUNT[t]:SEED[:POLICY]` is repeatable. COUNT with a trailing
`t` targets a minimum number of transitions -- the split runs whole episodes
until it has accumulated at least that many -- rather than a fixed episode
count. This is the RECOMMENDED way to size a split: target the training
budget you actually need (transitions are what a world model actually trains
on) rather than guessing an episode count first and checking how many
transitions it happened to produce. Episode counts (`train:512:0`) remain
supported for when the episode count itself is what matters.

`--noise off|cooperative|noncooperative` overrides the `--env-config` file's
`sensor_noise` with a named preset; omit it to leave the config's own setting
in place.

`--goal-error/--no-goal-error` overrides the `--env-config` file's
`observation.goal_error`, appending the dock-goal error block to
observations; omit it to leave the config's own setting in place.

## Validating a run

Two checks on a generated run before it is published:

    uv run --extra datasets python scripts/check_sensor_noise.py logs/run1
    uv run --extra datasets python scripts/plot_trajectories_3d.py logs/run1 --out logs/run1_traj

`check_sensor_noise.py` measures the residual between each frame's
`observation_vector` and its `state_vector` -- the sensor-noise draw the
simulator actually made -- and compares its RMS per channel against the sigmas
in the run's `env_config.yaml`, exiting non-zero when any channel is off by
more than `--tolerance` (default 10%). It also breaks position error down by
true range, which is what shows whether the non-cooperative preset's
range-proportional term landed. A run generated with noise disabled is held to
exact zeros.

`plot_trajectories_3d.py` writes one self-contained plotly page per split for
the eyeball check: each episode's true path, the dock poses they were flying
to, and the station origin, on equal-aspect axes. It draws the first 64
episodes of a split by default (`--max-episodes`), thinned to 1000 points each
(`--max-points`), since a published split holds far more than a browser will
open at once.

## iss-numerical commands

Trial run, one call per noise variant:

    for v in nonoise coop noncoop; do
      uv run owm-envs generate --out outputs/trial_num_${v}_goal \
        --env-config configs/iss_numerical_${v}_goal.toml \
        --gen-config configs/generation_numerical_trial.yaml \
        --render --render-workers 16 --gpu-index 1
    done

Full 500k run: same, with the 500k gen-config and output directory:

    for v in nonoise coop noncoop; do
      uv run owm-envs generate --out outputs/full_num_${v}_goal \
        --env-config configs/iss_numerical_${v}_goal.toml \
        --gen-config configs/generation_numerical_500k.yaml \
        --render --render-workers 16 --gpu-index 1
    done

Port sweep -- 3 noise variants x 8 ports x 3 successful docked episodes each,
covering the five ports the train split uses (`harmony_fwd_pma2`,
`harmony_nadir_cbm`, `zvezda_aft`, `pirs_nadir`, `rassvet_nadir`) and the
three held out of it (`harmony_zenith_cbm`, `poisk_zenith`,
`unity_nadir_cbm`):

    PORTS="harmony_fwd_pma2 harmony_zenith_cbm harmony_nadir_cbm zvezda_aft \
           poisk_zenith pirs_nadir rassvet_nadir unity_nadir_cbm"
    for v in nonoise coop noncoop; do
      for p in $PORTS; do
        uv run owm-envs rollout --out outputs/rollouts/${v}/${p} \
          --env iss-numerical --env-config configs/iss_numerical_${v}_goal.toml \
          --policy dock --port ${p} --episodes 3 --require-dock \
          --render-views fpv,dragon_iso --render-workers 3 --gpu-index 1
      done
    done

`--render-workers` fans out across episodes, not frames: a 3-episode rollout
gets no benefit from more than 3 workers, and a 1-episode rollout gets no
speedup at all. Parallelise a sweep by running several `rollout` invocations
concurrently, not by raising `--render-workers` above the episode count --
which is why the command above uses 3, not the 16 that suits `generate`'s
many-episode splits. Rendering costs about 0.094 s per frame per view; a
docked `iss-numerical` episode is roughly 6300 steps, so one 3-episode,
2-view invocation above takes roughly 20 minutes (the 3 episodes render in
parallel across the 3 workers). The sweep issues 24 such invocations -- 72
episodes total across 3 variants x 8 ports x 3 episodes -- about 8 hours if
run one after another.

Still sequence -- a single docked episode at `harmony_fwd_pma2`, with a
frame written every 6 s (`--frame-stride 120` at dt=0.05):

    uv run owm-envs rollout --out outputs/stills/dock_sequence \
      --env iss-numerical --env-config configs/iss_numerical_coop_goal.toml \
      --policy dock --port harmony_fwd_pma2 --episodes 1 --require-dock \
      --render-views fpv,dragon_iso --frame-stride 120 --gpu-index 1

## Asset acknowledgements

The 3D assets under `src/owm_envs/render/resources/` are third-party works,
reproduced here with the provenance that is known. Their upstream **licence
terms were not recorded** when they were first collected, so the licence column
below reflects what is actually documented rather than an assertion of terms.

| Asset | Source | Licence |
|---|---|---|
| `ISS_base.glb` | NASA 3D Resources / science.nasa.gov | not recorded |
| `spacex_dragon_2_exterior.glb` | Sketchfab — "SpaceX Dragon 2 Exterior" | not recorded |
| `nasa_starmap_2020/` | NASA SVS #4851, cubemapped via jaxry/panorama-to-cubemap | not recorded |
| `moon/moon_small.glb` | Texture from NASA SVS #14959 (CGI Moon Kit), https://svs.gsfc.nasa.gov/14959/. **Mesh geometry source not recorded.** | not recorded |
| `earth/maps/earth_color_fallback.jpg` (8192×4096), `earth_clouds_fallback.jpg` (4096×2048), `earth_bump_fallback.png` (2048×1024) | Downsampled from high-resolution equirectangular imagery collected from the Earth sources listed below | not recorded |

**Earth imagery sources.** The high-resolution equirectangular imagery the
shipped maps are downsampled from was collected from these four:

- https://sketchfab.com/3d-models/earth-41fc80d85dfd480281f21b74b2de2faa
- https://science.nasa.gov/resource/earth-3d-model/
- https://www.cgtrader.com/items/6013252/download-page
- https://maps.drsys.eu/

Which specific source produced each shipped map was not recorded per file, so
anyone assessing terms should check all four.

The three `*_fallback` maps above are the only Earth imagery this repository
ships, and each is a full-globe equirectangular downsample of a much larger
source image. Neither that source imagery nor the full-resolution maps
downsampled from it are redistributed here: both are gitignored, and
`scripts/downsample_earth_maps.py` (or the renderer's own auto-download of the
sources) regenerates the full maps — 16384×8192 for colour and clouds,
8192×4096 for the bump height map — on the machine that renders. The renderer
uses the committed fallbacks whenever those local full maps are absent.

**Before public release.** Licence terms were not recorded for any asset, and
two provenance gaps remain: which of the four Earth sources produced each map,
and the mesh geometry of `moon_small.glb` (its texture is attributed to NASA SVS
#14959). This is not cosmetic — Sketchfab and CGTrader assets carry per-item
terms, some requiring attribution, and NASA imagery has its own usage
guidelines. Confirm terms for each asset before this repository is made public
or redistributed.
