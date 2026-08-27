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

    uv run python scripts/check_sensor_noise.py logs/run1
    uv run python scripts/plot_trajectories_3d.py logs/run1 --out logs/run1_traj

`check_sensor_noise.py` measures the residual between the measured relative
view in each frame's `observation_vector` and the view of its true
`state_vector` -- the sensor-noise draw the simulator actually made -- and
compares its RMS per channel against the sigmas in the run's
`env_config.yaml`, exiting non-zero when any channel is off by more than
`--tolerance` (default 10%). Both scripts resolve which env generated the run
from its `dataset_card.json`, so they read `iss`, `iss-hcw` and
`iss-numerical` runs alike. It also breaks position error down by
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
        --env-config configs/iss-numerical/env/${v}_goal.toml \
        --gen-config configs/iss-numerical/gen/trial.yaml \
        --render --render-workers 16 --gpu-index 1
    done

Full 500k run: same, with the 500k gen-config and output directory:

    for v in nonoise coop noncoop; do
      uv run owm-envs generate --out outputs/full_num_${v}_goal \
        --env-config configs/iss-numerical/env/${v}_goal.toml \
        --gen-config configs/iss-numerical/gen/500k.yaml \
        --render --render-workers 16 --gpu-index 1
    done

Port sweep -- 3 noise variants x 8 ports x 3 successful docked episodes each.
`--port all` splits `--episodes` across the eight ports and holds each to its
own quota, so one call per variant covers the five ports the train split uses
(`harmony_fwd_pma2`, `harmony_nadir_cbm`, `zvezda_aft`, `pirs_nadir`,
`rassvet_nadir`) and the three held out of it (`harmony_zenith_cbm`,
`poisk_zenith`, `unity_nadir_cbm`) at a fixed number of docks each, rather
than at whatever mix uniform port draws and `--require-dock` rejection leave:

    for v in nonoise coop noncoop; do
      uv run owm-envs rollout --out outputs/rollouts/${v} \
        --env iss-numerical --env-config configs/iss-numerical/env/${v}_goal.toml \
        --policy dock --port all --episodes 24 --require-dock \
        --render-views fpv,dragon_iso --render-workers 16 --gpu-index 1
    done

Each call writes a LeRobot dataset under `<out>/rollout` beside the review
clips in `<out>/media/<view>/rollout/`, both encoded from one render pass.
`rollout.json` names the port, seed and outcome of every episode in the order
the dataset holds them; inside the dataset, `dock_target` is the goal pose the
episode flew to. Pass `--no-lerobot` for clips alone.

`--require-dock` retries per port, and per-port dock rates differ widely --
under the cooperative preset they run from 22% at `unity_nadir_cbm` to 53% at
`zvezda_aft` -- so each port gets its own `--max-attempts` cap (20x its quota
by default) and one hard port cannot spend the whole sweep's budget. An
attempt costs about the same whatever its lane count, since the lanes are
vectorised but the horizon is sequential, so the retries are the floor on how
fast a sweep can run.

`--render-workers` fans out across episodes, not frames, so it pays in
proportion to how many episodes one invocation keeps: the 24-episode call
above uses 16, where a 1-episode rollout gets no speedup at all. Each worker
materialises a whole episode's clips and `iter_batch_frames` submits
`workers + 1` episodes before consuming any, so peak memory is roughly
`(workers + 1)` episodes of video -- about 10 GB each at 512x512 over two
views and ~6300 frames. Rendering costs about 0.094 s per frame per view, so
one 24-episode, 2-view call is about 8 GPU-hours, near half an hour of wall
clock across 16 workers, and the three variants about 1.5 hours in sequence.

Still sequence -- a single docked episode at `harmony_fwd_pma2`, with a
frame written every 6 s (`--frame-stride 120` at dt=0.05):

    uv run owm-envs rollout --out outputs/stills/dock_sequence \
      --env iss-numerical --env-config configs/iss-numerical/env/coop_goal.toml \
      --policy dock --port harmony_fwd_pma2 --episodes 1 --require-dock \
      --render-views fpv,dragon_iso --frame-stride 120 --gpu-index 1

## Trajectory files: render and plot any episode

A single episode from any harness -- a scripted policy, an RL checkpoint, a
world-model planner -- can be stored as a directory holding `trajectory.npz`
and `meta.json` (`owm_envs.datasets.trajectory` defines the layout and
validates it on load). Rows are recorded at the environment's integration
step. `meta.json` carries the env name and its inline config, the port and
seed, `dt`, the policy's `rate_hz` and `action_repeat`, and the outcome, so
the file alone is enough to rebuild the scene.

    uv run owm-envs render-trajectory media/rollouts/harmony_fwd_a --views fpv,dragon_iso
    uv run owm-envs plot-trajectory  media/rollouts/harmony_fwd_a

`render-trajectory` writes `<method>_fpv.mp4`, `<method>_iso.mp4`, ... beside
the file at the file's own frame rate (20 fps for a 20 Hz run); `--fps` picks
nearest rows for a slower clip and `--stride N` thins frames for a quick
check. `plot-trajectory` writes `<method>_traj.png` (the full path, coloured
by speed, against the 313-box station hull) and `<method>_traj.mp4` (the same
path growing in step with the episode clock, at `--fps`, default 10). That
video redraws the station hull for every frame, so a full 360 s episode at 10
fps takes minutes -- lower `--fps` for a quick look.

Two harnesses that reset the same env config at the same `(port, seed)` fly
from a bit-identical start, so their files render and plot side by side.

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
| `sun/sun_2k.jpg` (2048×1024) | Solar System Scope, https://www.solarsystemscope.com/textures/ | CC BY 4.0 |
| `earth/maps/earth_color_fallback.jpg` (8192×4096), `earth_clouds_fallback.jpg` (4096×2048), `earth_bump_fallback.png` (2048×1024) | Downsampled from high-resolution equirectangular imagery collected from the Earth sources listed below | not recorded |

**Earth imagery sources.** The high-resolution equirectangular imagery the
shipped maps are downsampled from was collected from these four:

- https://sketchfab.com/3d-models/earth-41fc80d85dfd480281f21b74b2de2faa
- https://science.nasa.gov/resource/earth-3d-model/
- https://www.cgtrader.com/items/6013252/download-page
- https://maps.drsys.eu/

Which specific source produced each shipped map was not recorded per file, so
anyone assessing terms should check all four.

The three `*_fallback` maps above are the only Earth imagery committed to this
repository; both the full-resolution maps and the source imagery are gitignored.
Both are mirrored in the
[`sislaboratory/owm-earth-textures`](https://huggingface.co/datasets/sislaboratory/owm-earth-textures)
dataset, which carries the same provenance and licence disclosure as this
section. The renderer fetches the ~48 MB of full maps from it automatically
when a scene is built with texture downloading enabled; nothing else is
fetched at render time.

To regenerate the maps at a different width, fetch the sources first:

    uv run owm-envs earth pull-sources        # several gigabytes
    uv run owm-envs earth regenerate --kind color,clouds,bump

Resolution prefers a hosted map over downsampling a local source, so
`regenerate` is how a replaced source reaches the renderer.

**Before public release.** Licence terms were not recorded for any asset, and
two provenance gaps remain: which of the four Earth sources produced each map,
and the mesh geometry of `moon_small.glb` (its texture is attributed to NASA SVS
#14959). This is not cosmetic — Sketchfab and CGTrader assets carry per-item
terms, some requiring attribution, and NASA imagery has its own usage
guidelines. Confirm terms for each asset before this repository is made public
or redistributed.
