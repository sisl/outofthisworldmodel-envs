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

`--noise off|cooperative|noncooperative` overrides the `--config` file's
`sensor_noise` with a named preset; omit it to leave the config's own setting
in place.

`--goal-error/--no-goal-error` overrides the `--config` file's
`observation.goal_error`, appending the dock-goal error block to
observations; omit it to leave the config's own setting in place.

`--orbit/--no-orbit` overrides the `--config` file's `orbit.enabled`; omit
it to leave the config's own setting in place. See "Orbital dynamics"
below for what it turns on.

## Orbital dynamics

`ISSConfig.orbit` (`OrbitConfig`, off by default) adds Hill-Clohessy-Wiltshire
(HCW/CW) relative orbital dynamics on top of the docking physics: the
chaser's translational equations of motion gain the CW relative
accelerations (tidal stretching along the radial axis, Coriolis coupling
between the radial and along-track axes) driven by the mean motion `n` of a
configured ISS reference orbit, and its rotational equations gain the
gravity-gradient torque that a real spacecraft experiences from being
slightly off from the local vertical. With `orbit.enabled = False` (the
default), none of this is evaluated and the simulation is bit-identical to
before this feature existed.

**What is not modeled.** The chief (ISS reference orbit) propagates as
two-body Keplerian motion, with no J2 oblateness or drag; the CW relative
dynamics are linearized about a circular reference orbit; and the attitude
dynamics treat the LVLH frame as inertial, neglecting the O(n) frame-rotation
terms that a rotating LVLH frame would otherwise contribute.

**Frame convention.** World coordinates ARE the LVLH (local-vertical,
local-horizontal) frame of the reference orbit: `+z` is radial, pointing
away from Earth ("up"); `-y` is along-track (the direction of orbital
motion); `+x` is cross-track, completing a right-handed triad. This follows
from the renderer's existing placement of the Earth (straight below the
chaser, along `-z`) and the dock port's existing orientation (facing the ram
direction, `-y`) -- the orbital-dynamics feature aligns to those rather than
introducing a second frame.

**Epoch and per-episode sampling.** `orbit.epoch` (an ISO 8601 UTC
timestamp) plus the six classical orbital elements (`sma_m`, `ecc`,
`inc_deg`, `raan_deg`, `argp_deg`, `mean_anomaly_deg`) define the reference
orbit at `t = 0`. Each episode additionally samples a start-time offset
uniformly from `orbit.epoch_offset_range_s` -- so different episodes begin
at different points along the reference orbit and, when lighting follows the
epoch (below), under different sun/Earth-rotation conditions. The sampled
offset is recorded per episode as `epoch_offset_s`: the dataset
(`batch.epoch_offsets`, and the corresponding LeRobot feature when
`orbit.enabled`) is the source of truth. The gym envs' `info["epoch_offset_s"]`
reflects only an offset the caller explicitly provides via
`reset(options={"epoch_offset_s": ...})` -- the dataset-generation drivers
sample and record the offset themselves without threading it back through
`reset()`, so `info` stays at its default (`0.0`) during generation.

**Lighting.** The renderer's sun direction and Earth spin can follow the
same epoch: set `render.sun_from_epoch = True` (with `render.orbit` giving
the same elements/epoch as `orbit`, or a dict of `OrbitConfig` fields) and
each rendered frame's sun direction and Earth rotation angle are computed
from `ReferenceOrbit.sun_direction_world` at that frame's simulation time
past the epoch, replacing the static configured `sun_direction_world`. Off
by default: rendering is unaffected unless this is turned on explicitly. The
`generate` CLI wires this automatically -- `--orbit --render` follows the
epoch without a separate `render.sun_from_epoch` setting -- unless the
`--config` file's `render` block sets `sun_from_epoch` itself, which wins.

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
