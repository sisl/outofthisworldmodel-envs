#!/usr/bin/env bash
# Upload the dock-success run to the Hub as one dataset repo.
#
# `owm-envs push` cannot do this: it reads summary.json and a generation
# config, and only `generate` writes those. This does what push does for the
# parts that apply -- build a card, then MIRROR the run onto the repo
# (--delete '*' removes whatever the run no longer has).
#
#   ./scripts/push_dock_success.sh
#   DRY_RUN=1 ./scripts/push_dock_success.sh     # build the card, upload nothing
#   NAMESPACE=duncaneddy ./scripts/push_dock_success.sh
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

EPISODES=${EPISODES:-100}
OUT=${OUT:-outputs/v1/dock_success_${EPISODES}ep}
REPO_NAME=${REPO_NAME:-owm-iss-numerical-dock-success-${EPISODES}ep}
NAMESPACE=${NAMESPACE:-sislaboratory}
PRIVATE=${PRIVATE:-0}
DRY_RUN=${DRY_RUN:-0}

repo_id="$NAMESPACE/$REPO_NAME"

# Refuse a partial upload, for the reason push refuses one: the upload is a
# mirror, so a half-finished run silently deletes the previous complete one.
if [ ! -f "$OUT/rollout/meta/info.json" ]; then
  echo "[refusing] $OUT holds no finished LeRobot split; generate it first with:"
  echo "    EPISODES=$EPISODES ./scripts/rollout_dock_success.sh"
  exit 1
fi

uv run python - "$OUT" "$REPO_NAME" "$EPISODES" "$repo_id" <<'PY'
import collections, json, pathlib, sys

root, repo_name, episodes = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
repo_id = sys.argv[4]
run = json.loads((root / "rollout.json").read_text())
rows = run["episodes"]
if len(rows) != episodes:
    raise SystemExit(f"{root} holds {len(rows)} episodes, expected {episodes}")
undocked = [r for r in rows if not r["docked"]]
if undocked:
    raise SystemExit(f"{len(undocked)} episodes did not dock; this run is not all-success")

by_port = collections.Counter(r["port"] for r in rows)
held_out = {"harmony_zenith_cbm", "poisk_zenith", "unity_nadir_cbm"}
dt = 0.05
table = "\n".join(
    f"| `{port}` | {'held out' if port in held_out else 'training'} | {count} | "
    f"{sum(r['steps'] for r in rows if r['port'] == port) / count * dt:.0f} |"
    for port, count in by_port.items()
)
info = json.loads((root / "rollout" / "meta" / "info.json").read_text())

(root / "README.md").write_text(f"""---
license: mit
task_categories:
- robotics
tags:
- spacecraft
- docking
- iss
- lerobot
---

# {repo_name}

{episodes} successful docking approaches to the nominal ISS, as one LeRobot
dataset, flown by the scripted `dock` policy in the `iss-numerical`
environment under the cooperative sensor-noise preset.

Every episode ended docked. The station is the base asset (`ISS_base.glb`),
which berths no visiting vehicle, so all eight ports are free. Each port has
its own quota rather than its share of a uniform draw, so the three ports the
public training splits hold out are represented as heavily as the five they
use -- which a uniform draw would not do, since rejection sampling favours
whichever ports dock most easily.

## Layout

```
rollout/            LeRobot dataset: {info['total_episodes']} episodes, {info['total_frames']} frames
  data/ videos/ meta/
rollout.json        per-episode port, seed and outcome
env_config.toml     the config as flown, which is what makes a run reproducible
media/fpv/rollout/  ep_%04d.mp4, the same clips as review copies
```

`observation.images.fpv` is the Dragon first-person camera. `dock_target` is
the (1, 7) [position, quaternion] goal pose the episode flew to, which is what
identifies the port inside the dataset; `rollout.json` names the port of every
episode in the order the dataset holds them.

## Ports

| port | split role | episodes | mean duration (s) |
|---|---|---|---|
{table}

## Loading

The LeRobot files live under `rollout/` rather than at the repo root, which is
how the `owm-envs` datasets are laid out -- each split is a self-contained
LeRobot dataset in its own directory. So the repo is fetched first and opened
from that subdirectory; the id passed to `LeRobotDataset` is the one the split
was written with, not the Hub repo.

```python
from huggingface_hub import snapshot_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset

local = snapshot_download("{repo_id}", repo_type="dataset")
ds = LeRobotDataset("iss-numerical/rollout", root=f"{{local}}/rollout")
```

## Reproducing one episode

```
uv run owm-envs rollout --out run --env iss-numerical \\
    --env-config env_config.toml --policy dock \\
    --port <port from rollout.json> --episodes <wanted> \\
    --seed <seed> --no-require-dock
```

and take `batch_index` out of it. `wanted` matters: the driver is built with
`num_envs=wanted`, so the attempt's size sets lane assignment and how much of
the seed's stream each lane consumes.
""")
print(f"[card] {len(rows)} episodes, all docked, {len(by_port)} ports -> {root / 'README.md'}")
PY

echo "[push] $OUT -> $repo_id"
echo "[push] this MIRRORS the run onto that repo: files it does not have are deleted"
if [ "$DRY_RUN" = 1 ]; then echo "[dry] stopping before upload"; exit 0; fi

private_flag=()
[ "$PRIVATE" = 1 ] && private_flag=(--private)

uv run hf upload "$repo_id" "$OUT" . \
  --repo-type dataset "${private_flag[@]}" \
  --delete "*" \
  --commit-message "Successful docks at every port, nominal station"

echo "[push] https://huggingface.co/datasets/$repo_id"
