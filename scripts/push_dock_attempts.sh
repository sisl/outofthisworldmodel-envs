#!/usr/bin/env bash
# Upload the dock-attempts-by-port tree to the Hub as one dataset repo.
#
# `owm-envs push` cannot do this: it reads summary.json and a generation
# config, and only `generate` writes those. This does what push does for the
# parts that apply -- build a card, then MIRROR the tree onto the repo
# (--delete '*' removes whatever the tree no longer has).
#
#   ./scripts/push_dock_attempts.sh                    # 20ep tree -> ...-20ep
#   EPISODES=2 ./scripts/push_dock_attempts.sh         # rehearsal tree
#   DRY_RUN=1 ./scripts/push_dock_attempts.sh          # build the card only
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# The episode count picks the tree AND the repo, so runs of different sizes
# cannot mirror over one another on the Hub.
EPISODES=${EPISODES:-20}
OUT_ROOT=${OUT_ROOT:-outputs/v1/dock_attempts_${EPISODES}ep}
REPO_NAME=${REPO_NAME:-owm-iss-numerical-dock-attempts-by-port-${EPISODES}ep}
NAMESPACE=${NAMESPACE:-sislaboratory}
PRIVATE=${PRIVATE:-0}
DRY_RUN=${DRY_RUN:-0}

repo_id="$NAMESPACE/$REPO_NAME"
PAIRS=24

if [ ! -d "$OUT_ROOT" ]; then
  echo "[refusing] $OUT_ROOT does not exist; generate it first with:"
  echo "    EPISODES=$EPISODES ./scripts/rollout_dock_attempts.sh"
  exit 1
fi

# Refuse a partial upload, for the reason push refuses one: the upload is a
# mirror, so a half-finished tree silently deletes the previous complete one.
# Counted rather than merely found, and only for a finalized split:
# meta/info.json exists from the moment the dataset is created and updates as
# episodes are saved, so a killed pair leaves one behind looking healthy.
shopt -s nullglob  # an unmatched glob must count as no dirs, not as one literal
missing=0
for d in "$OUT_ROOT"/*/*/; do
  have=$(python3 scripts/split_episode_count.py "$d/rollout")
  [ "$have" -eq "$EPISODES" ] || {
    echo "[incomplete] $d holds $have of $EPISODES episodes"
    missing=$((missing + 1))
  }
done
count=$(find "$OUT_ROOT" -mindepth 2 -maxdepth 2 -type d | wc -l)
if [ "$missing" -ne 0 ] || [ "$count" -ne "$PAIRS" ]; then
  echo "[refusing] $count/$PAIRS pairs, $missing incomplete -- finish the run first"
  exit 1
fi

uv run python - "$OUT_ROOT" "$REPO_NAME" "$EPISODES" "$repo_id" <<'PY'
import json, pathlib, sys

root, repo_name, episodes = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
repo_id = sys.argv[4]
occupied = {"dragon": "harmony_fwd_pma2", "cygnus": "unity_nadir_cbm",
            "soyuz": "zvezda_aft"}

pairs, total, docked_total, frames_total = [], 0, 0, 0
for meta in sorted(root.glob("*/*/rollout.json")):
    run = json.loads(meta.read_text())
    model, port = meta.parent.parent.name, meta.parent.name
    info = json.loads((meta.parent / "rollout" / "meta" / "info.json").read_text())
    rows = run["episodes"]
    if len(rows) != episodes:
        raise SystemExit(f"{meta.parent} holds {len(rows)} episodes, expected {episodes}")
    docked = sum(r["docked"] for r in rows)
    collided = sum(r["collided"] for r in rows)
    anomaly = occupied[model] == port
    if not anomaly and docked != len(rows):
        raise SystemExit(f"{meta.parent} required docks but {len(rows) - docked} did not dock")
    pairs.append((model, port, anomaly, len(rows), docked, collided,
                  info["total_frames"]))
    total += len(rows)
    docked_total += docked
    frames_total += info["total_frames"]

table = "\n".join(
    f"| {m} | `{p}` | {'**anomaly**' if a else 'dock' } | {n} | {d} | {c} | {f} |"
    for m, p, a, n, d, c, f in pairs
)
anomalies = sum(n for _, _, a, n, _, _, _ in pairs if a)

(root / "README.md").write_text(f"""---
license: mit
task_categories:
- robotics
tags:
- spacecraft
- docking
- iss
- lerobot
- anomaly-detection
---

# {repo_name}

Docking approaches for the `iss-numerical` environment: every docking port on
the station, flown by each of three chaser vehicles, as {len(pairs)} LeRobot
datasets. {total} episodes, {frames_total} frames, {docked_total} ended docked.

Each model is flown to all eight ports under its own station asset and
collision hull. The one port that model's own vehicle already occupies in the
scene cannot be docked to -- the hull carries the berthed vehicle, so the goal
pose lies inside the station and the approach ends on contact. Those three
pairs are the anomalies: {anomalies} episodes flown without a docking
requirement, recording what the approach does instead. Every other pair
required a successful dock.

## Layout

```
<model>/<port>/
  rollout/          LeRobot dataset for that pair
    data/ videos/ meta/
  rollout.json      per-episode seed and outcome
  env_config.toml   the config as flown, which is what makes a run reproducible
  media/fpv/rollout/ep_%04d.mp4
```

Models: `dragon`, `cygnus`, `soyuz`. View recorded: `fpv`, the Dragon
first-person camera, under `observation.images.fpv`.

## Pairs

| model | port | role | episodes | docked | collided | frames |
|---|---|---|---|---|---|---|
{table}

## Loading one pair

Each pair is a self-contained LeRobot dataset under its own
`<model>/<port>/rollout/`, so the repo is fetched first and opened from the
pair's subdirectory; the id passed to `LeRobotDataset` is the one the split
was written with, not the Hub repo.

```python
from huggingface_hub import snapshot_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset

local = snapshot_download("{repo_id}", repo_type="dataset")
ds = LeRobotDataset("iss-numerical/rollout",
                    root=f"{{local}}/dragon/harmony_fwd_pma2/rollout")
```

To fetch one pair rather than all {len(pairs)}, pass
`allow_patterns="dragon/harmony_fwd_pma2/*"` to `snapshot_download`.

## Reproducing one episode

```
uv run owm-envs rollout --out run --env iss-numerical \\
    --env-config <model>/<port>/env_config.toml --policy dock \\
    --port <port> --episodes <wanted> --seed <seed> --no-require-dock
```

and take `batch_index` out of it. `wanted` matters: the driver is built with
`num_envs=wanted`, so the attempt's size sets lane assignment and how much of
the seed's stream each lane consumes.
""")
print(f"[card] {len(pairs)} pairs, {total} episodes, {docked_total} docked "
      f"-> {root / 'README.md'}")
PY

echo "[push] $OUT_ROOT -> $repo_id"
echo "[push] this MIRRORS the tree onto that repo: files it does not have are deleted"
if [ "$DRY_RUN" = 1 ]; then echo "[dry] stopping before upload"; exit 0; fi

private_flag=()
[ "$PRIVATE" = 1 ] && private_flag=(--private)

uv run hf upload "$repo_id" "$OUT_ROOT" . \
  --repo-type dataset "${private_flag[@]}" \
  --delete "*" \
  --commit-message "Dock attempts for every port, per chaser model"

echo "[push] https://huggingface.co/datasets/$repo_id"
