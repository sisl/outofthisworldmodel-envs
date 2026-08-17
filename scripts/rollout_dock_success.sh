#!/usr/bin/env bash
# Successful docks at every port of the nominal station, as one dataset.
#
# One `rollout` call covers the whole sweep: --port all splits --episodes
# across the eight ports and holds each to its own quota, so the result is a
# fixed number of docks PER PORT rather than whatever uniform port draws and
# --require-dock rejection happen to leave. The three ports the public
# training splits hold out -- harmony_zenith_cbm, poisk_zenith,
# unity_nadir_cbm -- get the same share as the five they use.
#
# The station is the base asset, which berths no visiting vehicle, so all
# eight ports are free and every episode can dock. Compare
# rollout_dock_attempts.sh, which flies the three berthed variants and whose
# occupied ports cannot be docked to at all.
#
#   ./scripts/rollout_dock_success.sh                # 100 episodes, ~2 h
#   EPISODES=8 RENDER_WORKERS=8 ./scripts/rollout_dock_success.sh   # rehearsal
#
# Measured per-port dock rates under coop_goal.toml run from 22%
# (unity_nadir_cbm) to 53% (zvezda_aft), so a 13-episode quota needs 25-60
# episodes rolled against a cap of 20x13. No port comes near the cap, but the
# retries are the floor on the run: an attempt costs about the same whatever
# its lane count, since the lanes are vectorised and the horizon is not.
#
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

EPISODES=${EPISODES:-100}
STEPS=${STEPS:-7200}
GPU_INDEX=${GPU_INDEX:-0}
RENDER_WORKERS=${RENDER_WORKERS:-12}
RENDER_VIEWS=${RENDER_VIEWS:-fpv}
ENV_CONFIG=${ENV_CONFIG:-configs/iss-numerical/env/coop_goal.toml}
OUT=${OUT:-outputs/v1/dock_success_${EPISODES}ep}
LOG_DIR=${LOG_DIR:-logs}
LOG=${LOG:-$LOG_DIR/dock_success_${EPISODES}ep.log}

mkdir -p "$LOG_DIR"

# meta/info.json is written when the split is finalized, so its presence is
# what separates a finished run from a killed one. A killed run leaves a
# corrupt parquet tail and cannot be resumed, which is why the partial is
# deleted rather than continued.
if [ -f "$OUT/rollout/meta/info.json" ]; then
  echo "[skip] $OUT already complete"
  exit 0
fi
if [ -e "$OUT" ]; then
  echo "[clean] removing partial $OUT"
  rm -rf "${OUT:?}"
fi

echo "[run] $EPISODES episodes over all 8 ports -> $OUT"
echo "[run] log: $LOG"

# stderr is kept but filtered: this environment's CUDA stack prints a cuDNN
# version-mismatch line per device probe, which buries anything real.
# --line-buffered because grep blocks its output otherwise, and a log that
# only fills in when a two-hour run ends is no log at all.
if uv run owm-envs rollout --out "$OUT" \
    --env iss-numerical --env-config "$ENV_CONFIG" \
    --policy dock --port all --episodes "$EPISODES" --require-dock \
    --steps "$STEPS" --render-views "$RENDER_VIEWS" \
    --render-workers "$RENDER_WORKERS" --gpu-index "$GPU_INDEX" \
    2>&1 | grep --line-buffered -v 'Loaded runtime CuDNN library' | tee "$LOG"; then
  echo "[done] $OUT"
else
  echo "[FAIL] see $LOG"
  exit 1
fi
