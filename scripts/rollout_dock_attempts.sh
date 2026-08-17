#!/usr/bin/env bash
# Dock attempts at every port, for every chaser model.
#
# Run with bash, not zsh: `for p in $PORTS` does not word-split in zsh, which
# would turn the eight ports into one argument.
#
#   ./scripts/rollout_dock_attempts.sh                # 20 episodes per pair
#   EPISODES=2 JOBS=2 ./scripts/rollout_dock_attempts.sh    # rehearsal
#
# Each model flies to all eight ports and each pair gets its own directory, so
# the tree stays <model>/<port>/ and a pair can be re-rolled on its own. Every
# pair writes a LeRobot dataset under its rollout/ beside the review clips.
#
# The one port a model's own vehicle already occupies in the scene cannot be
# docked to -- that variant's collision hull carries the berthed vehicle, so
# the goal pose lies inside the station and the approach ends on contact.
# Those three pairs are the anomalies: they are flown WITHOUT --require-dock,
# and what they record is whatever the approach does instead.
#
# Cost, from 0.094 s per frame per view and ~5600 steps for a docked episode:
# 3 x 8 x 20 = 480 episodes is about 70 GPU-hours of fpv render, near 4.5 h of
# wall clock at JOBS x RENDER_WORKERS = 16. Each invocation holds about
# (RENDER_WORKERS + 1) episodes of video in memory, ~5 GB each at one view.
#
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Exported because the per-rollout body below runs in a shell xargs spawns,
# which inherits the environment and nothing else.
export EPISODES=${EPISODES:-20}
export STEPS=${STEPS:-7200}
export GPU_INDEX=${GPU_INDEX:-0}
export RENDER_WORKERS=${RENDER_WORKERS:-4}
export RENDER_VIEWS=${RENDER_VIEWS:-fpv}
# Suffixed with the episode count so runs of different sizes cannot land on
# each other: the per-rollout body clears its output directory before writing.
export OUT_ROOT=${OUT_ROOT:-outputs/v1/dock_attempts_${EPISODES}ep}
export LOG_DIR=${LOG_DIR:-logs/dock_attempts_${EPISODES}ep}
JOBS=${JOBS:-4}
DRY_RUN=${DRY_RUN:-0}
export DRY_RUN

PORTS=(
  harmony_fwd_pma2 harmony_zenith_cbm harmony_nadir_cbm zvezda_aft
  poisk_zenith pirs_nadir rassvet_nadir unity_nadir_cbm
)
MODELS=(dragon cygnus soyuz)

occupied_port() {
  case $1 in
    dragon) echo harmony_fwd_pma2 ;;
    cygnus) echo unity_nadir_cbm ;;
    soyuz)  echo zvezda_aft ;;
  esac
}

mkdir -p "$LOG_DIR"

for m in "${MODELS[@]}"; do
  occupied=$(occupied_port "$m")
  for p in "${PORTS[@]}"; do
    if [ "$p" = "$occupied" ]; then req=--no-require-dock; else req=--require-dock; fi
    echo "$m $p $req"
  done
done | xargs -P "$JOBS" -L 1 bash -c '
  set -uo pipefail
  m=$1; p=$2; req=$3
  out="$OUT_ROOT/$m/$p"
  # Completion is read from rollout.json, which rollout writes last, after
  # the render and the dataset write. Nothing lerobot writes says the same:
  # meta/info.json is created with the dataset and meta/episodes is flushed
  # every ten episodes, so both look healthy in a pair killed halfway.
  # No apostrophes in this body -- it is single-quoted for xargs.
  have=$(python3 scripts/rollout_episode_count.py "$out")
  if [ "$have" -eq "$EPISODES" ]; then echo "[skip] $m/$p"; exit 0; fi
  if [ "$have" -ne 0 ]; then echo "[clean] $m/$p holds $have of $EPISODES"; fi
  if [ "$DRY_RUN" = 1 ]; then
    echo "[dry] $m/$p $req -> $out"
    exit 0
  fi
  # A killed run leaves a corrupt parquet tail and cannot be resumed, and the
  # writer refuses to write over a split that is already there.
  rm -rf "${out:?}"
  if uv run owm-envs rollout --out "$out" \
      --env iss-numerical --env-config "configs/iss-numerical/env/coop_${m}.toml" \
      --policy dock --port "$p" --episodes "$EPISODES" "$req" \
      --steps "$STEPS" --render-views "$RENDER_VIEWS" \
      --render-workers "$RENDER_WORKERS" --gpu-index "$GPU_INDEX" \
      > "$LOG_DIR/${m}_${p}.log" 2>&1; then
    echo "[done] $m/$p"
  else
    echo "[FAIL] $m/$p -- see $LOG_DIR/${m}_${p}.log"
    exit 1
  fi
' _
status=$?

echo
if [ "$status" -eq 0 ]; then
  echo "[all done] $((${#MODELS[@]} * ${#PORTS[@]})) rollouts under $OUT_ROOT"
else
  echo "[incomplete] at least one rollout failed; grep '\[FAIL\]' above"
fi
exit "$status"
