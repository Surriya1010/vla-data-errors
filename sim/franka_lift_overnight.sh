#!/usr/bin/env bash
# Overnight PPO training of the Franka Panda cube-lift task in Isaac Lab (rsl_rl).
# - headless, survives closing the terminal/VS Code (start with setsid, see bottom)
# - blocks sleep / lid-close suspend while running
# - resumes from the latest checkpoint after any crash (envs halved after OOM)
# - stops at DEADLINE_HOURS or when the file $OUT/STOP exists
set -uo pipefail
unset PYTHONPATH                      # ROS python3.14 paths break Isaac's python
ISAACLAB=~/IsaacLab
TASK=Isaac-Lift-Cube-Franka-v0
EXP=franka_lift_overnight
NUM_ENVS=${NUM_ENVS:-4096}
DEADLINE_HOURS=${DEADLINE_HOURS:-10}
MAX_RESTARTS=30
OUT=$ISAACLAB/logs/overnight
LOGROOT=$ISAACLAB/logs/rsl_rl/$EXP
mkdir -p "$OUT"
end=$(( $(date +%s) + DEADLINE_HOURS*3600 ))
log(){ echo "$(date '+%F %T') $*" | tee -a "$OUT/supervisor.log"; }

# GPU health log every 60 s
( while sleep 60; do echo "$(date '+%F %T'),$(nvidia-smi --query-gpu=temperature.gpu,memory.used,utilization.gpu,power.draw --format=csv,noheader)" >> "$OUT/gpu.csv"; done ) &
GPU_PID=$!
trap 'kill $GPU_PID 2>/dev/null' EXIT

restarts=0
while [ "$(date +%s)" -lt "$end" ] && [ ! -f "$OUT/STOP" ] && [ $restarts -le $MAX_RESTARTS ]; do
  latest=$(ls -t "$LOGROOT"/*/model_*.pt 2>/dev/null | head -1)
  resume=()
  if [ -n "$latest" ]; then
    resume=(--resume --load_run "$(basename "$(dirname "$latest")")" --checkpoint "$(basename "$latest")")
    log "resuming from $latest"
  else
    log "fresh start"
  fi
  left=$(( end - $(date +%s) ))
  log "train: envs=$NUM_ENVS, ${left}s left, restart #$restarts"
  ( cd "$ISAACLAB" && timeout --signal=INT --kill-after=60 "$left" \
      ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
      --task "$TASK" --num_envs "$NUM_ENVS" --headless --max_iterations 100000 \
      --experiment_name "$EXP" "${resume[@]}" ) >> "$OUT/train.log" 2>&1
  rc=$?
  log "train exited rc=$rc"
  [ $rc -eq 124 ] && break
  if tail -200 "$OUT/train.log" | grep -qiE "out of memory|CUDA error: out"; then
    NUM_ENVS=$(( NUM_ENVS > 512 ? NUM_ENVS/2 : 512 )); log "OOM -> envs=$NUM_ENVS"
  fi
  restarts=$((restarts+1)); sleep 30
done
log "done. latest checkpoint: $(ls -t "$LOGROOT"/*/model_*.pt 2>/dev/null | head -1)"

# Start (from anywhere):
#   setsid systemd-inhibit --what=sleep:idle:handle-lid-switch --why="Franka RL overnight" \
#     bash ~/vla-data-errors/sim/franka_lift_overnight.sh > /dev/null 2>&1 < /dev/null &
# Stop:   touch ~/IsaacLab/logs/overnight/STOP   (or: pkill -INT -f franka_lift_overnight)
# Watch:  tail -f ~/IsaacLab/logs/overnight/supervisor.log
#         tail -f ~/IsaacLab/logs/overnight/train.log | grep -E "Learning iteration|Mean reward"
