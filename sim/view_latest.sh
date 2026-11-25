#!/usr/bin/env bash
# Open an Isaac Sim window showing 4 Franka arms running the newest overnight checkpoint.
# Safe beside training: closes itself if free RAM drops below 1.5 GB.
unset PYTHONPATH
cd ~/IsaacLab
R=franka_lift_overnight
L=$(ls -t logs/rsl_rl/$R/*/model_*.pt | head -1)
echo "viewing $L"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Lift-Cube-Franka-Play-v0 \
  --num_envs 4 --viz kit --experiment_name $R --load_run "$(basename "$(dirname "$L")")" \
  --checkpoint "$L" --real-time > logs/overnight/view.log 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do
  a=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  if [ "$a" -lt 1500000 ]; then
    pkill -P $PID; kill $PID; echo "$(date) viewer closed: low RAM" >> logs/overnight/supervisor.log
  fi
  sleep 5
done
