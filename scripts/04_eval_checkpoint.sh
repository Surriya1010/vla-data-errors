#!/usr/bin/env bash
# Evaluate a fine-tuned checkpoint with the shared evaluation settings.
# Usage: bash scripts/04_eval_checkpoint.sh <condition> [seed]
set -euo pipefail
cd "$(dirname "$0")/.."
source configs/train.env

COND=${1:?condition name}
SEED=${2:-${SEEDS%% *}}
CKPT=outputs/train/${COND}_seed${SEED}/checkpoints/last/pretrained_model
OUT=results/eval/${COND}_seed${SEED}

[ -d "$CKPT" ] || { echo "Missing checkpoint $CKPT"; exit 1; }

lerobot-eval \
  --policy.path="$CKPT" \
  --env.type=libero \
  --env.task="$SUITE" \
  --eval.batch_size=1 \
  --eval.n_episodes="$EVAL_EPISODES" \
  --env.init_states=true \
  --policy.n_action_steps="$N_ACTION_STEPS" \
  --seed="$EVAL_SEED" \
  --output_dir="$OUT"

echo "Saved to $OUT"
