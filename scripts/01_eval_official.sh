#!/usr/bin/env bash
# Pipeline check: evaluate the official SmolVLA LIBERO checkpoint on one suite.
# Usage: bash scripts/01_eval_official.sh [suite] [episodes_per_task] [policy]
# Compare the printed success rate with the published level before trusting
# any other number from this setup.
set -euo pipefail
cd "$(dirname "$0")/.."
source configs/train.env

SUITE_ARG=${1:-$SUITE}
EPISODES=${2:-10}
POLICY=${3:-lerobot/smolvla_libero}
OUT=results/eval/official_$(basename "$POLICY")_${SUITE_ARG}_$(date +%Y%m%d-%H%M)

lerobot-eval \
  --policy.path="$POLICY" \
  --env.type=libero \
  --env.task="$SUITE_ARG" \
  --eval.batch_size=1 \
  --eval.n_episodes="$EPISODES" \
  --env.init_states=true \
  --rename_map="${RENAME_MAP:-{\"observation.images.image\": \"observation.images.camera1\", \"observation.images.image2\": \"observation.images.camera2\"}}" \
  --policy.n_action_steps="$N_ACTION_STEPS" \
  --seed="$EVAL_SEED" \
  --output_dir="$OUT"

echo "Saved to $OUT"
