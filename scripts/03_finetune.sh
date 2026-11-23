#!/usr/bin/env bash
# Fine-tune SmolVLA on one dataset condition, with the shared recipe.
# Usage: bash scripts/03_finetune.sh <condition> [seed]
# Requires: python scripts/02_make_corrupted.py --condition <condition>
set -euo pipefail
cd "$(dirname "$0")/.."
source configs/train.env

COND=${1:?condition name from configs/conditions.yaml}
SEED=${2:-${SEEDS%% *}}
ROOT=data/conditions/$COND
EPISODES_FILE=data/suite_episodes_${SUITE}.txt

[ -d "$ROOT" ] || { echo "Missing $ROOT. Run scripts/02_make_corrupted.py --condition $COND"; exit 1; }
[ -f "$EPISODES_FILE" ] || { echo "Missing $EPISODES_FILE. Run scripts/02_make_corrupted.py --list-suite-episodes"; exit 1; }

lerobot-train \
  --policy.path="$BASE_POLICY" \
  --policy.push_to_hub=false \
  --dataset.repo_id="$DATASET_REPO" \
  --dataset.root="$ROOT" \
  --dataset.episodes="$(cat "$EPISODES_FILE")" \
  --dataset.video_backend=torchcodec \
  --steps="$STEPS" \
  --batch_size="$BATCH_SIZE" \
  --seed="$SEED" \
  --output_dir="outputs/train/${COND}_seed${SEED}"
