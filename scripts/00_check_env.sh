#!/usr/bin/env bash
# Check GPU, CUDA in PyTorch, and that LeRobot, SmolVLA and LIBERO import.
set -uo pipefail
cd "$(dirname "$0")/.."
source configs/train.env

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "== Python packages"
python - <<'PY'
import importlib, sys
print("python", sys.version.split()[0])
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
for mod in ["lerobot", "lerobot.policies.smolvla", "libero", "vla_errors"]:
    try:
        m = importlib.import_module(mod)
        print(f"OK   {mod} {getattr(m, '__version__', '')}")
    except Exception as e:  # report every failure, do not stop at the first
        print(f"FAIL {mod}: {type(e).__name__}: {e}")
PY
