#!/usr/bin/env bash
# Create .venv (Python 3.12) with LeRobot + SmolVLA + LIBERO + this package.
# Kept separate from Isaac Sim / Isaac Lab on purpose.
# Downloads several GB (PyTorch CUDA wheels, LeRobot, LIBERO assets).
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/"; exit 1; }
command -v ffmpeg >/dev/null || echo "WARNING: ffmpeg not found (needed by torchcodec for LIBERO videos): sudo apt install ffmpeg"

if [ ! -d .venv ]; then
  uv venv .venv --python 3.12
fi
source .venv/bin/activate

# LeRobot from PyPI with the policy, simulation and training extras.
# If the 'libero' extra is missing in the PyPI release, install from source instead:
#   git clone https://github.com/huggingface/lerobot.git third_party/lerobot
#   uv pip install -e "third_party/lerobot[smolvla,libero,training]"
uv pip install "lerobot[smolvla,libero,training]"

# This project's own code (corruptions, detectors) + tests.
uv pip install -e ".[test]"

# Record exactly what was installed, for reproducibility.
uv pip freeze > setup/lock.txt
echo "Installed. Versions recorded in setup/lock.txt"
echo "Next: source .venv/bin/activate && bash scripts/00_check_env.sh"
