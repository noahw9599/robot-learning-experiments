#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="$PROJECT_ROOT/external"
PLAYGROUND_DIR="$EXTERNAL_DIR/mujoco_playground"

echo "== Project Jiraiya MuJoCo Playground Setup =="
echo "Project root: $PROJECT_ROOT"
echo

if ! command -v git >/dev/null 2>&1; then
  echo "git is required for the source install."
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Installing uv with the official installer."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

mkdir -p "$EXTERNAL_DIR"

if [ ! -d "$PLAYGROUND_DIR/.git" ]; then
  echo "Cloning MuJoCo Playground source."
  git clone https://github.com/google-deepmind/mujoco_playground.git "$PLAYGROUND_DIR"
else
  echo "MuJoCo Playground source already exists. Pulling latest changes."
  git -C "$PLAYGROUND_DIR" pull --ff-only
fi

cd "$PLAYGROUND_DIR"

echo
echo "Creating Python 3.12 virtual environment with uv."
uv venv --python 3.12

echo
echo "Activating environment."
# shellcheck disable=SC1091
source .venv/bin/activate

echo
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU detected. Installing CUDA 12 JAX."
  uv pip install -U "jax[cuda12]" --index-url https://pypi.org/simple
else
  echo "No NVIDIA GPU detected. Installing CPU JAX."
  uv pip install -U jax --index-url https://pypi.org/simple
fi

echo
echo "Checking JAX backend."
python -c "import jax; print('JAX backend:', jax.default_backend())"

echo
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "Installing MuJoCo Playground from source with all extras."
  uv --no-config sync --all-extras
else
  echo "Installing MuJoCo Playground from source without CUDA extras."
  uv --no-config sync --extra dev --extra notebooks --extra learning
fi

echo
echo "Verifying MuJoCo Playground import."
uv --no-config run python -c "import mujoco_playground; print('MuJoCo Playground import: success')"

if [ "${JIRAIYA_CLEAN_UV_CACHE:-1}" = "1" ]; then
  echo
  echo "Cleaning uv download cache to save disk space."
  uv cache clean || true
fi

echo
echo "Optionally prefetch a locomotion environment:"
echo "uv --no-config run python -c \"from mujoco_playground import locomotion; locomotion.load('G1JoystickFlatTerrain')\""
echo
echo "Setup complete."