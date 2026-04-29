#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

CORE_DEPS=(
  numpy
  pandas
  scipy
  matplotlib
  seaborn
  scikit-learn
  librosa
  audioread
  mutagen
  pydub
  requests
  pydot
  tqdm
  python-dotenv
  notebook
  ipywidgets
)

NOTEBOOK_DEPS=(
  soundfile
  torch
  torchaudio
)

if [ -d "$VENV_DIR" ]; then
  echo "Using existing Python virtual environment at $VENV_DIR..."
else
  echo "Creating Python virtual environment at $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
fi

echo "Activating virtual environment..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Upgrading packaging tools..."
python -m pip install --upgrade pip setuptools wheel

echo "Installing project dependencies..."
python -m pip install "${CORE_DEPS[@]}" "${NOTEBOOK_DEPS[@]}"

echo "Setup complete! Virtual environment is ready."
echo "To activate the virtual environment in the future, run: source .venv/bin/activate"
