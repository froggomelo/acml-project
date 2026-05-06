#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$PROJECT_ROOT/.venv"

FMA_SMALL_URL="https://os.unil.cloud.switch.ch/fma/fma_small.zip"
FMA_METADATA_URL="https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"

FMA_SMALL_ZIP="$PROJECT_ROOT/fma_small.zip"
FMA_METADATA_ZIP="$PROJECT_ROOT/fma_metadata.zip"

FMA_SMALL_SENTINEL="$PROJECT_ROOT/fma_small/000/000002.mp3"
FMA_METADATA_SENTINELS=(
  "$PROJECT_ROOT/fma_metadata/tracks.csv"
  "$PROJECT_ROOT/fma_metadata/features.csv"
  "$PROJECT_ROOT/fma_metadata/genres.csv"
)

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

require_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: required command '$cmd' is not installed or not on PATH." >&2
    exit 1
  fi
}

download_file() {
  local url="$1"
  local output_path="$2"

  echo "Downloading $(basename "$output_path")..."
  curl -L --fail --output "$output_path" "$url"
}

ensure_fma_small() {
  if [ -f "$FMA_SMALL_SENTINEL" ]; then
    echo "FMA small audio already present. Skipping download."
    return
  fi

  require_command curl
  require_command unzip

  download_file "$FMA_SMALL_URL" "$FMA_SMALL_ZIP"

  echo "Extracting $(basename "$FMA_SMALL_ZIP")..."
  unzip -q "$FMA_SMALL_ZIP" -d "$PROJECT_ROOT"
  rm -f "$FMA_SMALL_ZIP"
}

ensure_fma_metadata() {
  local missing=0
  local sentinel

  for sentinel in "${FMA_METADATA_SENTINELS[@]}"; do
    if [ ! -f "$sentinel" ]; then
      missing=1
      break
    fi
  done

  if [ "$missing" -eq 0 ]; then
    echo "FMA metadata already present. Skipping download."
    return
  fi

  require_command curl
  require_command unzip

  download_file "$FMA_METADATA_URL" "$FMA_METADATA_ZIP"

  echo "Extracting $(basename "$FMA_METADATA_ZIP")..."
  unzip -q "$FMA_METADATA_ZIP" -d "$PROJECT_ROOT"
  rm -f "$FMA_METADATA_ZIP"
}

ensure_conda_env() {
  if [ -d "$ENV_DIR" ]; then
    echo "Environment already present at $ENV_DIR. Skipping Conda setup."
    return
  fi

  require_command conda

  echo "Creating Conda environment at $ENV_DIR..."
  conda create --yes --prefix "$ENV_DIR" python=3.12 pip

  echo "Installing Python dependencies into $ENV_DIR..."
  conda run --prefix "$ENV_DIR" python -m pip install --upgrade pip setuptools wheel
  conda run --prefix "$ENV_DIR" python -m pip install "${CORE_DEPS[@]}" "${NOTEBOOK_DEPS[@]}"
}

ensure_fma_small
ensure_fma_metadata
ensure_conda_env

echo "Setup complete."
echo "Activate the Conda environment with: conda activate $ENV_DIR"
