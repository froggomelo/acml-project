#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$PROJECT_ROOT/.venv"
PYTHON_BIN="${PYTHON_BIN:-}"

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

find_python() {
  local candidate

  if [ -n "$PYTHON_BIN" ]; then
    require_command "$PYTHON_BIN"
    echo "$PYTHON_BIN"
    return
  fi

  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      echo "$candidate"
      return
    fi
  done

  echo "Error: Python 3 is not installed or not on PATH." >&2
  exit 1
}

check_python_version() {
  local python_bin="$1"

  if ! "$python_bin" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
  then
    echo "Error: Python 3.10 or newer is required." >&2
    echo "Set PYTHON_BIN to a compatible interpreter, for example: PYTHON_BIN=python3.12 bash setup.sh" >&2
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

ensure_python_env() {
  local python_bin
  local python_version

  if [ -x "$ENV_DIR/bin/python" ]; then
    echo "Environment already present at $ENV_DIR. Skipping Python environment setup."
    return
  fi

  if [ -d "$ENV_DIR" ]; then
    echo "Error: $ENV_DIR already exists but does not look like a Python virtual environment." >&2
    echo "Remove it and rerun setup.sh, or set ENV_DIR in the script to another path." >&2
    exit 1
  fi

  python_bin="$(find_python)"
  check_python_version "$python_bin"
  python_version="$("$python_bin" --version 2>&1)"

  echo "Creating Python virtual environment at $ENV_DIR using $python_version..."
  if ! "$python_bin" -m venv "$ENV_DIR"; then
    echo "Error: failed to create the virtual environment." >&2
    echo "On Debian/Ubuntu, install venv support with: sudo apt install python3-venv" >&2
    exit 1
  fi

  echo "Installing Python dependencies into $ENV_DIR..."
  "$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
  "$ENV_DIR/bin/python" -m pip install "${CORE_DEPS[@]}" "${NOTEBOOK_DEPS[@]}"
}

ensure_fma_small
ensure_fma_metadata
ensure_python_env

echo "Setup complete."
echo "Activate the virtual environment with: source $ENV_DIR/bin/activate"
