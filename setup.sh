#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$PROJECT_ROOT/.venv"
PYTHON_BIN="${PYTHON_BIN:-}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.11.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-$PYTORCH_VERSION}"
PYTORCH_BUILD="${PYTORCH_BUILD:-auto}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-}"

read_env_file_value() {
  local key="$1"
  local line

  if [ ! -f "$PROJECT_ROOT/.env" ]; then
    return 1
  fi

  line="$(grep -E "^[[:space:]]*${key}=" "$PROJECT_ROOT/.env" | head -n1 || true)"
  if [ -z "$line" ]; then
    return 1
  fi

  line="${line#*=}"
  line="${line%$'\r'}"
  line="${line%\"}"
  line="${line#\"}"
  printf '%s\n' "$line"
}

is_truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

# Directory where FMA zips are downloaded and extracted.
# Defaults to the project root; override with DATASET_DIR=... bash setup.sh
# or by setting DATASET_DIR in your .env file.
if [ -z "${DATASET_DIR:-}" ] && [ -f "$PROJECT_ROOT/.env" ]; then
  DATASET_DIR="$(read_env_file_value DATASET_DIR || true)"
fi
DATASET_DIR="${DATASET_DIR:-$PROJECT_ROOT}"
mkdir -p "$DATASET_DIR"

FMA_SMALL_URL="https://os.unil.cloud.switch.ch/fma/fma_small.zip"
FMA_MEDIUM_URL="https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
FMA_METADATA_URL="https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"

FMA_SMALL_ZIP="$DATASET_DIR/fma_small.zip"
FMA_MEDIUM_ZIP="$DATASET_DIR/fma_medium.zip"
FMA_METADATA_ZIP="$DATASET_DIR/fma_metadata.zip"

FMA_SMALL_SENTINEL="$DATASET_DIR/fma_small/000/000002.mp3"
# Track 000003.mp3 is subset=medium and lives only in fma_medium/.
FMA_MEDIUM_SENTINEL="$DATASET_DIR/fma_medium/000/000003.mp3"
FMA_METADATA_SENTINELS=(
  "$DATASET_DIR/fma_metadata/tracks.csv"
  "$DATASET_DIR/fma_metadata/features.csv"
  "$DATASET_DIR/fma_metadata/genres.csv"
)

if [ -z "${DATASET_SIZE:-}" ]; then
  DATASET_SIZE="$(read_env_file_value DATASET_SIZE || true)"
fi
DATASET_SIZE="${DATASET_SIZE:-small}"
DATASET_SIZE="${DATASET_SIZE,,}"
case "$DATASET_SIZE" in
  small|medium|both) ;;
  *)
    echo "Error: DATASET_SIZE must be small, medium, or both. Got '$DATASET_SIZE'." >&2
    exit 1
    ;;
esac

# Set DOWNLOAD_SPECTROGRAMS=1 to generate spectrogram .npy files during setup.
# PREPROCESS_FOR chooses which formats: cnn, crnn, both, or none.
if [ -z "${DOWNLOAD_SPECTROGRAMS:-}" ]; then
  DOWNLOAD_SPECTROGRAMS="$(read_env_file_value DOWNLOAD_SPECTROGRAMS || true)"
fi
DOWNLOAD_SPECTROGRAMS="${DOWNLOAD_SPECTROGRAMS:-0}"
if is_truthy "$DOWNLOAD_SPECTROGRAMS"; then
  DOWNLOAD_SPECTROGRAMS=1
else
  DOWNLOAD_SPECTROGRAMS=0
fi

if [ -z "${PREPROCESS_FOR:-}" ]; then
  PREPROCESS_FOR="$(read_env_file_value PREPROCESS_FOR || true)"
fi
PREPROCESS_FOR="${PREPROCESS_FOR:-both}"
PREPROCESS_FOR="${PREPROCESS_FOR,,}"

# Set DOWNLOAD_MEDIUM=1 to also download fma_medium (~22 GB).
# DATASET_SIZE=both in .env also enables this automatically.
# Example: DOWNLOAD_MEDIUM=1 bash setup.sh
if [ -z "${DOWNLOAD_MEDIUM:-}" ]; then
  DOWNLOAD_MEDIUM="$(read_env_file_value DOWNLOAD_MEDIUM || true)"
fi
DOWNLOAD_MEDIUM="${DOWNLOAD_MEDIUM:-0}"
if is_truthy "$DOWNLOAD_MEDIUM"; then
  DOWNLOAD_MEDIUM=1
else
  DOWNLOAD_MEDIUM=0
fi
if [ "$DATASET_SIZE" = "both" ]; then
  DOWNLOAD_MEDIUM=1
fi
if [ "$DOWNLOAD_SPECTROGRAMS" = "1" ] && [ "$PREPROCESS_FOR" != "none" ]; then
  case "$DATASET_SIZE" in
    medium|both) DOWNLOAD_MEDIUM=1 ;;
  esac
fi

CORE_DEPS=(
  numpy
  pandas
  scipy
  matplotlib
  seaborn
  scikit-learn
  xgboost
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
  ipython
)

NOTEBOOK_DEPS=(
  soundfile
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

detect_pytorch_build() {
  local cuda_version

  if [ "$PYTORCH_BUILD" != "auto" ]; then
    echo "$PYTORCH_BUILD"
    return
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "cpu"
    return
  fi

  cuda_version="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
  if [ -z "$cuda_version" ]; then
    echo "cpu"
    return
  fi

  "$ENV_DIR/bin/python" - "$cuda_version" <<'PY'
from decimal import Decimal
import sys

driver_cuda = Decimal(sys.argv[1])
if driver_cuda >= Decimal("13.0"):
    print("cu130")
elif driver_cuda >= Decimal("12.8"):
    print("cu128")
elif driver_cuda >= Decimal("12.6"):
    print("cu126")
else:
    print("unsupported")
PY
}

pytorch_build_cuda_version() {
  local build="$1"

  case "$build" in
    cu126) echo "12.6" ;;
    cu128) echo "12.8" ;;
    cu130) echo "13.0" ;;
    cpu) echo "None" ;;
    *)
      echo "Error: unsupported PYTORCH_BUILD '$build'. Use auto, cpu, cu126, cu128, or cu130." >&2
      exit 1
      ;;
  esac
}

pytorch_matches_target() {
  local build="$1"
  local quiet="${2:-0}"
  local target_cuda

  target_cuda="$(pytorch_build_cuda_version "$build")"

  "$ENV_DIR/bin/python" - "$PYTORCH_VERSION" "$TORCHAUDIO_VERSION" "$target_cuda" "$build" "$quiet" <<'PY'
import importlib.metadata as metadata
import sys

target_torch, target_torchaudio, target_cuda, target_build, quiet = sys.argv[1:6]

def fail(message):
    if quiet != "1":
        print(f"PyTorch verification failed: {message}", file=sys.stderr)
    raise SystemExit(1)

try:
    import torch
except Exception as exc:
    fail(f"could not import torch ({type(exc).__name__}: {exc})")

try:
    torch_version = metadata.version("torch")
    torchaudio_version = metadata.version("torchaudio")
except Exception as exc:
    fail(f"could not read torch/torchaudio package metadata ({type(exc).__name__}: {exc})")

def split_version(version):
    base, _, local = version.partition("+")
    return base, local.lower()

torch_base, torch_build = split_version(torch_version)
torchaudio_base, torchaudio_build = split_version(torchaudio_version)

if torch_base != target_torch:
    fail(f"expected torch {target_torch}, found {torch_version}")
if torchaudio_base != target_torchaudio:
    fail(f"expected torchaudio {target_torchaudio}, found {torchaudio_version}")

if target_build == "cpu":
    if torch.version.cuda is not None:
        fail(f"expected CPU-only torch, found CUDA runtime {torch.version.cuda}")
    if torch_build not in ("", "cpu"):
        fail(f"expected CPU-only torch wheel, found torch {torch_version}")
    if torchaudio_build not in ("", "cpu"):
        fail(f"expected CPU-only torchaudio wheel, found torchaudio {torchaudio_version}")
else:
    if torch_build != target_build:
        fail(f"expected torch wheel +{target_build}, found {torch_version}")
    if torchaudio_build != target_build:
        fail(f"expected torchaudio wheel +{target_build}, found {torchaudio_version}")
    if torch.version.cuda is None:
        fail(f"expected CUDA runtime {target_cuda}, but torch.version.cuda is None")

    actual_cuda = str(torch.version.cuda)
    actual_major_minor = ".".join(actual_cuda.split(".")[:2])
    if actual_major_minor != target_cuda:
        fail(f"expected CUDA runtime {target_cuda}.x, found {actual_cuda}")
PY
}

remove_stale_pytorch_cuda_deps() {
  local build="$1"
  local package
  local stale_packages=()

  while IFS= read -r package; do
    case "$build:$package" in
      cu12*:nvidia-*-cu13|\
      cu12*:nvidia-cublas|\
      cu12*:nvidia-cuda-cupti|\
      cu12*:nvidia-cuda-nvrtc|\
      cu12*:nvidia-cuda-runtime|\
      cu12*:nvidia-cufft|\
      cu12*:nvidia-cufile|\
      cu12*:nvidia-curand|\
      cu12*:nvidia-cusolver|\
      cu12*:nvidia-cusparse|\
      cu12*:nvidia-nvjitlink|\
      cu12*:nvidia-nvtx|\
      cu130:nvidia-*-cu12|\
      cpu:cuda-bindings|\
      cpu:cuda-pathfinder|\
      cpu:cuda-toolkit|\
      cpu:nvidia-*)
        stale_packages+=("$package")
        ;;
    esac
  done < <("$ENV_DIR/bin/python" -m pip list --format=freeze | sed 's/==.*//' || true)

  if [ "${#stale_packages[@]}" -gt 0 ]; then
    echo "Removing stale PyTorch CUDA packages: ${stale_packages[*]}"
    "$ENV_DIR/bin/python" -m pip uninstall -y "${stale_packages[@]}"
  fi
}

install_pytorch_deps() {
  local build
  local index_url

  build="$(detect_pytorch_build)"
  if [ "$build" = "unsupported" ]; then
    echo "Error: your NVIDIA driver is too old for the pinned PyTorch $PYTORCH_VERSION GPU wheels." >&2
    echo "Update the NVIDIA driver, or rerun setup for CPU-only PyTorch with: PYTORCH_BUILD=cpu bash setup.sh" >&2
    exit 1
  fi

  pytorch_build_cuda_version "$build" >/dev/null
  index_url="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/$build}"

  if pytorch_matches_target "$build" 1; then
    echo "PyTorch $PYTORCH_VERSION ($build) already installed. Skipping PyTorch install."
    return
  fi

  remove_stale_pytorch_cuda_deps "$build"

  echo "Installing PyTorch $PYTORCH_VERSION and torchaudio $TORCHAUDIO_VERSION ($build)..."
  "$ENV_DIR/bin/python" -m pip install --upgrade --force-reinstall \
    "torch==$PYTORCH_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION" \
    --index-url "$index_url"

  if ! pytorch_matches_target "$build"; then
    echo "Error: PyTorch installed, but verification failed for the requested '$build' target." >&2
    exit 1
  fi
}

download_file() {
  local url="$1"
  local output_path="$2"

  if [ -s "$output_path" ]; then
    echo "$(basename "$output_path") already exists. Skipping download."
    return
  fi

  if [ -e "$output_path" ]; then
    echo "$(basename "$output_path") exists but is empty. Re-downloading."
    rm -f "$output_path"
  fi

  echo "Downloading $(basename "$output_path")..."
  curl -L --fail --output "$output_path" "$url"
}

require_file_after_extract() {
  local file_path="$1"
  local label="$2"

  if [ ! -f "$file_path" ]; then
    echo "Error: expected $label at $file_path after extraction, but it was not found." >&2
    exit 1
  fi
}

ensure_fma_small() {
  if [ -f "$FMA_SMALL_SENTINEL" ]; then
    echo "FMA small audio already present. Skipping download and extraction."
    return
  fi

  require_command curl
  require_command unzip

  download_file "$FMA_SMALL_URL" "$FMA_SMALL_ZIP"

  echo "Extracting $(basename "$FMA_SMALL_ZIP")..."
  unzip -q "$FMA_SMALL_ZIP" -d "$DATASET_DIR"
  require_file_after_extract "$FMA_SMALL_SENTINEL" "FMA small audio"
  rm -f "$FMA_SMALL_ZIP"
}

ensure_fma_medium() {
  if [ "$DOWNLOAD_MEDIUM" != "1" ]; then
    echo "Skipping fma_medium download (set DOWNLOAD_MEDIUM=1, DATASET_SIZE=both, or request medium spectrograms to enable, ~22 GB)."
    return
  fi

  if [ -f "$FMA_MEDIUM_SENTINEL" ]; then
    echo "FMA medium audio already present. Skipping download and extraction."
    return
  fi

  require_command curl
  require_command unzip

  download_file "$FMA_MEDIUM_URL" "$FMA_MEDIUM_ZIP"

  echo "Extracting $(basename "$FMA_MEDIUM_ZIP")..."
  unzip -q "$FMA_MEDIUM_ZIP" -d "$DATASET_DIR"
  require_file_after_extract "$FMA_MEDIUM_SENTINEL" "FMA medium audio"
  rm -f "$FMA_MEDIUM_ZIP"
}

fma_metadata_present() {
  local sentinel

  for sentinel in "${FMA_METADATA_SENTINELS[@]}"; do
    if [ ! -f "$sentinel" ]; then
      return 1
    fi
  done

  return 0
}

require_fma_metadata_after_extract() {
  local sentinel

  for sentinel in "${FMA_METADATA_SENTINELS[@]}"; do
    require_file_after_extract "$sentinel" "FMA metadata"
  done
}

ensure_fma_metadata() {
  if fma_metadata_present; then
    echo "FMA metadata already present. Skipping download and extraction."
    return
  fi

  require_command curl
  require_command unzip

  download_file "$FMA_METADATA_URL" "$FMA_METADATA_ZIP"

  echo "Extracting $(basename "$FMA_METADATA_ZIP")..."
  unzip -q "$FMA_METADATA_ZIP" -d "$DATASET_DIR"
  require_fma_metadata_after_extract
  rm -f "$FMA_METADATA_ZIP"
}

ensure_python_env() {
  local python_bin
  local python_version

  if [ -x "$ENV_DIR/bin/python" ]; then
    echo "Environment already present at $ENV_DIR. Skipping venv creation."
  else
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
  fi

  echo "Installing/updating Python dependencies in $ENV_DIR..."
  "$ENV_DIR/bin/python" -m pip install --upgrade pip "setuptools<82" wheel
  "$ENV_DIR/bin/python" -m pip install "${CORE_DEPS[@]}" "${NOTEBOOK_DEPS[@]}"
  install_pytorch_deps
}

run_data_preprocessing() {
  local effective_preprocess_for

  if [ "$DOWNLOAD_SPECTROGRAMS" != "1" ]; then
    effective_preprocess_for="none"
    echo "Generating cleaned CSVs and feature CSVs without spectrograms."
    echo "Set DOWNLOAD_SPECTROGRAMS=1 to also generate CNN/CRNN spectrograms."
  elif [ "$PREPROCESS_FOR" = "none" ]; then
    effective_preprocess_for="none"
    echo "Generating cleaned CSVs and feature CSVs without spectrograms (PREPROCESS_FOR=none)..."
  else
    effective_preprocess_for="$PREPROCESS_FOR"
    echo "Generating spectrograms for DATASET_SIZE=$DATASET_SIZE, PREPROCESS_FOR=$PREPROCESS_FOR..."
  fi

  DATASET_DIR="$DATASET_DIR" \
  DATASET_SIZE="$DATASET_SIZE" \
  PREPROCESS_FOR="$effective_preprocess_for" \
    "$ENV_DIR/bin/python" "$PROJECT_ROOT/code/data_preprocessing.py"
}

ensure_fma_metadata
ensure_fma_small
ensure_fma_medium
ensure_python_env
run_data_preprocessing

echo "Setup complete."
echo "Activate the virtual environment with: source $ENV_DIR/bin/activate"
