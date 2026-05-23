#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$PROJECT_ROOT/.venv"
PYTHON_BIN="${PYTHON_BIN:-}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.11.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-$PYTORCH_VERSION}"
PYTORCH_BUILD="${PYTORCH_BUILD:-auto}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-}"
PYTHON_BUILD_VERSION="${PYTHON_BUILD_VERSION:-3.12.10}"
SQLITE_BUILD_VERSION="${SQLITE_BUILD_VERSION:-3460100}"
SQLITE_BUILD_YEAR="${SQLITE_BUILD_YEAR:-2024}"
LOCAL_PYTHON_DIR="$PROJECT_ROOT/.python"

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

if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(read_env_file_value PYTHON_BIN || true)"
fi
if [ -n "$PYTHON_BIN" ]; then
  if [[ "$PYTHON_BIN" == "~/"* ]]; then
    PYTHON_BIN="$HOME/${PYTHON_BIN:2}"
  elif [[ "$PYTHON_BIN" == '$HOME/'* ]]; then
    PYTHON_BIN="$HOME/${PYTHON_BIN:6}"
  elif [[ "$PYTHON_BIN" == '${HOME}/'* ]]; then
    PYTHON_BIN="$HOME/${PYTHON_BIN:8}"
  fi
fi

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

if [ -z "${DOWNLOAD_SPECTROGRAMS:-}" ]; then
  DOWNLOAD_SPECTROGRAMS="$(read_env_file_value DOWNLOAD_SPECTROGRAMS || true)"
fi
if is_truthy "${DOWNLOAD_SPECTROGRAMS:-0}"; then
  DOWNLOAD_SPECTROGRAMS=1
else
  DOWNLOAD_SPECTROGRAMS=0
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
  tensorflow
  optuna
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

python_meets_min_version() {
  local python_bin="$1"

  "$python_bin" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
}

python_has_required_stdlib() {
  local python_bin="$1"

  "$python_bin" - <<'PY' >/dev/null 2>&1
import ctypes
import sqlite3
PY
}

python_has_venv() {
  local python_bin="$1"
  "$python_bin" - <<'PY' >/dev/null 2>&1
import venv
import ensurepip
PY
}

sqlite3_headers_available() {
  if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists sqlite3 2>/dev/null; then
    return 0
  fi
  local dir
  for dir in /usr/include /usr/local/include "$LOCAL_PYTHON_DIR/include"; do
    [ -f "$dir/sqlite3.h" ] && return 0
  done
  return 1
}

build_sqlite_from_source() {
  local prefix="$LOCAL_PYTHON_DIR"
  local version="$SQLITE_BUILD_VERSION"
  local year="$SQLITE_BUILD_YEAR"
  local tarball="/tmp/sqlite-autoconf-${version}.tar.gz"
  local src_dir="/tmp/sqlite-autoconf-${version}"
  local url="https://www.sqlite.org/${year}/sqlite-autoconf-${version}.tar.gz"

  if [ -f "$prefix/include/sqlite3.h" ]; then
    echo "SQLite already built at $prefix. Skipping." >&2
    return
  fi

  echo "SQLite3 development headers not found. Building SQLite from source (needed for Python's sqlite3 module)..."

  if [ ! -f "$tarball" ]; then
    echo "Downloading SQLite $version source..."
    curl -L --fail --output "$tarball" "$url"
  fi

  rm -rf "$src_dir"
  echo "Extracting SQLite source..."
  tar -xf "$tarball" -C /tmp

  echo "Building SQLite $version..."
  (cd "$src_dir" && CFLAGS="-fPIC" ./configure --prefix="$prefix" \
    --enable-static --disable-shared && make -j"$(nproc)" && make install)

  rm -rf "$src_dir" "$tarball"
  echo "SQLite built and installed at $prefix." >&2
}

build_python_from_source() {
  local version="$PYTHON_BUILD_VERSION"
  local prefix="$LOCAL_PYTHON_DIR"
  local tarball src_dir url installed_bin

  echo "No suitable Python >= 3.10 found. Building Python $version from source..."
  echo "This may take several minutes."

  for tool in gcc make curl tar; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      echo "Error: '$tool' is required to build Python from source but was not found on PATH." >&2
      echo "Install build tools and rerun: sudo apt install build-essential curl" >&2
      exit 1
    fi
  done

  tarball="/tmp/Python-${version}.tar.xz"
  src_dir="/tmp/Python-${version}"
  url="https://www.python.org/ftp/python/${version}/Python-${version}.tar.xz"

  if [ ! -f "$tarball" ]; then
    echo "Downloading Python $version source from python.org..."
    curl -L --fail --output "$tarball" "$url"
  fi

  rm -rf "$src_dir"
  echo "Extracting Python source..."
  tar -xf "$tarball" -C /tmp

  # Ensure SQLite headers are available so Python's sqlite3 module gets compiled in.
  local sqlite_cppflags="" sqlite_ldflags=""
  if ! sqlite3_headers_available; then
    build_sqlite_from_source
    sqlite_cppflags="-I$LOCAL_PYTHON_DIR/include"
    sqlite_ldflags="-L$LOCAL_PYTHON_DIR/lib"
  fi

  echo "Configuring Python $version (installing to $prefix)..."
  (cd "$src_dir" && \
    CPPFLAGS="${sqlite_cppflags:+$sqlite_cppflags }${CPPFLAGS:-}" \
    LDFLAGS="${sqlite_ldflags:+$sqlite_ldflags }${LDFLAGS:-}" \
    ./configure --prefix="$prefix" --with-ensurepip=install)

  echo "Building Python $version..."
  (cd "$src_dir" && make -j"$(nproc)")

  echo "Installing Python $version..."
  (cd "$src_dir" && make install)

  rm -rf "$src_dir" "$tarball"

  installed_bin="$prefix/bin/python3"
  if [ ! -x "$installed_bin" ]; then
    installed_bin="$(find "$prefix/bin" -name 'python3*' -executable 2>/dev/null | sort -V | tail -n1 || true)"
  fi

  if [ -z "$installed_bin" ] || [ ! -x "$installed_bin" ]; then
    echo "Error: Python build completed but no executable was found in $prefix/bin." >&2
    exit 1
  fi

  echo "Python $version built and installed at $installed_bin." >&2
  echo "$installed_bin"
}

find_python() {
  local candidate local_python

  if [ -n "$PYTHON_BIN" ]; then
    require_command "$PYTHON_BIN"
    echo "$PYTHON_BIN"
    return
  fi

  # Reuse a previously self-built Python
  local_python="$(find "$LOCAL_PYTHON_DIR/bin" -name 'python3*' -executable 2>/dev/null | sort -V | tail -n1 || true)"
  if [ -n "$local_python" ] && python_meets_min_version "$local_python" && python_has_required_stdlib "$local_python" && python_has_venv "$local_python"; then
    echo "$local_python"
    return
  fi

  for candidate in python python3.11 python3.12 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && python_meets_min_version "$candidate" && python_has_required_stdlib "$candidate" && python_has_venv "$candidate"; then
      echo "$candidate"
      return
    fi
  done

  build_python_from_source
}

check_python_version() {
  local python_bin="$1"

  if ! python_meets_min_version "$python_bin"; then
    echo "Error: Python 3.10 or newer is required." >&2
    echo "Set PYTHON_BIN to a compatible interpreter, for example: PYTHON_BIN=\$HOME/opt/python-3.11.9/bin/python3.11 bash setup.sh" >&2
    exit 1
  fi
}

check_python_required_stdlib() {
  local python_bin="$1"
  local output

  if ! output="$("$python_bin" - <<'PY' 2>&1
import ctypes
import sqlite3
PY
  )"; then
    echo "Error: Python at $python_bin is missing required standard-library modules (ctypes or sqlite3)." >&2
    echo "$output" >&2
    echo "This usually means Python was built without libffi or SQLite development headers." >&2
    echo "Install the missing headers, rebuild/reinstall Python, remove $ENV_DIR, and rerun setup.sh." >&2
    echo "On Debian/Ubuntu: sudo apt install libffi-dev libsqlite3-dev" >&2
    echo "On RHEL/Fedora:   sudo dnf install libffi-devel sqlite-devel" >&2
    echo "If you use pyenv, reinstall the selected Python after installing the headers, for example: pyenv install --force \$(pyenv version-name)" >&2
    echo "Or use a working system Python explicitly: PYTHON_BIN=/usr/bin/python3 bash setup.sh" >&2
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
    check_python_version "$ENV_DIR/bin/python"
    check_python_required_stdlib "$ENV_DIR/bin/python"
  else
    if [ -d "$ENV_DIR" ]; then
      echo "Error: $ENV_DIR already exists but does not look like a Python virtual environment." >&2
      echo "Remove it and rerun setup.sh, or set ENV_DIR in the script to another path." >&2
      exit 1
    fi

    python_bin="$(find_python)"
    check_python_version "$python_bin"
    check_python_required_stdlib "$python_bin"
    python_version="$("$python_bin" --version 2>&1)"

    echo "Creating Python virtual environment at $ENV_DIR using $python_version..."
    if ! "$python_bin" -m venv "$ENV_DIR"; then
      echo "Error: failed to create the virtual environment." >&2
      echo "The Python at $python_bin does not have venv support." >&2
      echo "On Debian/Ubuntu: sudo apt install python3-venv  (or python3.XX-venv for a specific version)" >&2
      echo "Or let setup build its own Python by unsetting PYTHON_BIN." >&2
      exit 1
    fi
  fi

  echo "Installing/updating Python dependencies in $ENV_DIR..."
  "$ENV_DIR/bin/python" -m pip install --upgrade pip "setuptools<82" wheel
  "$ENV_DIR/bin/python" -m pip install "${CORE_DEPS[@]}" "${NOTEBOOK_DEPS[@]}"
  install_pytorch_deps
}

ensure_python_env
ensure_fma_metadata

if [ "$DOWNLOAD_SPECTROGRAMS" = "1" ]; then
  case "$DATASET_SIZE" in
    small|both) ensure_fma_small ;;
  esac
  case "$DATASET_SIZE" in
    medium|both) ensure_fma_medium ;;
  esac
else
  echo "Skipping audio download (set DOWNLOAD_SPECTROGRAMS=1 in .env to download fma_small/ or fma_medium/)."
fi

echo "Setup complete."
echo "Activate the virtual environment with: source $ENV_DIR/bin/activate"
