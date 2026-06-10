#!/usr/bin/env bash
set -euo pipefail

# setup.sh — Recreate development environment without pixi
# - Creates a local install prefix for LibRaw in .pixi/libraw-install
# - Builds LibRaw from external/LibRaw-master if present
# - Creates a Python virtualenv in .venv and installs extras
# - Sets LIBRAW_LOCAL_PREFIX for local builds

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIXIPREFIX="$ROOT_DIR/.pixi/libraw-install"
VENV_DIR="$ROOT_DIR/.venv"

echo "Root: $ROOT_DIR"

function ensure_system_deps() {
  echo "Checking system deps: cmake, make, pkg-config"
  for cmd in cmake make pkg-config; do
    if ! command -v $cmd >/dev/null 2>&1; then
      echo "Warning: $cmd not found. Please install via Homebrew: brew install $cmd"
    fi
  done
  # macOS OpenMP (libomp) hint
  if [[ "$(uname)" == "Darwin" ]]; then
    if ! brew --prefix libomp >/dev/null 2>&1; then
      echo "Note: OpenMP (libomp) not found via Homebrew. Consider: brew install libomp"
    fi
  fi
}

function build_libraw() {
  if [ ! -d "$ROOT_DIR/external/LibRaw-master" ]; then
    echo "external/LibRaw-master not found. Skipping LibRaw build."
    return 0
  fi

  echo "Building LibRaw into $PIXIPREFIX"
  mkdir -p "$PIXIPREFIX"
  pushd "$ROOT_DIR/external/LibRaw-master" >/dev/null
  ./configure --prefix="$PIXIPREFIX"
  make -j$(sysctl -n hw.ncpu)
  make install
  popd >/dev/null
}

function create_venv_and_install() {
  echo "Creating Python venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip setuptools wheel
  echo "Installing package in editable mode with dev extras"
  # Ensure environment variable used by setup.py points to our local libraw
  export LIBRAW_LOCAL_PREFIX="$PIXIPREFIX"
  pip install -e ".[dev]"
  deactivate
}

function finalize() {
  cat <<EOF
Setup complete.
- Virtualenv: $VENV_DIR
- Local LibRaw prefix: $PIXIPREFIX

To use the environment:
  source $VENV_DIR/bin/activate
  export LIBRAW_LOCAL_PREFIX="$PIXIPREFIX"
  python -m pytest -v
EOF
}

ensure_system_deps
build_libraw
create_venv_and_install
finalize
