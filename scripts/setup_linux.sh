#!/usr/bin/env bash
# Run manually inside Ubuntu/WSL 2. This script never installs or changes GPU drivers.
set -euo pipefail
MODE="${1:-cpu}"
if [[ "$MODE" != cpu && "$MODE" != cuda ]]; then
  printf '%s\n' 'Usage: bash scripts/setup_linux.sh cpu|cuda' >&2
  exit 2
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  printf '%s\n' 'This setup targets x86_64 Linux/WSL 2.' >&2
  exit 2
fi
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$REPO_DIR"
WORK_DIR="${DOOM_NEXUS_WORK_DIR:-$(cd -- "$REPO_DIR/../.." && pwd)/work/linux}"
mkdir -p -- "$WORK_DIR/tools" "$WORK_DIR/cache"
export UV_CACHE_DIR="$WORK_DIR/cache/uv"
export UV_PYTHON_INSTALL_DIR="$WORK_DIR/python"
export XDG_CACHE_HOME="$WORK_DIR/cache"
UV="$WORK_DIR/tools/uv"
if [[ ! -x "$UV" ]]; then
  curl --fail --location --retry 3 --output "$WORK_DIR/uv.tar.gz" \
    'https://github.com/astral-sh/uv/releases/download/0.9.18/uv-x86_64-unknown-linux-gnu.tar.gz'
  tar -xzf "$WORK_DIR/uv.tar.gz" -C "$WORK_DIR/tools" --strip-components=1
fi
"$UV" python install 3.13.9
if [[ ! -x "$WORK_DIR/venv/bin/python" ]]; then
  "$UV" venv --python 3.13.9 "$WORK_DIR/venv"
fi
PYTHON="$WORK_DIR/venv/bin/python"
INDEX='https://download.pytorch.org/whl/cpu'
if [[ "$MODE" == cuda ]]; then INDEX='https://download.pytorch.org/whl/cu128'; fi
"$UV" pip install --python "$PYTHON" --reinstall --index-url "$INDEX" 'torch==2.9.1'
"$UV" pip install --python "$PYTHON" -r requirements-training.txt -e env
"$UV" pip check --python "$PYTHON"
export PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ARGS=()
if [[ "$MODE" == cuda ]]; then ARGS+=(--require-cuda); fi
"$PYTHON" -m scripts.preflight "${ARGS[@]}" --out benchmarks/continuation/linux-setup.json
printf 'Environment ready. Run: bash scripts/run_linux.sh %s\n' "$MODE"
