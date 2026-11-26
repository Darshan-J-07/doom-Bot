#!/usr/bin/env bash
# Run from the user's normal Ubuntu terminal. Copy the existing checkout; reuse verified Python.
set -euo pipefail
if [[ "$(uname -s)" != Linux ]]; then
  printf '%s\n' 'Run this script inside the working Ubuntu terminal.' >&2
  exit 2
fi
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${DOOM_NEXUS_PYTHON:-$HOME/doom-test/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then printf 'Verified Python was not found: %s\n' "$PYTHON" >&2; exit 2; fi
"$PYTHON" -c 'import sys, torch, vizdoom; assert sys.version_info[:2]==(3,13); assert vizdoom.__version__=="1.3.3"; assert torch.cuda.is_available(); print(sys.version); print(torch.__version__); print(torch.cuda.get_device_name(0))'
DEST_DIR="${DOOM_NEXUS_DEST:-$HOME/doom-nexus-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
# mkdir without -p refuses to overwrite an existing project.
mkdir -- "$DEST_DIR"
cp -a -- "$SOURCE_DIR/." "$DEST_DIR/"
cd -- "$DEST_DIR"
printf 'Preserved original: %s\nWorking Linux copy: %s\n' "$SOURCE_DIR" "$DEST_DIR"
printf '%s\n' "$DEST_DIR" > "$SOURCE_DIR/../DOOM_NEXUS_Linux_Working_Copy.txt"
export_results() {
  local status=$?
  set +e
  mkdir -p -- "$DEST_DIR/benchmarks"
  printf '%s\n' "$status" > "$DEST_DIR/benchmarks/verified-launch-exit-code.txt"
  local paths=(benchmarks submission/weights)
  if [[ -d "$DEST_DIR/runs" ]]; then paths+=(runs); fi
  local archive="$SOURCE_DIR/../$(basename -- "$DEST_DIR")-results.tar.gz"
  tar -czf "$archive" -C "$DEST_DIR" "${paths[@]}"
  printf 'Actual result archive: %s\nLauncher exit code: %s\n' "$archive" "$status"
  exit "$status"
}
trap export_results EXIT
export PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then "$PYTHON" -m ensurepip --upgrade; fi
# Torch is deliberately absent from this install: retain the verified 2.11.0+cu128 build.
"$PYTHON" -m pip install --upgrade-strategy only-if-needed \
  'gymnasium>=1.0' 'numpy>=2.0' 'pytest>=8' 'psutil>=7,<8' -e env
"$PYTHON" -m pip check
"$PYTHON" -m scripts.continue_verified --seconds "${DOOM_NEXUS_TRAIN_SECONDS:-1800}"
