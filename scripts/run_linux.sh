#!/usr/bin/env bash
# Bounded smoke training, resume, validation promotion, tests and official validation.
set -euo pipefail
DEVICE="${1:-cpu}"
if [[ "$DEVICE" != cpu && "$DEVICE" != cuda ]]; then exit 2; fi
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$REPO_DIR"
WORK_DIR="${DOOM_NEXUS_WORK_DIR:-$(cd -- "$REPO_DIR/../.." && pwd)/work/linux}"
PYTHON="${DOOM_NEXUS_PYTHON:-$WORK_DIR/venv/bin/python}"
export PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XDG_CACHE_HOME="$WORK_DIR/cache"
RESULT_DIR="benchmarks/continuation/linux-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p -- "$RESULT_DIR"
run_step() {
  local name="$1"; shift
  if "$@" > >(tee "$RESULT_DIR/$name.txt") 2>&1; then
    printf '%s 0\n' "$name" >> "$RESULT_DIR/exit-codes.txt"
  else
    local status=$?
    printf '%s %s\n' "$name" "$status" >> "$RESULT_DIR/exit-codes.txt"
    printf 'Stopped at %s. Preserve %s and inspect its log.\n' "$name" "$RESULT_DIR" >&2
    exit "$status"
  fi
}
CUDA_ARGS=()
if [[ "$DEVICE" == cuda ]]; then CUDA_ARGS+=(--require-cuda); fi
run_step preflight "$PYTHON" -m scripts.preflight --engine "${CUDA_ARGS[@]}" --out "$RESULT_DIR/preflight.json"
run_step throughput "$PYTHON" -m scripts.benchmark --throughput-only --max-envs 1
RUN_DIR="$WORK_DIR/runs/$(basename -- "$RESULT_DIR")"
# Small lifetime targets establish real checkpoints first. They are not a competitive training budget.
run_step mlp-train "$PYTHON" -m training.train --model mlp --stage movement --device "$DEVICE" \
  --envs 1 --steps 4096 --seconds 300 --rollout 64 --sequence 16 --epochs 2 --out "$RUN_DIR/mlp"
run_step mlp-resume "$PYTHON" -m training.train --model mlp --stage movement --device "$DEVICE" \
  --envs 1 --steps 8192 --seconds 300 --rollout 64 --sequence 16 --epochs 2 \
  --resume "$RUN_DIR/mlp/latest.pt" --out "$RUN_DIR/mlp"
run_step gru-train "$PYTHON" -m training.train --model gru --stage auto --device "$DEVICE" \
  --envs 1 --steps 8192 --seconds 600 --rollout 64 --sequence 16 --epochs 2 --out "$RUN_DIR/gru"
run_step select "$PYTHON" -m training.evaluate --checkpoint "$RUN_DIR/mlp/latest.pt" \
  --checkpoint "$RUN_DIR/gru/latest.pt" --split validation --seeds 0 1 --ticks 2100 \
  --device "$DEVICE" --promote --out "$RESULT_DIR/validation.json"
run_step heldout "$PYTHON" -m training.evaluate --checkpoint submission/weights/best_model.pt \
  --split test --seeds 0 1 --ticks 2100 --device "$DEVICE" --out "$RESULT_DIR/heldout.json"
run_step tests "$PYTHON" -m pytest -q -ra --junitxml="$RESULT_DIR/tests.xml"
run_step performance "$PYTHON" -m scripts.benchmark --device "$DEVICE" --steps 512
run_step git-lfs git lfs install --local
run_step stage-weights git add -- .gitattributes submission/weights/best_model.pt
run_step source-audit "$PYTHON" -m scripts.check_submission
run_step validator "$PYTHON" validate.py --device "$DEVICE"
printf 'Local baseline checks completed. No COMPETITION READY claim is implied. Evidence: %s\n' "$RESULT_DIR"
