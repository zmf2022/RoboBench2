#!/usr/bin/env bash
# Compare pi05 and pi05 + API VLM on all benchmark tasks.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
PI05_NUM_ENVS="${PI05_NUM_ENVS:-5}"
NUM_RUNS="${NUM_RUNS:-5}" # Total episodes per task, equal for both policies.
if ! [[ "$NUM_RUNS" =~ ^[1-9][0-9]*$ && "$PI05_NUM_ENVS" =~ ^[1-9][0-9]*$ ]]; then
  echo 'NUM_RUNS and PI05_NUM_ENVS must be positive integers' >&2
  exit 2
fi
if (( NUM_RUNS % PI05_NUM_ENVS != 0 )); then
  echo 'NUM_RUNS must be divisible by PI05_NUM_ENVS so both policies evaluate the same episode count' >&2
  exit 2
fi

common=(
  --task-dirs benchmark --instruction-type default
  --headless --seed "${SEED:-0}" --device "${DEVICE:-cuda:0}"
  --remote-host "${REMOTE_HOST:-localhost}" --remote-port "${REMOTE_PORT:-8000}"
  --video-mode "${VIDEO_MODE:-sensor}" --renderer realtime
)

echo "Evaluating pi05: $PI05_NUM_ENVS environments, $NUM_RUNS episodes per task"
"$PYTHON_BIN" policies/pi0_family/run.py \
  --policy pi05 --open-loop-horizon 15 \
  --num-envs "$PI05_NUM_ENVS" --num-runs "$((NUM_RUNS / PI05_NUM_ENVS))" "${common[@]}"

echo "Evaluating pi05 + API VLM: 1 environment, $NUM_RUNS episodes per task"
"$PYTHON_BIN" policies/ai_native/orchestrated/run.py \
  --review-backend api --allow-model-requests \
  --num-envs 1 --num-runs "$NUM_RUNS" \
  --reasoning-effort "${REASONING_EFFORT:-medium}" "${common[@]}"
