#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DDNM_ROOT="${DDNM_ROOT:-}"
GPU="${GPU:-0}"
CASE_ID="${CASE_ID:-mela_0050}"
SETUP="ddnm_orig"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-}"
GT_PICKLE="${GT_PICKLE:-}"
INPUT_NPY="${INPUT_NPY:-$REPO_ROOT/examples/mela_0050/mela_0050_projection_4x_128x128.npy}"

MODEL_ARGS=()
if [[ -n "$MODEL_CHECKPOINT" ]]; then
  MODEL_ARGS+=(--model-checkpoint "$MODEL_CHECKPOINT")
fi

if [[ -z "$DDNM_ROOT" ]]; then
  echo "Set DDNM_ROOT=/path/to/DDNM before running this script." >&2
  exit 2
fi

if [[ -z "$GT_PICKLE" ]]; then
  echo "Set GT_PICKLE=/path/to/MELA_GT_512_rmbed/mela_0050_rmbed.pickle before running this script." >&2
  exit 2
fi

python "$REPO_ROOT/ddnm_inference/run_ddnm_projection_sr.py" \
  --ddnm-root "$DDNM_ROOT" \
  --input-npy "$INPUT_NPY" \
  --gt-pickle "$GT_PICKLE" \
  --case-id "$CASE_ID" \
  --scale 4 \
  --setup "$SETUP" \
  --gpu "$GPU" \
  --eta 0.990 \
  --sigma-y 0.001 \
  --clip-max 1.05 \
  --output-name "${CASE_ID}_ddnm_x4_${SETUP}" \
  "${MODEL_ARGS[@]}" \
  "$@"
