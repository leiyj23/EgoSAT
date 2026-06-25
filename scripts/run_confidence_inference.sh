#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_confidence_inference.sh MODEL --task TASK --gt-root GT_ROOT --ego4d-root VIDEO_ROOT [options]

Generate cand_conf raw predictions for the main-table confidence columns.

Supported tasks:
  sh_pred | ms_pred | ms_rtrv

Supported models:
  qwen2_5_vl_7b | qwen2_5_vl_32b

Options:
  --task TASK
  --gt-root DIR
  --mcq-root DIR
  --ego4d-root DIR
  --runs-root DIR
  --output FILE        Optional normalized JSONL; official scorer reads raw JSON.
  --split SPLIT
  --model-id ID_OR_PATH
  --num-frames N
  --torch-dtype DTYPE
  --gpu CUDA_VISIBLE_DEVICES
  --help
USAGE
}

fail() {
  echo "[ERROR] $*" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ "${1:-}" == "--help" || $# -eq 0 ]]; then
  usage
  exit 0
fi

MODEL="$1"
shift

TASK=""
GT_ROOT=""
MCQ_ROOT=""
EGO4D_ROOT=""
RUNS_ROOT="$PROJECT_ROOT/outputs/runs"
OUTPUT=""
SPLIT="val"
MODEL_ID=""
NUM_FRAMES=""
TORCH_DTYPE=""
GPU=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="${2:-}"; shift 2 ;;
    --gt-root) GT_ROOT="${2:-}"; shift 2 ;;
    --mcq-root) MCQ_ROOT="${2:-}"; shift 2 ;;
    --ego4d-root) EGO4D_ROOT="${2:-}"; shift 2 ;;
    --runs-root) RUNS_ROOT="${2:-}"; shift 2 ;;
    --output) OUTPUT="${2:-}"; shift 2 ;;
    --split) SPLIT="${2:-}"; shift 2 ;;
    --model-id) MODEL_ID="${2:-}"; shift 2 ;;
    --num-frames) NUM_FRAMES="${2:-}"; shift 2 ;;
    --torch-dtype) TORCH_DTYPE="${2:-}"; shift 2 ;;
    --gpu) GPU="${2:-}"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done

[[ -n "$TASK" ]] || fail "Missing --task"
[[ -n "$GT_ROOT" ]] || fail "Missing --gt-root"
[[ -n "$EGO4D_ROOT" ]] || fail "Missing --ego4d-root"
[[ -d "$GT_ROOT" ]] || fail "--gt-root does not exist: $GT_ROOT"
[[ -d "$EGO4D_ROOT" ]] || fail "--ego4d-root does not exist: $EGO4D_ROOT"

case "$TASK" in
  sh_pred|ms_pred|ms_rtrv)
    MCQ_ROOT="${MCQ_ROOT:-$PROJECT_ROOT/data/mcq_shuffled/$TASK}"
    ;;
  *)
    fail "Unsupported confidence task: $TASK (expected sh_pred, ms_pred, or ms_rtrv)"
    ;;
esac
[[ -d "$MCQ_ROOT" ]] || fail "--mcq-root does not exist: $MCQ_ROOT"

ADAPTER_PY=""
case "$MODEL" in
  qwen2_5_vl_7b)
    ADAPTER_PY="$PROJECT_ROOT/benchmark_val/llms/qwen2_5_vl_7b_adapter.py"
    MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
    NUM_FRAMES="${NUM_FRAMES:-16}"
    TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
    ;;
  qwen2_5_vl_32b)
    ADAPTER_PY="$PROJECT_ROOT/benchmark_val/llms/qwen2_5_vl_32b_adapter.py"
    MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-32B-Instruct}"
    NUM_FRAMES="${NUM_FRAMES:-16}"
    TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
    ;;
  gemini_pro)
    fail "gemini_pro is not supported for cand_conf confidence because the public adapter does not expose the required p_cond fields"
    ;;
  *)
    fail "Unsupported model for confidence inference: $MODEL"
    ;;
esac

if [[ -n "$GPU" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

mkdir -p "$RUNS_ROOT"

export TASK
export MODEL_NAME="$MODEL"
export SPLIT
export GT_ROOT
export VIDEO_ROOT="$EGO4D_ROOT"
export RUNS_ROOT
export HELPER_PY="$PROJECT_ROOT/benchmark_val/helper.py"
export ADAPTER_PY
export PRED_FLAVOR="cand_conf"
export MCP_ROOT="$MCQ_ROOT"
export MODEL_ID NUM_FRAMES TORCH_DTYPE
export DEVICE_MAP="${DEVICE_MAP:-auto}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
export RETURN_LOGPROBS="1"
export STOP_ON_CLOSE_TAGS="${STOP_ON_CLOSE_TAGS:-1}"
export CAND_CONF_MAX_NEW_TOKENS="${CAND_CONF_MAX_NEW_TOKENS:-4}"

if [[ "$TASK" == "sh_pred" ]]; then
  export SH_PRED_ALL_STATES=1
fi

python "$PROJECT_ROOT/benchmark_val/runner.py" --mode cand

PRED_DIR="$RUNS_ROOT/$MODEL/$TASK/cand_conf"
echo "[OK] cand_conf raw predictions: $PRED_DIR"

if [[ -n "$OUTPUT" ]]; then
  python "$PROJECT_ROOT/scripts/normalize_predictions.py" \
    --input-dir "$PRED_DIR" \
    --output "$OUTPUT" \
    --model "$MODEL" \
    --task "$TASK" \
    --task-type mcq \
    --flavor cand_conf
fi

