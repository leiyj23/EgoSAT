#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_mcq_inference.sh MODEL --task TASK --gt-root GT_ROOT --ego4d-root VIDEO_ROOT --output OUTPUT [options]

Models:
  qwen2_5_vl_7b | qwen2_5_vl_32b | gemini_pro

Options:
  --task TASK
  --gt-root DIR
  --mcq-root DIR
  --ego4d-root DIR
  --output FILE
  --runs-root DIR
  --split SPLIT
  --ss-pos POS
  --model-id ID_OR_PATH
  --num-frames N
  --torch-dtype DTYPE
  --gpu CUDA_VISIBLE_DEVICES
  --openrouter-model MODEL_ID
  --help
USAGE
}

fail() {
  echo "[ERROR] $*" >&2
  exit 2
}

sanitize_path() {
  local value="${1:-}"
  value="${value//[^a-zA-Z0-9._-]/_}"
  while [[ "$value" == *"__"* ]]; do
    value="${value//__/_}"
  done
  value="${value##_}"
  value="${value%%_}"
  printf '%s' "${value:-empty}"
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
OUTPUT=""
RUNS_ROOT="$PROJECT_ROOT/outputs/runs"
SPLIT="val"
SS_POS=""
MODEL_ID=""
NUM_FRAMES=""
TORCH_DTYPE=""
GPU=""
OPENROUTER_MODEL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="${2:-}"; shift 2 ;;
    --gt-root) GT_ROOT="${2:-}"; shift 2 ;;
    --mcq-root) MCQ_ROOT="${2:-}"; shift 2 ;;
    --ego4d-root) EGO4D_ROOT="${2:-}"; shift 2 ;;
    --output) OUTPUT="${2:-}"; shift 2 ;;
    --runs-root) RUNS_ROOT="${2:-}"; shift 2 ;;
    --split) SPLIT="${2:-}"; shift 2 ;;
    --ss-pos) SS_POS="${2:-}"; shift 2 ;;
    --model-id) MODEL_ID="${2:-}"; shift 2 ;;
    --num-frames) NUM_FRAMES="${2:-}"; shift 2 ;;
    --torch-dtype) TORCH_DTYPE="${2:-}"; shift 2 ;;
    --gpu) GPU="${2:-}"; shift 2 ;;
    --openrouter-model) OPENROUTER_MODEL="${2:-}"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done

[[ -n "$TASK" ]] || fail "Missing --task"
[[ -n "$GT_ROOT" ]] || fail "Missing --gt-root"
[[ -n "$EGO4D_ROOT" ]] || fail "Missing --ego4d-root"
[[ -n "$OUTPUT" ]] || fail "Missing --output"
[[ -d "$GT_ROOT" ]] || fail "--gt-root does not exist: $GT_ROOT"
[[ -d "$EGO4D_ROOT" ]] || fail "--ego4d-root does not exist: $EGO4D_ROOT"

RUNNER_MODE=""
PRED_FLAVOR_VALUE=""
NORMALIZE_FLAVOR=""
DEFAULT_MCQ_SUBDIR=""
case "$TASK" in
  now_narration)
    RUNNER_MODE="cand"
    PRED_FLAVOR_VALUE="cand"
    NORMALIZE_FLAVOR="cand_mcq"
    DEFAULT_MCQ_SUBDIR="now_narration_action"
    ;;
  now_state_switch)
    RUNNER_MODE="cand_state"
    PRED_FLAVOR_VALUE="cand_state"
    NORMALIZE_FLAVOR="cand_state"
    SS_POS="${SS_POS:-t:2.0}"
    ;;
  sh_rtrv)
    RUNNER_MODE="cand"
    PRED_FLAVOR_VALUE="cand"
    NORMALIZE_FLAVOR="cand"
    DEFAULT_MCQ_SUBDIR="sh_rtrv"
    ;;
  ms_rtrv)
    RUNNER_MODE="cand"
    PRED_FLAVOR_VALUE="cand"
    NORMALIZE_FLAVOR="cand"
    DEFAULT_MCQ_SUBDIR="ms_rtrv"
    ;;
  ms_pred)
    RUNNER_MODE="cand"
    PRED_FLAVOR_VALUE="cand"
    NORMALIZE_FLAVOR="cand"
    DEFAULT_MCQ_SUBDIR="ms_pred"
    ;;
  sh_pred)
    RUNNER_MODE="cand"
    PRED_FLAVOR_VALUE="cand_full"
    NORMALIZE_FLAVOR="cand_full"
    DEFAULT_MCQ_SUBDIR="sh_pred"
    export SH_PRED_ALL_STATES=1
    ;;
  *)
    fail "Unsupported MCQ task: $TASK"
    ;;
esac

if [[ -n "$DEFAULT_MCQ_SUBDIR" ]]; then
  MCQ_ROOT="${MCQ_ROOT:-$PROJECT_ROOT/data/mcq_shuffled/$DEFAULT_MCQ_SUBDIR}"
  [[ -d "$MCQ_ROOT" ]] || fail "--mcq-root does not exist: $MCQ_ROOT"
fi

ADAPTER_PY=""
case "$MODEL" in
  qwen2_5_vl_7b)
    ADAPTER_PY="$PROJECT_ROOT/benchmark_val/llms/qwen2_5_vl_7b_adapter.py"
    MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
    NUM_FRAMES="${NUM_FRAMES:-16}"
    TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
    export MODEL_ID NUM_FRAMES TORCH_DTYPE
    export DEVICE_MAP="${DEVICE_MAP:-auto}"
    export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
    export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
    export RETURN_LOGPROBS="${RETURN_LOGPROBS:-1}"
    export STOP_ON_CLOSE_TAGS="${STOP_ON_CLOSE_TAGS:-1}"
    ;;
  qwen2_5_vl_32b)
    ADAPTER_PY="$PROJECT_ROOT/benchmark_val/llms/qwen2_5_vl_32b_adapter.py"
    MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-32B-Instruct}"
    NUM_FRAMES="${NUM_FRAMES:-16}"
    TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
    export MODEL_ID NUM_FRAMES TORCH_DTYPE
    export DEVICE_MAP="${DEVICE_MAP:-auto}"
    export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
    export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
    export RETURN_LOGPROBS="${RETURN_LOGPROBS:-1}"
    export STOP_ON_CLOSE_TAGS="${STOP_ON_CLOSE_TAGS:-1}"
    ;;
  gemini_pro)
    ADAPTER_PY="$PROJECT_ROOT/benchmark_val/llms/llm_adapter.py"
    [[ -n "${OPENROUTER_API_KEY:-}" ]] || fail "gemini_pro requires OPENROUTER_API_KEY in the environment"
    export OPENROUTER_MODEL="${OPENROUTER_MODEL:-google/gemini-2.5-pro}"
    export OPENROUTER_URL="${OPENROUTER_URL:-https://openrouter.ai/api/v1/chat/completions}"
    export MAX_TOKENS="${MAX_TOKENS:-2048}"
    export TEMPERATURE="${TEMPERATURE:-0.0}"
    export REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-120}"
    export RETRY="${RETRY:-2}"
    ;;
  *)
    fail "Unsupported model: $MODEL"
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
export PRED_FLAVOR="$PRED_FLAVOR_VALUE"
if [[ -n "$MCQ_ROOT" ]]; then
  export MCP_ROOT="$MCQ_ROOT"
fi

RUNNER_ARGS=(--mode "$RUNNER_MODE")
PRED_DIR="$RUNS_ROOT/$MODEL/$TASK/$NORMALIZE_FLAVOR"
if [[ "$RUNNER_MODE" == "cand_state" ]]; then
  RUNNER_ARGS+=(--ss_pos "$SS_POS")
  PRED_DIR="$PRED_DIR/sspos_$(sanitize_path "$SS_POS")"
fi

python "$PROJECT_ROOT/benchmark_val/runner.py" "${RUNNER_ARGS[@]}"

python "$PROJECT_ROOT/scripts/normalize_predictions.py" \
  --input-dir "$PRED_DIR" \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --task "$TASK" \
  --task-type mcq \
  --flavor "$NORMALIZE_FLAVOR"
