#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash training/train_full_sft.sh --model-variant VARIANT --sft-task TASK --manifest FILE --ego4d-root DIR --timechat-repo-root DIR --base-model ID --output-dir DIR [options]

Variants:
  timechat | roi_timechat

SFT tasks:
  mixed5_cand | mixed7_stateheavy

Required for roi_timechat:
  --roi-cache-root DIR

Options:
  --config FILE              Optional YAML config. CLI values override config.
  --lora-r N
  --lora-alpha N
  --lora-dropout FLOAT
  --learning-rate FLOAT
  --epochs N
  --batch-size N
  --grad-accum N
  --max-frames N
  --seed N
  --save-every-steps N
  --log-every N
  --weight-decay FLOAT
  --max-steps N
  --resume-from DIR          Optional checkpoint continuation; not the public default.
  --dry-run                  Inspect config and 1-2 manifest rows without loading the model.
  --help

Examples:
  bash training/train_full_sft.sh \
    --model-variant timechat \
    --sft-task mixed5_cand \
    --manifest /path/to/egosat_sft/train_manifest_mixed5_cand.sanitized.jsonl \
    --ego4d-root /path/to/ego4d/full_scale \
    --timechat-repo-root /path/to/TimeChat-Online \
    --base-model wyccccc/TimeChatOnline-7B \
    --output-dir outputs/sft/timechat_mixed5_cand

  bash training/train_full_sft.sh \
    --model-variant roi_timechat \
    --sft-task mixed7_stateheavy \
    --manifest /path/to/egosat_sft/train_manifest_mixed7_stateheavy.sanitized.jsonl \
    --ego4d-root /path/to/ego4d/full_scale \
    --roi-cache-root /path/to/egosat_roi_cache \
    --timechat-repo-root /path/to/TimeChat-Online \
    --base-model wyccccc/TimeChatOnline-7B \
    --output-dir outputs/sft/roi_timechat_mixed7_stateheavy
USAGE
}

fail() {
  echo "[ERROR] $*" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

if [[ $# -eq 0 || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

CONFIG=""
MODEL_VARIANT=""
SFT_TASK=""
MANIFEST=""
EGO4D_ROOT=""
ROI_CACHE_ROOT=""
TIMECHAT_REPO_ROOT=""
BASE_MODEL=""
OUTPUT_DIR=""

args=("$@")
idx=0
while [[ $idx -lt ${#args[@]} ]]; do
  arg="${args[$idx]}"
  case "$arg" in
    --config)
      ((idx + 1 < ${#args[@]})) || fail "Missing value for --config"
      CONFIG="${args[$((idx + 1))]}"
      idx=$((idx + 2))
      ;;
    --model-variant)
      ((idx + 1 < ${#args[@]})) || fail "Missing value for --model-variant"
      MODEL_VARIANT="${args[$((idx + 1))]}"
      idx=$((idx + 2))
      ;;
    --sft-task)
      ((idx + 1 < ${#args[@]})) || fail "Missing value for --sft-task"
      SFT_TASK="${args[$((idx + 1))]}"
      idx=$((idx + 2))
      ;;
    --manifest)
      ((idx + 1 < ${#args[@]})) || fail "Missing value for --manifest"
      MANIFEST="${args[$((idx + 1))]}"
      idx=$((idx + 2))
      ;;
    --ego4d-root)
      ((idx + 1 < ${#args[@]})) || fail "Missing value for --ego4d-root"
      EGO4D_ROOT="${args[$((idx + 1))]}"
      idx=$((idx + 2))
      ;;
    --roi-cache-root)
      ((idx + 1 < ${#args[@]})) || fail "Missing value for --roi-cache-root"
      ROI_CACHE_ROOT="${args[$((idx + 1))]}"
      idx=$((idx + 2))
      ;;
    --timechat-repo-root)
      ((idx + 1 < ${#args[@]})) || fail "Missing value for --timechat-repo-root"
      TIMECHAT_REPO_ROOT="${args[$((idx + 1))]}"
      idx=$((idx + 2))
      ;;
    --base-model)
      ((idx + 1 < ${#args[@]})) || fail "Missing value for --base-model"
      BASE_MODEL="${args[$((idx + 1))]}"
      idx=$((idx + 2))
      ;;
    --output-dir)
      ((idx + 1 < ${#args[@]})) || fail "Missing value for --output-dir"
      OUTPUT_DIR="${args[$((idx + 1))]}"
      idx=$((idx + 2))
      ;;
    --lora-r|--lora-alpha|--lora-dropout|--learning-rate|--epochs|--batch-size|--grad-accum|--max-frames|--seed|--save-every-steps|--log-every|--weight-decay|--max-steps|--resume-from)
      ((idx + 1 < ${#args[@]})) || fail "Missing value for $arg"
      idx=$((idx + 2))
      ;;
    --dry-run)
      idx=$((idx + 1))
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown argument: $arg"
      ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  [[ -n "$MODEL_VARIANT" ]] || fail "Missing --model-variant"
  [[ -n "$SFT_TASK" ]] || fail "Missing --sft-task"
  [[ -n "$MANIFEST" ]] || fail "Missing --manifest"
  [[ -n "$EGO4D_ROOT" ]] || fail "Missing --ego4d-root"
  [[ -n "$TIMECHAT_REPO_ROOT" ]] || fail "Missing --timechat-repo-root"
  [[ -n "$BASE_MODEL" ]] || fail "Missing --base-model"
  [[ -n "$OUTPUT_DIR" ]] || fail "Missing --output-dir"
fi

if [[ "$MODEL_VARIANT" == "roi_timechat" && -z "$ROI_CACHE_ROOT" && -z "$CONFIG" ]]; then
  fail "--roi-cache-root is required for --model-variant roi_timechat"
fi

exec "$PYTHON_BIN" "$PROJECT_ROOT/training/train_full_sft.py" "$@"
