# EgoSAT Inference

This initial release exposes a minimal inference pipeline around the recovered EgoSAT runner assets. It is intended to show and run the public inference path only: enumerate GT JSON/JSONL files, cut interval clips from Ego4D RGB videos, build prompts with the task helper, call a selected model adapter, and normalize per-GT JSON outputs into prediction JSONL.

## Supported Models

The first wrapper version officially supports:

- `qwen2_5_vl_7b`
- `qwen2_5_vl_32b`
- `gemini_pro` via OpenRouter

Future or experimental adapters are not wired into the public wrappers yet:

- Flash-VStream
- Dispider
- VideoLLM-Online
- InternVideo
- Video-LLaVA
- LLaVA-OneVision

Cleaned TimeChat-Online and ROI-TimeChat adapters are now included for full-SFT handoff, but the public shell wrappers still do not parse `configs/models.yaml` and are not yet the recommended TimeChat SFT inference entry.

`configs/models.yaml` records the current model registry for documentation. The shell wrappers do not parse YAML yet; they use an internal `case "$MODEL"` mapping.

## TimeChat Full-SFT Checkpoints

Full-SFT training writes a LoRA adapter directory plus optional `extra_trainable.pt`; see [../training/README.md](../training/README.md). To evaluate a trained TimeChat SFT checkpoint through the copied adapter, set:

```bash
export TIMECHAT_REPO_ROOT=/path/to/TimeChat-Online
export TIMECHAT_FT_DIR=/path/to/outputs/sft/timechat_mixed5_cand/final/step_XXXXXXX
```

For ROI-TimeChat also set:

```bash
export ROI_CACHE_ROOT=/path/to/egosat_roi_cache
```

The model registry entries are:

- `timechat_online_7b_full_sft`
- `roi_timechat_online_7b_full_sft`

The current `run_*_inference.sh` wrappers still need a small model-case integration before these names can be used exactly like the Qwen entries. Until then, advanced users can call `benchmark_val/runner.py` with `ADAPTER_PY=benchmark_val/llms/timechat_online_7b_adapter.py` or `ADAPTER_PY=benchmark_val/llms/roi_timechat_online_7b_adapter.py`.

## Data And Videos

Raw Ego4D RGB videos are not redistributed. Download Ego4D videos from the official Ego4D channels and provide the local root with `--ego4d-root`. The runner looks for videos by `video_uid`, primarily as:

```text
<ego4d-root>/<video_uid>.mp4
```

The current wrappers use `--gt-root` and `--mcq-root`, not a unified `--manifest`. A GT root should contain split subdirectories such as:

```text
examples/gt/sh_rtrv/val/<stem>.jsonl
```

MCQ roots should mirror the split and file stem:

```text
examples/mcq_shuffled/sh_rtrv/val/<stem>.jsonl
```

## Open-Ended Inference

```bash
bash scripts/run_openended_inference.sh qwen2_5_vl_7b \
  --task sh_rtrv \
  --gt-root examples/gt/sh_rtrv \
  --ego4d-root /path/to/ego4d/full_scale \
  --output outputs/qwen_openended_predictions.jsonl
```

For `now_state_switch`, the wrapper passes `--ss-pos` to the runner and defaults to `t:1.0`:

```bash
bash scripts/run_openended_inference.sh qwen2_5_vl_7b \
  --task now_state_switch \
  --gt-root examples/gt/now_state_switch \
  --ego4d-root /path/to/ego4d/full_scale \
  --ss-pos t:1.0 \
  --output outputs/qwen_now_state_switch_open.jsonl
```

## MCQ Inference

```bash
bash scripts/run_mcq_inference.sh qwen2_5_vl_7b \
  --task sh_rtrv \
  --gt-root examples/gt/sh_rtrv \
  --mcq-root examples/mcq_shuffled/sh_rtrv \
  --ego4d-root /path/to/ego4d/full_scale \
  --output outputs/qwen_mcq_predictions.jsonl
```

Task mapping:

- `now_narration`: runner mode `cand`, normalized from `cand_mcq`, default MCQ root `data/mcq_shuffled/now_narration_action`
- `now_state_switch`: runner mode `cand_state`, default `--ss-pos t:2.0`
- `sh_rtrv`: runner mode `cand`, default MCQ root `data/mcq_shuffled/sh_rtrv`
- `ms_rtrv`: runner mode `cand`, default MCQ root `data/mcq_shuffled/ms_rtrv`
- `ms_pred`: runner mode `cand`, default MCQ root `data/mcq_shuffled/ms_pred`
- `sh_pred`: runner mode `cand`, output flavor `cand_full`, exports `SH_PRED_ALL_STATES=1`, default MCQ root `data/mcq_shuffled/sh_pred`

## Confidence Inference For Evaluation

Main-table confidence columns for `sh_pred`, `ms_pred`, and `ms_rtrv` require `cand_conf` raw predictions, not ordinary `cand` predictions.

```bash
bash scripts/run_confidence_inference.sh qwen2_5_vl_7b \
  --task ms_pred \
  --gt-root examples/gt/ms_pred \
  --mcq-root examples/mcq_shuffled/ms_pred \
  --ego4d-root /path/to/ego4d/full_scale \
  --runs-root outputs/runs
```

This wrapper sets `PRED_FLAVOR=cand_conf` and `RETURN_LOGPROBS=1`. It currently supports the local Qwen wrappers. `gemini_pro` is rejected for `cand_conf` unless a future adapter exposes the required confidence fields.

See [../evaluation/README.md](../evaluation/README.md) for official scorer usage.

## OpenRouter / Gemini

Set your API key outside the repository. Do not write it into scripts or config files.

```bash
read -r -s OPENROUTER_API_KEY
export OPENROUTER_API_KEY

bash scripts/run_openended_inference.sh gemini_pro \
  --task sh_rtrv \
  --gt-root examples/gt/sh_rtrv \
  --ego4d-root /path/to/ego4d/full_scale \
  --openrouter-model google/gemini-2.5-pro \
  --output outputs/gemini_openended_predictions.jsonl
```

API costs may apply. The adapter does not print or save the API key.

## Local Qwen

The Qwen wrappers default to public Hugging Face model IDs. You can override them with a local checkpoint path:

```bash
bash scripts/run_mcq_inference.sh qwen2_5_vl_32b \
  --task sh_rtrv \
  --gt-root examples/gt/sh_rtrv \
  --mcq-root examples/mcq_shuffled/sh_rtrv \
  --ego4d-root /path/to/ego4d/full_scale \
  --model-id /path/to/Qwen2.5-VL-32B-Instruct \
  --num-frames 16 \
  --torch-dtype bfloat16 \
  --gpu 0,1 \
  --output outputs/qwen32b_mcq_predictions.jsonl
```

## Output JSONL

The runner writes intermediate per-GT JSON files under:

```text
<runs-root>/<model>/<task>/<flavor>/
```

The wrapper then calls `scripts/normalize_predictions.py` to produce the requested JSONL. Each row contains at least:

```json
{
  "sample_id": "...",
  "video_uid": "...",
  "task": "sh_rtrv",
  "task_type": "mcq",
  "model": "qwen2_5_vl_7b",
  "prediction": "...",
  "prediction_text": "...",
  "raw_response": "...",
  "parsed": {},
  "confidence": null,
  "source_file": "...",
  "metadata": {}
}
```

## Current Limits

- The public CLI uses `--gt-root` and `--mcq-root`; there is no complete manifest CLI yet.
- The runner still generates intermediate per-GT JSON before normalization.
- Official evaluation reads the raw per-GT JSON outputs, not normalized JSONL, because scoring needs effective-queryset metadata and confidence diagnostics.
- No raw Ego4D RGB videos are redistributed.
- No full benchmark construction scripts are included in this inference-only release.
- Full SFT training now has a minimal public skeleton under `training/`, but full data release, full TimeChat inference wrapper integration, and branchiness/surprise construction remain follow-up work.
