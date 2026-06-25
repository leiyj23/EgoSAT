# Model Adapter Guide

This guide explains how to plug a new video-language model into the current EgoSAT runner/helper/evaluator pipeline.

## Overview

EgoSAT inference is split into three pieces:

1. `benchmark_val/runner.py` enumerates GT JSON/JSONL files, finds the raw Ego4D video by `video_uid`, cuts the annotated interval clip, calls the helper, then calls the selected model adapter.
2. `benchmark_val/helper.py` turns each GT file into an effective queryset. It builds task-specific prompts, scheduled query times, strict-online window metadata, and MCQ metadata when needed.
3. A model adapter reads that effective queryset, runs the model on the interval clip without using future frames, and writes one raw per-GT prediction JSON file.

The official main-table evaluator reads the raw per-GT JSON files under `outputs/runs/<model>/<task>/<flavor>/`. Normalized JSONL files are useful for inspection, but they are not the official scorer input.

## Where Adapters Live

Adapters live under:

```text
benchmark_val/llms/
```

The runner does not currently use a Python package registry from `benchmark_val/llms/__init__.py`. Instead, it loads an adapter module from the `ADAPTER_PY` environment variable and requires that module to define `create_adapter()`.

## Existing Reference Adapters

Current adapter files include:

- `benchmark_val/llms/llm_adapter.py`: OpenRouter-style API adapter. This is a task-agnostic reference implementation, not an abstract base class.
- `benchmark_val/llms/qwen2_5_vl_7b_adapter.py`: local Qwen2.5-VL-7B adapter with strict-online windowing, XML-ish response parsing, retries, token probabilities, and `cand_conf` support.
- `benchmark_val/llms/qwen2_5_vl_32b_adapter.py`: local Qwen2.5-VL-32B variant with the same runner contract.
- `benchmark_val/llms/timechat_online_7b_adapter.py`: TimeChat-Online adapter, including optional full-SFT LoRA loading and `cand_conf` diagnostics. Current public shell wrappers do not fully wire this model name yet.
- `benchmark_val/llms/roi_timechat_online_7b_adapter.py`: ROI-aware TimeChat-Online adapter. It requires ROI cache metadata and a DTD/ROI-capable TimeChat implementation. Current public shell wrappers do not fully wire this model name yet.

## Adapter Interface

There is no abstract adapter base class in the current release. The interface is a runner contract:

```python
from pathlib import Path


class MyAdapter:
    def run(
        self,
        video_path: Path,
        queryset_path: Path,
        pred_flavor: str,
        out_json_path: Path,
    ) -> Path:
        ...


def create_adapter():
    return MyAdapter()
```

`runner.py` checks that the module has `create_adapter()` and that the returned object has a `run(...)` method. Existing local adapters also accept flexible `*args, **kwargs`, but a new adapter should support the keyword signature above because that is what `runner.py` calls.

## Input Expectations

`video_path` is the interval clip produced by the runner from the raw Ego4D video. Query times in the effective queryset are relative to this interval clip unless `NOW_TIME_MODE` is changed.

`queryset_path` points to the helper-produced effective queryset. It is a JSON object with fields such as:

- `dataset`, `task`, `task_name`, `video_uid`
- `video_metadata`, including the original interval metadata
- `params`, including `pred_flavor`, `time_mode`, `time_offset_sec`, and task-specific values
- `samples`, a list of query samples

Each sample should be treated as the source of truth for inference. Important sample fields include:

- `idx`: sample identifier used by scoring
- `t_eval` and often `t_eval_rel`: query time
- `prompt`: task-specific prompt already built by `helper.py`
- `window_start_sec` and `window_end_sec`: strict-online evidence window when present
- `lookback_sec`, `lag_sec`, `horizon_sec`, `context_sec`, or `t_target_sec` depending on task
- `mcq`: MCQ answer/options metadata in candidate modes
- `memory`: optional sparse-memory metadata when memory mode is enabled

The adapter should not reshuffle MCQ options. In candidate modes, the helper has already inserted A/B/C/D options into `sample["prompt"]` and copied the matching MCQ metadata into `sample["mcq"]`.

## Output Expectations

The adapter should write `out_json_path` and return its path. For official evaluation, prefer the raw JSON layout used by the Qwen and TimeChat adapters:

```json
{
  "dataset": "Ego4D",
  "task": "sh_rtrv",
  "video_uid": "...",
  "model_name": "my_model",
  "model_id": "/path/to/model-or-checkpoint",
  "source_queryset": "/path/to/effective_queryset.json",
  "video_clip": "/path/to/interval_clip.mp4",
  "generated_at_unix": 0.0,
  "params": {},
  "samples": []
}
```

Each prediction sample should include:

```json
{
  "idx": 0,
  "t_eval": 12.0,
  "t_eval_rel": 12.0,
  "prompt": "...",
  "response_text": "...",
  "clean_response": "...",
  "parsed": {},
  "gen_tokens": [],
  "gen_token_probs": [],
  "sent_logp": null,
  "mean_logp": null,
  "latency_sec": 0.0,
  "diagnostics": {},
  "raw": null
}
```

For MCQ tasks, `parsed["ans"]` should be a single `A`, `B`, `C`, or `D` when possible. The evaluator can also recover answers from `<ANS>...</ANS>`, `clean_response`, `response_text`, or early generated tokens, but an explicit `parsed["ans"]` is safer.

For open-ended XML-ish tasks, adapters usually parse tags into `parsed`, for example:

- now tasks: `state`, `verb`, `noun`, `desc`, `conf`
- past tasks: `ans`, `verb`, `noun`, `desc`, `conf`
- future tasks: `ans`, `verb`, `noun`, `desc`, `conf`

The evaluator resolves the effective queryset from `source_queryset` first. If you cannot write that field, it can also look for `video_metadata.queryset_path`, but `source_queryset` is the recommended field.

## Implementing A New Adapter

Use the helper prompt directly and keep the evidence window strict-online:

```python
import json
import os
import time
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class MyModelAdapter:
    def __init__(self):
        self.model_name = os.environ.get("MODEL_NAME", "my_model")
        self.model_id = os.environ.get("MODEL_ID", "/path/to/model")
        # Load tokenizer/model/processor here.

    def _infer_window(self, sample, params):
        offset = float(params.get("time_offset_sec", 0.0))
        t_eval = float(sample.get("t_eval", 0.0))
        t_eval_rel = float(sample.get("t_eval_rel", t_eval - offset))
        if "window_start_sec" in sample and "window_end_sec" in sample:
            start = max(0.0, float(sample["window_start_sec"]) - offset)
            end = min(t_eval_rel, float(sample["window_end_sec"]) - offset)
        else:
            lookback = float(sample.get("lookback_sec", 20.0))
            end = t_eval_rel
            start = max(0.0, end - lookback)
        return start, max(start, end), t_eval_rel

    def run(self, video_path: Path, queryset_path: Path, pred_flavor: str, out_json_path: Path) -> Path:
        qs = load_json(Path(queryset_path))
        params = qs.get("params", {}) if isinstance(qs.get("params"), dict) else {}
        out = {
            "dataset": qs.get("dataset", "Ego4D"),
            "task": qs.get("task_name") or qs.get("task"),
            "video_uid": qs.get("video_uid") or (qs.get("video_metadata") or {}).get("video_uid", ""),
            "model_name": self.model_name,
            "model_id": self.model_id,
            "source_queryset": str(queryset_path),
            "video_clip": str(video_path),
            "generated_at_unix": time.time(),
            "params": params,
            "samples": [],
        }

        for sample in qs.get("samples", []):
            prompt = str(sample.get("prompt", "")).strip()
            if not prompt:
                continue
            start_sec, end_sec, t_eval_rel = self._infer_window(sample, params)

            # Decode only frames from [start_sec, end_sec] in video_path.
            # Run your model with `prompt`.
            response_text = ""
            clean_response = response_text.strip()
            parsed = {}

            out["samples"].append({
                "idx": sample.get("idx"),
                "t_eval": sample.get("t_eval"),
                "t_eval_rel": sample.get("t_eval_rel", t_eval_rel),
                "prompt": prompt,
                "response_text": response_text,
                "clean_response": clean_response,
                "parsed": parsed,
                "gen_tokens": [],
                "gen_token_probs": [],
                "sent_logp": None,
                "mean_logp": None,
                "latency_sec": None,
                "diagnostics": {
                    "window_start_rel": start_sec,
                    "window_end_rel": end_sec,
                    "pred_flavor_active": pred_flavor,
                },
                "raw": None,
            })

        out_json_path = Path(out_json_path)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_json_path


def create_adapter():
    return MyModelAdapter()
```

## Registering The Model

There are two current paths.

For direct runner usage, set `ADAPTER_PY` yourself:

```bash
export MODEL_NAME=my_model
export TASK=sh_rtrv
export GT_ROOT=/path/to/EgoSAT-data/egosat/gt/sh_rtrv
export MCP_ROOT=/path/to/EgoSAT-data/egosat/mcq_shuffled/sh_rtrv
export VIDEO_ROOT=/path/to/Ego4D/videos
export RUNS_ROOT=outputs/runs
export HELPER_PY=benchmark_val/helper.py
export ADAPTER_PY=benchmark_val/llms/my_model_adapter.py
export PRED_FLAVOR=cand

python benchmark_val/runner.py --mode cand
```

For public shell-wrapper usage, add a model case to the relevant script if needed:

- `scripts/run_mcq_inference.sh`
- `scripts/run_openended_inference.sh`
- `scripts/run_confidence_inference.sh`

Also add a documentation entry to `configs/models.yaml` with the model name, adapter path, default checkpoint, and required environment variables. Current shell wrappers do not parse `configs/models.yaml`; they use an internal `case "$MODEL"` mapping.

## Running Inference

Open-ended:

```bash
bash scripts/run_openended_inference.sh qwen2_5_vl_7b \
  --task sh_rtrv \
  --gt-root /path/to/EgoSAT-data/egosat/gt/sh_rtrv \
  --ego4d-root /path/to/Ego4D/videos \
  --output outputs/qwen_openended_predictions.jsonl
```

MCQ:

```bash
bash scripts/run_mcq_inference.sh qwen2_5_vl_7b \
  --task sh_rtrv \
  --gt-root /path/to/EgoSAT-data/egosat/gt/sh_rtrv \
  --mcq-root /path/to/EgoSAT-data/egosat/mcq_shuffled/sh_rtrv \
  --ego4d-root /path/to/Ego4D/videos \
  --output outputs/qwen_mcq_predictions.jsonl
```

Confidence for main-table `sh_pred`, `ms_pred`, and `ms_rtrv`:

```bash
bash scripts/run_confidence_inference.sh qwen2_5_vl_7b \
  --task ms_pred \
  --gt-root /path/to/EgoSAT-data/egosat/gt/ms_pred \
  --mcq-root /path/to/EgoSAT-data/egosat/mcq_shuffled/ms_pred \
  --ego4d-root /path/to/Ego4D/videos \
  --runs-root outputs/runs
```

## Confidence Outputs

Ordinary candidate outputs may include parsed `<CONF>` values, but the main-table confidence columns for `sh_pred`, `ms_pred`, and `ms_rtrv` use the separate `cand_conf` flavor.

Supported local adapters compute next-token probability mass over A/B/C/D and write it under:

```text
samples[*].diagnostics.prompt_and_encoding_debug.cand_conf_probe.p_cond
```

The evaluator prefers `p_cond[predicted_answer]`, then falls back to the maximum valid `p_cond`, then to `gen_token_probs[0]`. If your model cannot expose logits or token probabilities, it can still run MCQ accuracy, but `cand_conf` confidence metrics may be missing or null.

## ROI-Aware Models

`roi_timechat_online_7b_adapter.py` is an ROI-aware TimeChat adapter. It expects ROI cache metadata through:

- `ROI_CACHE_PATH`: explicit ROI cache file
- `ROI_CACHE_ROOT`: root containing canonical ROI cache files
- `ROI_COORD`: `norm` or `pixel`
- `GAZE_R_DEFAULT`: fallback gaze radius

The adapter converts the interval clip to a 1 FPS cache, selects explicit frame indices, builds `roi_cache["frames"]` in the same order as the selected images, and calls the ROI-DTD-capable TimeChat model with `roi_cache` plus `drop_method`, `drop_threshold`, and `drop_absolute`.

Current limitation: the public shell wrappers do not fully wire the TimeChat and ROI-TimeChat model names yet. Advanced users should call `benchmark_val/runner.py` directly with `ADAPTER_PY` until the wrapper cases are added.

## Debugging Checklist

- `VIDEO_ROOT` does not contain `<video_uid>.mp4`, or the file is nested under a different directory.
- Raw Ego4D videos are missing or inaccessible.
- `MCP_ROOT` does not mirror the GT split and file stem for MCQ tasks.
- `sample["prompt"]` is missing because the helper was not called or `PRED_FLAVOR` was wrong.
- The adapter used frames after `t_eval`; clamp `window_end` to the query time.
- MCQ output is not a single A/B/C/D and cannot be parsed into `parsed["ans"]`.
- `source_queryset` points to a stale absolute path. Use `--path-map` during evaluation or write portable paths in new outputs.
- `cand_conf` files are missing for `sh_pred`, `ms_pred`, or `ms_rtrv` confidence.
- The model does not return token probabilities, so confidence diagnostics are unavailable.
- GPU memory is insufficient; reduce frames, use a smaller checkpoint, or set model-specific memory options.

## Minimal Checklist For Adding A New Model

- Add `benchmark_val/llms/<your_model>_adapter.py`.
- Define `create_adapter()` and implement `run(video_path, queryset_path, pred_flavor, out_json_path)`.
- Use `sample["prompt"]` as built by `helper.py`.
- Decode only the strict-online evidence window.
- Write raw JSON with `source_queryset`, `samples[*].idx`, `response_text`, `clean_response`, `parsed`, and diagnostics.
- For MCQ, write `parsed["ans"]` as A/B/C/D when possible.
- If supporting confidence, implement `cand_conf` and write `cand_conf_probe.p_cond`.
- Add a documentation entry to `configs/models.yaml`.
- Add a shell-wrapper `case "$MODEL"` branch if the model should work through `scripts/run_*_inference.sh`.
- Run a tiny dry or smoke test on a small local subset before launching full evaluation.
