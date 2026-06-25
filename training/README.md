# EgoSAT Full SFT Training

This directory contains the minimal public full-SFT skeleton for EgoSAT. It is a wrapper-driven release: public users should start from `train_full_sft.sh` or `train_full_sft.py`, not from the reference scripts in `legacy/`.

## Scope

Full SFT in this repo means supervised fine-tuning of TimeChat-Online style models on EgoSAT SFT JSONL manifests. The public skeleton covers four recipes:

- `timechat + mixed5_cand`
- `timechat + mixed7_stateheavy`
- `roi_timechat + mixed5_cand`
- `roi_timechat + mixed7_stateheavy`

`mixed5_cand` and `mixed7_stateheavy` are two different SFT tasks with different `target_text` distributions. They are not stage 1 and stage 2, and `mixed7_stateheavy` is not treated as a continuation of `mixed5_cand` by default.

`timechat` and `roi_timechat` are parallel model variants:

- TimeChat full SFT uses TimeChat-Online without ROI cache.
- ROI-TimeChat full SFT uses ROI-aware TimeChat code and requires a full ROI cache root.

AB/BB cross-scenario rebuttal SFT assets are intentionally excluded from this first public full-SFT release.

## External Assets

This GitHub repo does not redistribute:

- raw Ego4D RGB videos
- full SFT manifests
- ROI cache files
- LoRA weights or large checkpoints
- wandb/tensorboard logs

Prepare these separately:

- Ego4D RGB videos from the official Ego4D channels, passed with `--ego4d-root`.
- Sanitized SFT manifests, preferably from a Hugging Face Dataset or generated with `sanitize_sft_manifest.py`.
- ROI cache for `roi_timechat`, expected to be hosted separately, for example on Hugging Face, and passed with `--roi-cache-root`.
- A local TimeChat-Online repo, passed with `--timechat-repo-root`.

## Dry-Run

Dry-run reads config, checks the manifest exists, prints the resolved plan, and previews 1-2 JSONL rows. It does not initialize TimeChat, load weights, cut video, run GPU kernels, or start training.

```bash
python training/train_full_sft.py \
  --dry-run \
  --model-variant timechat \
  --sft-task mixed5_cand \
  --manifest examples/sft/tiny_mixed5_cand_manifest.jsonl \
  --ego4d-root /path/to/ego4d/full_scale \
  --timechat-repo-root /path/to/TimeChat-Online \
  --base-model wyccccc/TimeChatOnline-7B \
  --output-dir outputs/sft/debug
```

Missing external roots are warnings in dry-run and hard errors for real training.

## Recipes

### timechat + mixed5_cand

```bash
bash training/train_full_sft.sh \
  --model-variant timechat \
  --sft-task mixed5_cand \
  --manifest /path/to/egosat_sft/train_manifest_mixed5_cand.sanitized.jsonl \
  --ego4d-root /path/to/ego4d/full_scale \
  --timechat-repo-root /path/to/TimeChat-Online \
  --base-model wyccccc/TimeChatOnline-7B \
  --output-dir outputs/sft/timechat_mixed5_cand
```

### timechat + mixed7_stateheavy

```bash
bash training/train_full_sft.sh \
  --model-variant timechat \
  --sft-task mixed7_stateheavy \
  --manifest /path/to/egosat_sft/train_manifest_mixed7_stateheavy.sanitized.jsonl \
  --ego4d-root /path/to/ego4d/full_scale \
  --timechat-repo-root /path/to/TimeChat-Online \
  --base-model wyccccc/TimeChatOnline-7B \
  --output-dir outputs/sft/timechat_mixed7_stateheavy
```

### roi_timechat + mixed5_cand

```bash
bash training/train_full_sft.sh \
  --model-variant roi_timechat \
  --sft-task mixed5_cand \
  --manifest /path/to/egosat_sft/train_manifest_mixed5_cand.sanitized.jsonl \
  --ego4d-root /path/to/ego4d/full_scale \
  --roi-cache-root /path/to/egosat_roi_cache \
  --timechat-repo-root /path/to/TimeChat-Online \
  --base-model wyccccc/TimeChatOnline-7B \
  --output-dir outputs/sft/roi_timechat_mixed5_cand
```

### roi_timechat + mixed7_stateheavy

```bash
bash training/train_full_sft.sh \
  --model-variant roi_timechat \
  --sft-task mixed7_stateheavy \
  --manifest /path/to/egosat_sft/train_manifest_mixed7_stateheavy.sanitized.jsonl \
  --ego4d-root /path/to/ego4d/full_scale \
  --roi-cache-root /path/to/egosat_roi_cache \
  --timechat-repo-root /path/to/TimeChat-Online \
  --base-model wyccccc/TimeChatOnline-7B \
  --output-dir outputs/sft/roi_timechat_mixed7_stateheavy
```

## Configs

Equivalent YAML configs live in `training/configs/`:

- `mixed5_cand_timechat.yaml`
- `mixed5_cand_roi_timechat.yaml`
- `mixed7_stateheavy_timechat.yaml`
- `mixed7_stateheavy_roi_timechat.yaml`

Use them with:

```bash
bash training/train_full_sft.sh --config training/configs/mixed5_cand_timechat.yaml --dry-run
```

CLI arguments override YAML values.

## Manifest Preparation

Do not commit raw internal `train_manifest_mixed*.jsonl` files to GitHub. They may contain private absolute paths for interval clips, ROI cache, MCQ files, and source GT.

Use:

```bash
python training/sanitize_sft_manifest.py \
  --input /path/to/raw/train_manifest_mixed5_cand.jsonl \
  --output /tmp/train_manifest_mixed5_cand.sanitized.jsonl \
  --task mixed5_cand
```

For an ROI-specific dataset view:

```bash
python training/sanitize_sft_manifest.py \
  --input /path/to/raw/train_manifest_mixed5_cand.jsonl \
  --output /tmp/train_manifest_mixed5_cand.roi.sanitized.jsonl \
  --task mixed5_cand \
  --requires-roi-cache
```

The training wrapper creates a runtime manifest under `output_dir/_runtime/` and adds local interval clip cache paths there. ROI cache paths are derived from `--roi-cache-root` when needed.

## Outputs

The cleaned legacy trainers save PEFT-style checkpoints under the output directory:

- `checkpoints/step_XXXXXXX/`
- `final/step_XXXXXXX/`
- LoRA adapter files written by `save_pretrained`
- optional `extra_trainable.pt` for projector/merger-style trainable weights
- `meta.json`

## Inference And Evaluation

Set `TIMECHAT_FT_DIR` to a trained LoRA adapter directory, usually a `final/step_XXXXXXX` path, and keep `extra_trainable.pt` in the same directory if it was saved.

TimeChat inference needs:

```bash
export TIMECHAT_REPO_ROOT=/path/to/TimeChat-Online
export TIMECHAT_FT_DIR=/path/to/outputs/sft/timechat_mixed5_cand/final/step_XXXXXXX
```

ROI-TimeChat inference also needs:

```bash
export ROI_CACHE_ROOT=/path/to/egosat_roi_cache
```

Cleaned TimeChat and ROI-TimeChat adapters are present under `benchmark_val/llms/` and model registry entries are documented in `configs/models.yaml`. The existing public shell inference wrappers still focus on the Qwen/Gemini minimal release; full wrapper integration for TimeChat SFT inference remains a follow-up.

## Optional Resume

`--resume-from` is available for explicit continuation experiments and delegates to the resume-capable legacy references. It is not used by the public default recipes. In particular, `mixed7_stateheavy` can be run as an independent full SFT task.
