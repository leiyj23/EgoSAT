# Synthetic SFT Manifest Examples

These JSONL files are schema examples only. They are not real Ego4D samples and are not suitable for training a model.

- `tiny_mixed5_cand_manifest.jsonl` shows candidate/action-style SFT rows with MCQ options and tagged action targets such as `<FUTURE>...</FUTURE>` and `<PAST>...</PAST>`.
- `tiny_mixed7_stateheavy_manifest.jsonl` shows state-heavy rows with targets such as `<NOW><STATE>INTERACTION</STATE>...</NOW>`.

The public training entry can dry-run on these files:

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

Real full-SFT manifests should be sanitized with `training/sanitize_sft_manifest.py` and distributed separately, for example through a Hugging Face Dataset.
