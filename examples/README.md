# Examples

This directory contains tiny JSON examples only. Raw Ego4D videos are not included. Place Ego4D full_scale videos under /path/to/ego4d/full_scale and pass that path with --ego4d-root.

The current inference-only release expects task GT roots to contain `train/` or `val/` subdirectories and MCQ roots to contain matching shuffled JSON/JSONL files under the same split. If no tiny sample is present for a task, use your local EgoSAT/Ego4D-derived GT and MCQ files with `--gt-root` and `--mcq-root`.
