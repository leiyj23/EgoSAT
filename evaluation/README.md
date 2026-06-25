# EgoSAT Evaluation

This directory contains the minimal public main-table evaluator for EgoSAT.

The recommended entry point is:

```bash
python evaluation/evaluate_main_table.py \
  --pred-root outputs/runs/qwen2_5_vl_7b \
  --model qwen2_5_vl_7b \
  --mcq-root data/mcq_shuffled \
  --gt-root data/gt \
  --effective-queryset-root outputs/runs/qwen2_5_vl_7b \
  --out-dir outputs/evaluation/qwen2_5_vl_7b \
  --ss-pos ss_pos_1=sspos_t_1_0 \
  --ss-pos ss_pos_2=sspos_t_2_0 \
  --ss-pos ss_pos_3=sspos_t_4_0
```

## Inputs

The evaluator reads raw per-GT prediction JSON files produced by `benchmark_val/runner.py`.
It does not use normalized JSONL as the official input, because normalized rows may drop effective-queryset paths, token probabilities, `cand_conf` diagnostics, state-switch pair metadata, and other fields required by the main-table scorer.

Expected prediction layout:

```text
<pred-root>/
  now_narration/
    cand_state/
    cand_mcq/
  now_state_switch/
    cand_state/
      sspos_*/
  sh_pred/
    cand_full/
    cand_conf/
  ms_pred/
    cand/
    cand_conf/
  ms_rtrv/
    cand/
    cand_conf/
  sh_rtrv/
    cand/
```

The released GT/MCQ/effective-queryset metadata must be available separately. Raw Ego4D RGB videos are not redistributed in this repository.

If raw prediction JSON stores old absolute effective-queryset paths, use repeated `--path-map OLD=NEW`. If the raw path does not exist, the evaluator can fall back to matching by filename stem under `--effective-queryset-root`, `--gt-root`, `--mcq-root`, and `--pred-root`.

## Accuracy vs Confidence Inputs

Accuracy and confidence use different prediction flavors:

```text
now_narration state rec/prec: now_narration/cand_state
now_narration MCQ accuracy:  now_narration/cand_mcq
now_state_switch acc/conf:   now_state_switch/cand_state/sspos_*
sh_pred accuracy:            sh_pred/cand_full
sh_pred confidence:          sh_pred/cand_conf
ms_pred accuracy:            ms_pred/cand
ms_pred confidence:          ms_pred/cand_conf
ms_rtrv accuracy:            ms_rtrv/cand
ms_rtrv confidence:          ms_rtrv/cand_conf
sh_rtrv accuracy:            sh_rtrv/cand
```

Do not compute `sh_pred`, `ms_pred`, or `ms_rtrv` confidence from ordinary `cand` predictions. Use `cand_conf`.

To generate confidence predictions for supported local Qwen models:

```bash
bash scripts/run_confidence_inference.sh qwen2_5_vl_7b \
  --task ms_pred \
  --gt-root data/gt/ms_pred \
  --mcq-root data/mcq_shuffled/ms_pred \
  --ego4d-root /path/to/ego4d/full_scale \
  --runs-root outputs/runs
```

`run_confidence_inference.sh` sets `PRED_FLAVOR=cand_conf` and `RETURN_LOGPROBS=1`. Gemini confidence is not promised by the public wrapper because the required `p_cond` fields may be unavailable.

## Metrics

`now_narration.rec` is `TP / (TP + FN)` for interaction-visible state. GT `INTERACTION` with missing, invalid, or `NO_INTERACTION` prediction is an FN.

`now_narration.prec` is `TP / (TP + FP)`. GT non-interaction predicted as `INTERACTION` is an FP. Invalid and missing predictions are not silently removed.

`now_narration.mcq_acc` uses the full MCQ denominator. Invalid or missing answers count wrong.

`now_state_switch` evaluates three positions:

```text
ss_pos_1 = t:1.0
ss_pos_2 = t:2.0
ss_pos_3 = t:4.0
```

Each switch pair succeeds only if both before and after probes are correct. Main accuracy uses valid-pair denominator and the debug counts also include total seen, missing-role, and invalid-pair counts.

State-switch confidence is the after-transition confidence only, conditioned on successful switch pairs only:

- FG -> BG uses the successful `segment_to_gap` pair's `scan_gap` probe.
- BG -> FG uses the successful `gap_to_segment` pair's `scan_segment` probe.

`sh_pred` reports four groups:

```text
NN = predictable
PN = branch_only
NP = surprise_only
PP = branch_and_surprise
```

Accuracy uses full denominator per group. Confidence is the mean over valid `cand_conf` values per group.

`ms_pred` and `ms_rtrv` report step 0, 1, 2 accuracy and confidence, plus the average over the three steps. Accuracy uses `correct / samples_total`; confidence is mean over valid `cand_conf` values.

`sh_rtrv` reports full-denominator MCQ accuracy only.

## Outputs

The evaluator writes:

```text
<out-dir>/main_table_metrics.json
<out-dir>/main_table_metrics.csv
<out-dir>/main_table_metrics.md
<out-dir>/task_details/*/summary.json
```

Use `--write-details` to also write per-sample JSONL details.

## Legacy Scorers

`evaluation/legacy/` contains copied legacy scorer scripts used as references and for preserved scoring logic. Hardcoded private defaults were neutralized where needed. Teacher/API fallback is disabled unless `EGOSAT_ENABLE_LEGACY_TEACHER_FALLBACK=1` is explicitly set, and official main-table scoring does not use teacher fallback.

## Limitations

- No raw Ego4D RGB videos are included.
- GT, MCQ, and effective-queryset metadata must be provided by the data release.
- The evaluator has not been run on the full private benchmark in this patch.
- Normalized JSONL is useful for inspection but is not the official scorer input.
- Gemini confidence may be unavailable.
- Advanced adapters are not fully wired into this minimal scorer/inference release.

