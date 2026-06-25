# Synthetic Evaluation Fixtures

These files are synthetic smoke-test fixtures for the public evaluator.
They are not EgoSAT benchmark samples and must not be reported as benchmark results.

Run from the repository root:

```bash
python evaluation/evaluate_main_table.py \
  --pred-root examples/evaluation_fixtures/pred \
  --model synthetic_model \
  --gt-root examples/evaluation_fixtures/gt \
  --mcq-root examples/evaluation_fixtures/mcq \
  --effective-queryset-root examples/evaluation_fixtures/effective \
  --out-dir examples/evaluation_fixtures/out \
  --ss-pos ss_pos_1=sspos_t_1_0 \
  --ss-pos ss_pos_2=sspos_t_2_0 \
  --ss-pos ss_pos_3=sspos_t_4_0 \
  --write-details
```

