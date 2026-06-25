# Legacy SFT References

These files are cleaned references derived from the original internal SFT scripts:

- `lora_projector_sft.py`
- `lora_projector_sft_roi.py`
- `lora_projector_sft_new.py`
- `lora_projector_sft_roi_new.py`

Public users should prefer `training/train_full_sft.sh` and `training/train_full_sft.py`.

The public wrapper treats `mixed5_cand` and `mixed7_stateheavy` as separate SFT tasks. The `_new` files retain explicit resume/continuation mechanics for reference, but that is not the default public recipe.

AB/BB cross-scenario rebuttal scripts and manifests are intentionally excluded from this directory and from the first public full-SFT release.

The hardcoded private defaults from the internal scripts were replaced with environment variables or `/path/to/...` placeholders. These references still depend on the external TimeChat-Online training environment, Ego4D RGB videos, and optional ROI cache assets.
