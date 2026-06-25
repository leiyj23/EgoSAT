# SFT Data Format

EgoSAT full SFT manifests are JSONL files: one JSON object per training sample.

## Required Fields

Each row should include:

```json
{
  "dataset": "Ego4D",
  "video_uid": "...",
  "sft_task": "mixed5_cand",
  "task": "ms_pred",
  "video_metadata": {
    "clip_uid": "...",
    "clip_id": "...",
    "interval_start_sec": 0.0,
    "interval_end_sec": 300.0,
    "duration_sec": 300.0,
    "split": "train"
  },
  "params": {},
  "sample": {
    "prompt": "..."
  },
  "target_text": "...",
  "assets": {
    "video_source": "ego4d",
    "requires_ego4d_video": true,
    "requires_roi_cache": false
  }
}
```

The trainer requires `sample.prompt` and `target_text`. `video_uid` and `video_metadata.interval_start_sec` / `interval_end_sec` are used to cut an interval clip from the local Ego4D RGB video root.

`clip_uid` and `clip_id` are optional but recommended. ROI-TimeChat uses them to derive canonical ROI cache filenames.

## Optional Fields

Useful optional fields include:

- `sample.mcq` for candidate/MCQ-style tasks.
- `sample.state`, `sample.region`, `sample.visible_interaction`, and state-switch metadata for state-heavy tasks.
- task-specific timing fields such as `t_eval`, `lookback_sec`, `horizon_sec`, `window_start_sec`, and `window_end_sec`.
- `params.pred_flavor`, `params.time_mode`, `params.leads_sec`, or `params.lags_sec`.

The sanitizer preserves non-private fields so downstream debugging can still inspect task metadata.

## mixed5_cand vs mixed7_stateheavy

`mixed5_cand` contains candidate/action-style targets. Examples include:

```text
<FUTURE><ANS>A</ANS><VERB>pick</VERB><NOUN>cup</NOUN><DESC>YOU pick cup</DESC><CONF>0.00</CONF></FUTURE>
<PAST><ANS>C</ANS><VERB>move</VERB><NOUN>bowl</NOUN><DESC>YOU move bowl</DESC><CONF>0.00</CONF></PAST>
```

`mixed7_stateheavy` contains a different target distribution, including state-heavy rows such as:

```text
<NOW><STATE>INTERACTION</STATE><CONF>0.00</CONF></NOW>
<NOW><STATE>NO_INTERACTION</STATE><CONF>0.00</CONF></NOW>
```

These are two independent SFT tasks. Do not describe `mixed7_stateheavy` as a second stage of `mixed5_cand`.

## Ego4D Dependency

Raw Ego4D RGB videos are not redistributed in this repository. The public trainer expects users to pass:

```bash
--ego4d-root /path/to/ego4d/full_scale
```

The runtime code looks for `<ego4d-root>/<video_uid>.mp4` and may recursively search below the root.

## ROI Cache Dependency

The same sanitized manifest can be used by TimeChat and ROI-TimeChat. ROI-TimeChat additionally requires:

```bash
--roi-cache-root /path/to/egosat_roi_cache
```

The public manifest should not contain private absolute ROI cache paths.

## Private Path Removal

Raw internal manifests may contain fields such as:

- `interval_clip_path`
- `roi_cache_path`
- `full_interval_file`
- `mcp_path`
- `source_gt`
- `source_effective_queryset`
- values containing private `/home/...` or `/data/...` paths

Use `training/sanitize_sft_manifest.py` before publishing a manifest. If any private path remains, the sanitizer exits nonzero unless `--allow-private-paths` is explicitly set.

## Recommended Release Format

Publish full SFT manifests as a Hugging Face Dataset or another external dataset artifact. Keep GitHub limited to schema examples and tooling.
