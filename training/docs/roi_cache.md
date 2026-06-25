# ROI Cache

ROI-TimeChat full SFT requires an ROI cache. The cache is not stored in GitHub and should be distributed separately, for example through Hugging Face.

## Config Key

Use either CLI:

```bash
--roi-cache-root /path/to/egosat_roi_cache
```

or YAML:

```yaml
model_variant: roi_timechat
roi_cache_root: /path/to/egosat_roi_cache
```

The public wrapper rejects real `roi_timechat` training without this root.

## Expected Layout

The runtime wrapper derives ROI cache paths from manifest metadata. The preferred filename is:

```text
<roi_cache_root>/<video_uid>__<clip_id>__<clip_uid>.roi_cache_merged_fps1.jsonl
```

Fallback names checked by the runtime wrapper include:

```text
<video_uid>__<clip_id>__<clip_uid>.roi_cache_merged_fps1.json
<video_uid>__<clip_id>__<clip_uid>.roi_cache.jsonl
<video_uid>.roi_cache_merged_fps1.jsonl
<video_uid>.roi_cache.jsonl
```

The copied ROI inference adapter uses the same public `ROI_CACHE_ROOT` environment variable.

## Public Manifest Policy

Public manifests should avoid private absolute ROI cache paths. They may include:

```json
{
  "assets": {
    "video_source": "ego4d",
    "requires_ego4d_video": true,
    "requires_roi_cache": true,
    "roi_cache_layout_hint": "{roi_cache_root}/{video_uid}__{clip_id}__{clip_uid}.roi_cache_merged_fps1.jsonl"
  }
}
```

The trainer fills local `roi_cache_path` values into a runtime manifest under `output_dir/_runtime/`.

## Limits

ROI-TimeChat still depends on ROI-aware TimeChat model code in the external TimeChat-Online repo and on a complete ROI cache release. No ROI cache files are committed here.
