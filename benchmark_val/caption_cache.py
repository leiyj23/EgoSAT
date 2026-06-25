#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small utilities for Memory Proxy caption and keyframe cache paths."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_CAPTION_CACHE_ROOT = "memory_cache/captions"
DEFAULT_KEYFRAME_CACHE_ROOT = "memory_cache/keyframes"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha1_json(obj: Any) -> str:
    return hashlib.sha1(canonical_json(obj).encode("utf-8")).hexdigest()


def slugify(s: Any) -> str:
    text = str(s if s is not None else "").strip()
    if not text:
        return "unknown"
    text = re.sub(r"[^\w.\-]+", "_", text, flags=re.UNICODE)
    text = text.strip("._-")
    return text or "unknown"


def _ctx(memory_meta: Dict[str, Any]) -> Dict[str, Any]:
    value = memory_meta.get("cache_context", {})
    return value if isinstance(value, dict) else {}


def _sparse(memory_meta: Dict[str, Any]) -> Dict[str, Any]:
    value = memory_meta.get("sparse_window", {})
    return value if isinstance(value, dict) else {}


def _dense(memory_meta: Dict[str, Any]) -> Dict[str, Any]:
    value = memory_meta.get("dense_window", {})
    return value if isinstance(value, dict) else {}


def _round_list(values: Any) -> Any:
    if not isinstance(values, list):
        return []
    out = []
    for value in values:
        try:
            out.append(round(float(value), 6))
        except Exception:
            out.append(value)
    return out


def _interval_id(ctx: Dict[str, Any]) -> str:
    clip_uid = ctx.get("clip_uid")
    clip_id = ctx.get("clip_id")
    if clip_uid not in (None, "") or clip_id not in (None, ""):
        return slugify(f"{clip_id or 'NA'}__{clip_uid or 'NA'}")
    start = ctx.get("interval_start_sec")
    end = ctx.get("interval_end_sec")
    if start is not None or end is not None:
        try:
            return slugify(f"{float(start or 0.0):.3f}-{float(end or 0.0):.3f}")
        except Exception:
            return slugify(f"{start}-{end}")
    return "unknown_interval"


def _fps_tag(value: Any) -> str:
    try:
        text = f"{float(value):g}"
    except Exception:
        text = str(value if value is not None else "0")
    return "fps" + text.replace(".", "p")


def segment_payload(memory_meta: Dict[str, Any]) -> Dict[str, Any]:
    ctx = _ctx(memory_meta)
    sparse = _sparse(memory_meta)
    return {
        "schema_version": memory_meta.get("schema_version", "memory_proxy_v1"),
        "video_uid": ctx.get("video_uid"),
        "clip_uid": ctx.get("clip_uid"),
        "clip_id": ctx.get("clip_id"),
        "interval_start_sec": ctx.get("interval_start_sec"),
        "interval_end_sec": ctx.get("interval_end_sec"),
        "sparse_start_sec": sparse.get("start_sec"),
        "sparse_end_sec": sparse.get("end_sec"),
        "memory_sparse_sec": sparse.get("duration_sec"),
        "memory_keyframe_fps": sparse.get("keyframe_fps"),
        "keyframe_sampling": sparse.get("keyframe_sampling"),
        "keyframe_timestamps_sec": _round_list(sparse.get("expected_keyframe_timestamps_sec", [])),
        "caption_source": memory_meta.get("caption_source"),
        "caption_model": memory_meta.get("caption_model"),
        "caption_prompt_version": memory_meta.get("caption_prompt_version"),
        "caption_input_type": memory_meta.get("caption_input_type"),
    }


def usage_payload(memory_meta: Dict[str, Any]) -> Dict[str, Any]:
    ctx = _ctx(memory_meta)
    sparse = _sparse(memory_meta)
    dense = _dense(memory_meta)
    return {
        "task": ctx.get("task"),
        "split": ctx.get("split"),
        "gt_file_stem": ctx.get("gt_file_stem"),
        "sample_idx": ctx.get("sample_idx"),
        "sample_id": ctx.get("sample_id"),
        "t_eval_rel": ctx.get("t_eval_rel"),
        "dense_start_sec": dense.get("start_sec"),
        "dense_end_sec": dense.get("end_sec"),
        "sparse_start_sec": sparse.get("start_sec"),
        "sparse_end_sec": sparse.get("end_sec"),
        "segment_key": build_segment_key(memory_meta),
    }


def build_segment_key(memory_meta: Dict[str, Any]) -> str:
    return "memcap_" + sha1_json(segment_payload(memory_meta))


def build_usage_key(memory_meta: Dict[str, Any]) -> str:
    return "memuse_" + sha1_json(usage_payload(memory_meta))


def caption_record_path(cache_root: str, memory_meta: Dict[str, Any]) -> str:
    root = Path(cache_root or DEFAULT_CAPTION_CACHE_ROOT).expanduser()
    ctx = _ctx(memory_meta)
    source = slugify(memory_meta.get("caption_source", "openrouter_generated"))
    model = slugify(memory_meta.get("caption_model", "unknown_model"))
    prompt_version = slugify(memory_meta.get("caption_prompt_version", "memory_caption_v1"))
    input_type = slugify(memory_meta.get("caption_input_type", "keyframes"))
    video_uid = slugify(ctx.get("video_uid", "unknown_video"))
    interval = _interval_id(ctx)
    return str(root / source / model / prompt_version / input_type / video_uid / interval / f"{build_segment_key(memory_meta)}.json")


def keyframe_dir_path(keyframe_root: str, memory_meta: Dict[str, Any]) -> str:
    root = Path(keyframe_root or DEFAULT_KEYFRAME_CACHE_ROOT).expanduser()
    ctx = _ctx(memory_meta)
    sparse = _sparse(memory_meta)
    sampling = slugify(sparse.get("keyframe_sampling", "fixed_fps_floor_plus_one_v1"))
    fps = _fps_tag(sparse.get("keyframe_fps", 0.25))
    video_uid = slugify(ctx.get("video_uid", "unknown_video"))
    interval = _interval_id(ctx)
    return str(root / sampling / fps / video_uid / interval / build_segment_key(memory_meta))


def load_caption_record(path: str) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.is_file():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, str(p))
    except Exception:
        try:
            os.unlink(tmp_name)
        except Exception:
            pass
        raise
