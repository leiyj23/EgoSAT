#!/usr/bin/env python3
"""Sanitize internal EgoSAT SFT manifests for public dataset release."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


BANNED_PATH_KEYS = {
    "interval_clip_path",
    "roi_cache_path",
    "full_interval_file",
    "mcp_path",
    "mcp_file",
    "source_gt",
    "source_gt_path",
    "source_effective_queryset",
    "matched_full_interval",
}

VIDEO_METADATA_KEEP = {
    "interval_start_sec",
    "interval_end_sec",
    "duration_sec",
    "split",
    "clip_uid",
    "clip_id",
    "interval_len_sec",
    "fps",
    "width",
    "height",
}

PRIVATE_PATH_RE = re.compile(r"(^|[\"'=:\s])/(home|data)/")
MISSING = object()


def contains_private_path(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return bool(PRIVATE_PATH_RE.search(value)) or "/home/" in value or "/data/" in value


def sanitize_any(value: Any, stats: Dict[str, int], key: str = "") -> Any:
    if key.lower() in BANNED_PATH_KEYS:
        stats["removed_private_path_fields"] += 1
        return MISSING
    if isinstance(value, str):
        if contains_private_path(value):
            stats["removed_private_path_fields"] += 1
            return MISSING
        return value
    if isinstance(value, list):
        out = []
        for item in value:
            cleaned = sanitize_any(item, stats)
            if cleaned is not MISSING:
                out.append(cleaned)
        return out
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for child_key, child_value in value.items():
            cleaned = sanitize_any(child_value, stats, key=str(child_key))
            if cleaned is not MISSING:
                out[str(child_key)] = cleaned
        return out
    return value


def scan_private_paths(value: Any, path: str = "$") -> List[str]:
    hits: List[str] = []
    if isinstance(value, str):
        if contains_private_path(value):
            hits.append(path)
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            hits.extend(scan_private_paths(item, f"{path}[{idx}]"))
    elif isinstance(value, dict):
        for key, item in value.items():
            hits.extend(scan_private_paths(item, f"{path}.{key}"))
    return hits


def sanitize_video_metadata(value: Any, stats: Dict[str, int]) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, Any] = {}
    for key in VIDEO_METADATA_KEEP:
        if key in value:
            cleaned = sanitize_any(value[key], stats, key=key)
            if cleaned is not MISSING:
                out[key] = cleaned
    return out


def sanitize_record(rec: Dict[str, Any], sft_task: str, requires_roi_cache: bool, stats: Dict[str, int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ["dataset", "video_uid", "task", "target_text"]:
        if key in rec:
            cleaned = sanitize_any(rec[key], stats, key=key)
            if cleaned is not MISSING:
                out[key] = cleaned

    out["sft_task"] = sft_task
    out["video_metadata"] = sanitize_video_metadata(rec.get("video_metadata"), stats)

    params = sanitize_any(rec.get("params", {}), stats, key="params")
    out["params"] = params if isinstance(params, dict) else {}

    sample = sanitize_any(rec.get("sample", {}), stats, key="sample")
    out["sample"] = sample if isinstance(sample, dict) else {}

    out["assets"] = {
        "video_source": "ego4d",
        "requires_ego4d_video": True,
        "requires_roi_cache": bool(requires_roi_cache),
    }
    if requires_roi_cache:
        out["assets"]["roi_cache_layout_hint"] = (
            "{roi_cache_root}/{video_uid}__{clip_id}__{clip_uid}.roi_cache_merged_fps1.jsonl"
        )

    return out


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise RuntimeError(f"JSONL row must be an object at {path}:{line_no}")
            yield line_no, obj


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove private path fields from raw EgoSAT full-SFT manifests."
    )
    parser.add_argument("--input", required=True, help="Raw internal JSONL manifest.")
    parser.add_argument("--output", required=True, help="Sanitized JSONL output path.")
    parser.add_argument("--task", required=True, choices=["mixed5_cand", "mixed7_stateheavy"])
    parser.add_argument(
        "--requires-roi-cache",
        action="store_true",
        help="Mark assets.requires_roi_cache=true for an ROI-specific manifest view.",
    )
    parser.add_argument(
        "--allow-private-paths",
        action="store_true",
        help="Write output even if suspicious private paths remain; exits zero only with this flag.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    if not input_path.exists():
        print(f"[ERROR] --input does not exist: {input_path}", file=sys.stderr)
        return 2

    stats: Dict[str, int] = {
        "samples": 0,
        "removed_private_path_fields": 0,
        "samples_with_suspicious_path": 0,
    }
    tasks: Dict[str, int] = {}
    suspicious_examples: List[Dict[str, Any]] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for line_no, rec in iter_jsonl(input_path):
            cleaned = sanitize_record(rec, args.task, args.requires_roi_cache, stats)
            hits = scan_private_paths(cleaned)
            if hits:
                stats["samples_with_suspicious_path"] += 1
                if len(suspicious_examples) < 10:
                    suspicious_examples.append({"line": line_no, "paths": hits[:10]})
            task_name = str(cleaned.get("task", ""))
            tasks[task_name] = tasks.get(task_name, 0) + 1
            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            stats["samples"] += 1

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "sft_task": args.task,
        "samples": stats["samples"],
        "tasks": dict(sorted(tasks.items())),
        "removed_private_path_fields": stats["removed_private_path_fields"],
        "samples_with_suspicious_path": stats["samples_with_suspicious_path"],
        "suspicious_examples": suspicious_examples,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if stats["samples_with_suspicious_path"] and not args.allow_private_paths:
        print(
            "[ERROR] Suspicious private paths remain. Re-run with --allow-private-paths only for debugging.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
