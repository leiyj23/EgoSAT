#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalize EgoSAT per-GT adapter JSON outputs into prediction JSONL."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SKIP_NAMES = {"manifest.json", "manifest.jsonl"}


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] skip unreadable JSON: {path} ({exc})", file=sys.stderr)
        return None
    return obj if isinstance(obj, dict) else None


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _first_present(mapping: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _dict_get_casefold(mapping: Dict[str, Any], names: Iterable[str]) -> Any:
    wanted = {name.lower() for name in names}
    for key, value in mapping.items():
        if str(key).lower() in wanted and value not in (None, ""):
            return value
    return None


def _extract_letter(text: Any) -> Optional[str]:
    value = _stringify(text).strip()
    if not value:
        return None
    tag_match = re.search(r"<\s*ANS\s*>\s*([A-D])\s*<\s*/\s*ANS\s*>", value, flags=re.IGNORECASE)
    if tag_match:
        return tag_match.group(1).upper()
    answer_match = re.search(r"\b(?:ANS|ANSWER)\s*[:=]\s*([A-D])\b", value, flags=re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()
    bare = re.fullmatch(r"\s*([A-D])\s*[\).:]?\s*", value, flags=re.IGNORECASE)
    if bare:
        return bare.group(1).upper()
    first = re.search(r"\b([A-D])\b", value, flags=re.IGNORECASE)
    return first.group(1).upper() if first else None


def _prediction_for_mcq(sample: Dict[str, Any]) -> Any:
    parsed = sample.get("parsed")
    if isinstance(parsed, dict):
        value = _dict_get_casefold(parsed, ["ans", "answer"])
        if value not in (None, ""):
            return _stringify(value).strip()

    for key in ["prediction", "clean_response", "response_text"]:
        value = sample.get(key)
        if value not in (None, ""):
            letter = _extract_letter(value)
            return letter if letter else _stringify(value).strip()
    return ""


def _prediction_for_openended(sample: Dict[str, Any]) -> Any:
    parsed = sample.get("parsed")
    if parsed not in (None, {}, ""):
        return parsed
    return _first_present(sample, ["clean_response", "response_text", "raw_response"]) or ""


def _metadata(sample: Dict[str, Any], top: Dict[str, Any], flavor: Optional[str]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for key in ["idx", "t_eval", "t_eval_rel", "latency_sec", "sent_logp", "mean_logp", "prompt"]:
        if key in sample:
            meta[key] = sample[key]
    for key in ["dataset", "mode", "run_id", "model_name", "model_id", "source_queryset", "video_clip", "params"]:
        if key in top:
            meta[key] = top[key]
    if flavor:
        meta["flavor"] = flavor
    return meta


def _sample_id(sample: Dict[str, Any], top: Dict[str, Any], source: Path, idx: int) -> str:
    value = _first_present(sample, ["sample_id", "id", "uid", "qid"])
    if value not in (None, ""):
        return str(value)
    video_uid = _first_present(sample, ["video_uid"]) or top.get("video_uid") or source.stem
    raw_idx = _first_present(sample, ["idx", "sample_idx"])
    try:
        idx_value = int(raw_idx) if raw_idx is not None else idx
    except Exception:
        idx_value = idx
    return f"{video_uid}__idx{idx_value}"


def _confidence(sample: Dict[str, Any]) -> Any:
    if "confidence" in sample:
        return sample.get("confidence")
    parsed = sample.get("parsed")
    if isinstance(parsed, dict):
        return _dict_get_casefold(parsed, ["conf", "confidence"])
    return None


def _normalize_file(path: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    if path.name in SKIP_NAMES:
        return []
    top = _load_json(path)
    if not top or "samples" not in top or not isinstance(top.get("samples"), list):
        return []

    records: List[Dict[str, Any]] = []
    top_video_uid = top.get("video_uid") or (top.get("video_metadata") or {}).get("video_uid")
    source_task = top.get("task") or args.task

    for idx, sample in enumerate(top["samples"]):
        if not isinstance(sample, dict):
            continue
        prediction = _prediction_for_mcq(sample) if args.task_type == "mcq" else _prediction_for_openended(sample)
        prediction_text = _stringify(prediction)
        raw_response = _first_present(sample, ["raw_response", "raw", "response_text"])
        parsed = sample.get("parsed") if isinstance(sample.get("parsed"), dict) else {}

        records.append(
            {
                "sample_id": _sample_id(sample, top, path, idx),
                "video_uid": str(sample.get("video_uid") or top_video_uid or ""),
                "task": args.task or source_task,
                "task_type": args.task_type,
                "model": args.model,
                "prediction": prediction,
                "prediction_text": prediction_text,
                "raw_response": _stringify(raw_response),
                "parsed": parsed,
                "confidence": _confidence(sample),
                "source_file": str(path),
                "metadata": _metadata(sample, top, args.flavor),
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing per-GT adapter JSON outputs.")
    parser.add_argument("--output", required=True, help="Output prediction JSONL path.")
    parser.add_argument("--model", required=True, help="Model name written to JSONL.")
    parser.add_argument("--task", required=True, help="Task name written to JSONL.")
    parser.add_argument("--task-type", required=True, choices=["mcq", "openended"])
    parser.add_argument("--flavor", default="", help="Optional runner prediction flavor.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser()
    output = Path(args.output).expanduser()

    if not input_dir.is_dir():
        print(f"[ERROR] --input-dir does not exist or is not a directory: {input_dir}", file=sys.stderr)
        return 2

    records: List[Dict[str, Any]] = []
    for path in sorted(input_dir.rglob("*.json")):
        records.extend(_normalize_file(path, args))

    if not records:
        print(f"[ERROR] No adapter samples found under {input_dir}", file=sys.stderr)
        return 2

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"normalized {len(records)} samples -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
