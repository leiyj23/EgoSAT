#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""I/O helpers for the public EgoSAT evaluator."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SKIP_JSON_NAMES = {
    "manifest.json",
    "manifest.jsonl",
    "summary.json",
    "summary_all.json",
    "details.json",
    "teacher_cache.json",
    "main_table_metrics.json",
}


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: str | Path) -> List[Any]:
    records: List[Any] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_json_or_jsonl(path: str | Path) -> Any:
    p = Path(path)
    raw = p.read_text(encoding="utf-8").strip()
    if not raw:
        raw = p.read_text(encoding="utf-8-sig").strip()
    if not raw:
        raise ValueError(f"empty JSON/JSONL file: {p}")
    try:
        return json.loads(raw)
    except Exception:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if len(records) == 1:
            return records[0]
        return records


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        ordered: List[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    ordered.append(str(key))
                    seen.add(key)
        fieldnames = ordered
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_markdown_table(path: str | Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col)
            if isinstance(val, float):
                text = f"{val:.6f}"
            elif val is None:
                text = ""
            else:
                text = str(val)
            vals.append(text.replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def scan_prediction_jsons(pred_dir: str | Path) -> List[Path]:
    root = Path(pred_dir)
    if not root.is_dir():
        return []
    out: List[Path] = []
    for path in sorted(root.rglob("*.json"), key=lambda p: p.as_posix()):
        name = path.name.lower()
        if name in SKIP_JSON_NAMES:
            continue
        if "manifest" in name or "summary" in name or "details" in name or "teacher_cache" in name:
            continue
        out.append(path)
    return out


def parse_path_maps(items: Optional[Sequence[str]]) -> List[Tuple[str, str]]:
    maps: List[Tuple[str, str]] = []
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--path-map must be OLD=NEW, got: {item}")
        old, new = item.split("=", 1)
        old = old.strip()
        new = new.strip()
        if not old or not new:
            raise ValueError(f"--path-map must be OLD=NEW, got: {item}")
        maps.append((old, new))
    return maps


def apply_path_maps(raw_path: str, path_maps: Sequence[Tuple[str, str]]) -> str:
    out = raw_path
    for old, new in path_maps:
        if out == old or out.startswith(old.rstrip("/\\") + "/") or out.startswith(old.rstrip("/\\") + "\\"):
            return new.rstrip("/\\") + out[len(old.rstrip("/\\")) :]
    return out


def resolve_metadata_path(
    raw_path: Optional[str],
    *,
    roots: Sequence[str | Path],
    path_maps: Sequence[Tuple[str, str]] = (),
) -> Optional[Path]:
    candidates: List[Path] = []
    if raw_path:
        mapped = apply_path_maps(str(raw_path).strip(), path_maps)
        candidates.append(Path(mapped).expanduser())
        if mapped != raw_path:
            candidates.append(Path(str(raw_path)).expanduser())

    for cand in candidates:
        if cand.is_file():
            return cand

    if not raw_path:
        return None

    name = Path(str(raw_path)).name
    stem = Path(str(raw_path)).stem
    suffixes = [Path(str(raw_path)).suffix] if Path(str(raw_path)).suffix else [".json", ".jsonl"]
    for root in roots:
        r = Path(root).expanduser()
        if not r.is_dir():
            continue
        exact = list(r.rglob(name)) if name else []
        for hit in sorted(exact, key=lambda p: p.as_posix()):
            if hit.is_file():
                return hit
        for suffix in suffixes + [".json", ".jsonl"]:
            for hit in sorted(r.rglob(stem + suffix), key=lambda p: p.as_posix()):
                if hit.is_file():
                    return hit
    return None


def collect_queryset_path_strings(obj: Any, key_path: str = "") -> List[str]:
    out: List[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_key = f"{key_path}.{key}" if key_path else str(key)
            if isinstance(value, str):
                lk = str(key).lower()
                lv = value.lower()
                if "queryset" in lk or "queryset" in lv or "derived_" in lv:
                    out.append(value)
            out.extend(collect_queryset_path_strings(value, next_key))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            out.extend(collect_queryset_path_strings(value, f"{key_path}[{i}]"))
    return out


def resolve_queryset_for_prediction(
    pred_obj: Mapping[str, Any],
    *,
    roots: Sequence[str | Path],
    path_maps: Sequence[Tuple[str, str]] = (),
) -> Optional[Path]:
    values: List[str] = []
    v = pred_obj.get("source_queryset")
    if isinstance(v, str) and v.strip():
        values.append(v)
    vm = pred_obj.get("video_metadata")
    if isinstance(vm, Mapping):
        v2 = vm.get("queryset_path")
        if isinstance(v2, str) and v2.strip():
            values.append(v2)
    values.extend(collect_queryset_path_strings(pred_obj))

    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        path = resolve_metadata_path(value, roots=roots, path_maps=path_maps)
        if path is not None:
            return path
    return None


def index_samples_by_key(samples: Iterable[Any]) -> Dict[Any, Dict[str, Any]]:
    out: Dict[Any, Dict[str, Any]] = {}
    for pos, sample in enumerate(samples):
        if not isinstance(sample, dict):
            continue
        key = sample.get("idx", None)
        if isinstance(key, float) and key.is_integer():
            key = int(key)
        if key is None:
            key = sample.get("sample_id") or sample.get("id") or pos
        out[key] = sample
    return out

