#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared source statistics helpers for MCQ scorer scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STRONG_DISTURB_SOURCES = {"prev", "future"}
KNOWN_OPTION_SOURCES = ("gt", "prev", "future", "absurd")
LETTERS = ("A", "B", "C", "D")


def letter_to_idx(letter: Any) -> Optional[int]:
    if not isinstance(letter, str):
        return None
    s = letter.strip().upper()
    if len(s) != 1 or s not in LETTERS:
        return None
    return LETTERS.index(s)


def idx_to_letter(idx: Any) -> Optional[str]:
    try:
        i = int(idx)
    except Exception:
        return None
    if 0 <= i < len(LETTERS):
        return LETTERS[i]
    return None


def safe_rate(num: Any, den: Any) -> Optional[float]:
    try:
        n = float(num)
        d = float(den)
    except Exception:
        return None
    if d <= 0.0:
        return None
    return float(n / d)


def _legacy_rate(num: Any, den: Any) -> float:
    r = safe_rate(num, den)
    return float(r) if r is not None else 0.0


def normalize_option_sources(option_sources: Any) -> Optional[List[str]]:
    if not isinstance(option_sources, list) or len(option_sources) < 4:
        return None
    out: List[str] = []
    for src in option_sources[:4]:
        if src is None:
            return None
        s = str(src).strip()
        if not s:
            return None
        out.append(s)
    return out if len(out) == 4 else None


def map_pred_to_source(pred_letter: Any, option_sources: Any) -> Tuple[Optional[int], Optional[str]]:
    idx = letter_to_idx(pred_letter)
    sources = normalize_option_sources(option_sources)
    if idx is None or sources is None:
        return idx, None
    return idx, sources[idx]


def map_answer_to_source(
    answer_letter: Any,
    answer_idx: Any,
    option_sources: Any,
) -> Tuple[Optional[int], Optional[str]]:
    sources = normalize_option_sources(option_sources)
    idx: Optional[int] = None
    try:
        if answer_idx is not None:
            cand = int(answer_idx)
            if 0 <= cand < 4:
                idx = cand
    except Exception:
        idx = None
    if idx is None:
        idx = letter_to_idx(answer_letter)
    if idx is None or sources is None:
        return idx, None
    return idx, sources[idx]


def inc_count(d: Dict[str, int], key: Any, amount: int = 1) -> None:
    if key is None:
        return
    k = str(key)
    d[k] = int(d.get(k, 0)) + int(amount)


def init_source_stats() -> Dict[str, Any]:
    return {
        "total_count": 0,
        "valid_count": 0,
        "invalid_count": 0,
        "mapped_count": 0,
        "correct_mapped": 0,
        "wrong_count": 0,
        "selected_source_counts": {},
        "wrong_source_counts": {},
        "gt_source_counts": {},
        "option_source_missing_count": 0,
    }


def update_source_stats(
    stats: Dict[str, Any],
    pred_letter: Any,
    answer_letter: Any,
    answer_idx: Any,
    option_sources: Any,
    valid: bool,
) -> Dict[str, Any]:
    stats["total_count"] = int(stats.get("total_count", 0)) + 1
    if valid:
        stats["valid_count"] = int(stats.get("valid_count", 0)) + 1
    else:
        stats["invalid_count"] = int(stats.get("invalid_count", 0)) + 1

    sources = normalize_option_sources(option_sources)
    pred_idx, pred_source = map_pred_to_source(pred_letter, sources)
    gt_idx, gt_source = map_answer_to_source(answer_letter, answer_idx, sources)
    if sources is None:
        stats["option_source_missing_count"] = int(stats.get("option_source_missing_count", 0)) + 1

    wrong = bool(valid and pred_letter and answer_letter and str(pred_letter).strip().upper() != str(answer_letter).strip().upper())
    is_mapped = bool(valid and pred_source is not None)
    if is_mapped:
        stats["mapped_count"] = int(stats.get("mapped_count", 0)) + 1
        inc_count(stats.setdefault("selected_source_counts", {}), pred_source, 1)
        if gt_source is not None:
            inc_count(stats.setdefault("gt_source_counts", {}), gt_source, 1)
        if wrong:
            stats["wrong_count"] = int(stats.get("wrong_count", 0)) + 1
            inc_count(stats.setdefault("wrong_source_counts", {}), pred_source, 1)
        else:
            stats["correct_mapped"] = int(stats.get("correct_mapped", 0)) + 1

    is_disturb = bool(pred_source in STRONG_DISTURB_SOURCES) if pred_source is not None else None
    return {
        "pred_idx": pred_idx,
        "pred_source": pred_source,
        "gt_idx": gt_idx,
        "gt_source": gt_source,
        "is_disturb": is_disturb,
        "is_strong_distractor": is_disturb,
        "wrong": wrong,
        "source_mapped": is_mapped,
    }


def summarize_source_stats(
    *,
    total_count: int,
    valid_count: int,
    invalid_count: int,
    mapped_count: int,
    correct_mapped: int,
    selected_source_counts: Dict[str, int],
    wrong_source_counts: Optional[Dict[str, int]] = None,
    gt_source_counts: Optional[Dict[str, int]] = None,
    option_source_missing_count: int = 0,
) -> Dict[str, Any]:
    selected_counts = {str(k): int(v) for k, v in (selected_source_counts or {}).items()}
    wrong_counts = {str(k): int(v) for k, v in (wrong_source_counts or {}).items()}
    gt_counts = {str(k): int(v) for k, v in (gt_source_counts or {}).items()}

    strong_count = int(sum(selected_counts.get(src, 0) for src in STRONG_DISTURB_SOURCES))
    prev_count = int(selected_counts.get("prev", 0))
    future_count = int(selected_counts.get("future", 0))
    absurd_count = int(selected_counts.get("absurd", 0))
    gt_count = int(selected_counts.get("gt", 0))

    wrong_count = int(sum(wrong_counts.values())) if wrong_counts else int(max(0, mapped_count - correct_mapped))
    strong_wrong = int(sum(wrong_counts.get(src, 0) for src in STRONG_DISTURB_SOURCES))
    prev_wrong = int(wrong_counts.get("prev", 0))
    future_wrong = int(wrong_counts.get("future", 0))
    absurd_wrong = int(wrong_counts.get("absurd", 0))
    gt_wrong = int(wrong_counts.get("gt", 0))

    return {
        "total_count": int(total_count),
        "valid_count": int(valid_count),
        "invalid_count": int(invalid_count),
        "invalid_rate": safe_rate(invalid_count, total_count),
        "mapped_count": int(mapped_count),
        "valid_mapped_count": int(mapped_count),
        "wrong_count": int(wrong_count),
        "selected_source_counts": selected_counts,
        "pred_source_counts": selected_counts,
        "wrong_source_counts": wrong_counts,
        "gt_source_counts": gt_counts,
        "option_source_missing_count": int(option_source_missing_count),
        "selected_gt_rate": safe_rate(gt_count, mapped_count),
        "selected_prev_rate": safe_rate(prev_count, mapped_count),
        "selected_future_rate": safe_rate(future_count, mapped_count),
        "selected_absurd_rate": safe_rate(absurd_count, mapped_count),
        "selected_disturb_rate": safe_rate(strong_count, mapped_count),
        "selected_strong_distractor_rate": safe_rate(strong_count, mapped_count),
        "disturb_when_wrong": safe_rate(strong_wrong, wrong_count),
        "prev_when_wrong": safe_rate(prev_wrong, wrong_count),
        "future_when_wrong": safe_rate(future_wrong, wrong_count),
        "absurd_when_wrong": safe_rate(absurd_wrong, wrong_count),
        "gt_when_wrong": safe_rate(gt_wrong, wrong_count),
        "strong_distractor_sources": sorted(STRONG_DISTURB_SOURCES),
        "num_samples_mapped_to_option_sources": int(mapped_count),
        "answer_correct_mapped": int(correct_mapped),
        "wrong_mapped": int(wrong_count),
        "strong_distractor_choice_count": int(strong_count),
        "strong_distractor_choice_rate_over_mapped": _legacy_rate(strong_count, mapped_count),
        "strong_prev_choice_count": int(prev_count),
        "strong_future_choice_count": int(future_count),
        "strong_prev_choice_rate_over_mapped": _legacy_rate(prev_count, mapped_count),
        "strong_future_choice_rate_over_mapped": _legacy_rate(future_count, mapped_count),
        "strong_distractor_choice_rate_given_wrong_over_mapped": _legacy_rate(strong_wrong, wrong_count),
    }


def finalize_source_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    return summarize_source_stats(
        total_count=int(stats.get("total_count", 0)),
        valid_count=int(stats.get("valid_count", 0)),
        invalid_count=int(stats.get("invalid_count", 0)),
        mapped_count=int(stats.get("mapped_count", 0)),
        correct_mapped=int(stats.get("correct_mapped", 0)),
        selected_source_counts=stats.get("selected_source_counts", {}),
        wrong_source_counts=stats.get("wrong_source_counts", {}),
        gt_source_counts=stats.get("gt_source_counts", {}),
        option_source_missing_count=int(stats.get("option_source_missing_count", 0)),
    )


def _fmt_rate(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "n/a"


def source_stats_report_lines(summary: Dict[str, Any]) -> List[str]:
    return [
        f"Total / valid / invalid        : {summary.get('total_count')} / {summary.get('valid_count')} / {summary.get('invalid_count')}",
        f"Invalid rate                   : {_fmt_rate(summary.get('invalid_rate'))}",
        f"Mapped samples                 : {summary.get('mapped_count')}",
        f"Selected disturb rate          : {_fmt_rate(summary.get('selected_disturb_rate'))}",
        f"Selected prev rate             : {_fmt_rate(summary.get('selected_prev_rate'))}",
        f"Selected future rate           : {_fmt_rate(summary.get('selected_future_rate'))}",
        f"Selected absurd rate           : {_fmt_rate(summary.get('selected_absurd_rate'))}",
        f"Disturb rate | wrong(valid)    : {_fmt_rate(summary.get('disturb_when_wrong'))}",
        f"Prev rate | wrong(valid)       : {_fmt_rate(summary.get('prev_when_wrong'))}",
        f"Future rate | wrong(valid)     : {_fmt_rate(summary.get('future_when_wrong'))}",
        f"Absurd rate | wrong(valid)     : {_fmt_rate(summary.get('absurd_when_wrong'))}",
        f"Selected source counts         : {summary.get('selected_source_counts')}",
        f"Wrong source counts            : {summary.get('wrong_source_counts')}",
    ]


def _collect_queryset_candidates(obj: Any, key_path: str = "") -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kp = f"{key_path}.{k}" if key_path else str(k)
            if isinstance(v, str):
                kk = str(k).lower()
                vv = v.lower()
                if ("queryset" in kk) or ("queryset" in vv):
                    out.append((kp, v))
            out.extend(_collect_queryset_candidates(v, kp))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            kp = f"{key_path}[{i}]"
            out.extend(_collect_queryset_candidates(v, kp))
    return out


def resolve_queryset_path_from_pred(pred: Dict[str, Any]) -> Optional[str]:
    v = pred.get("source_queryset", None)
    if isinstance(v, str) and v.strip():
        p = Path(v).expanduser()
        if p.is_file():
            return str(p)

    vm = pred.get("video_metadata", {}) if isinstance(pred.get("video_metadata"), dict) else {}
    v2 = vm.get("queryset_path", None)
    if isinstance(v2, str) and v2.strip():
        p = Path(v2).expanduser()
        if p.is_file():
            return str(p)

    for _kp, vv in _collect_queryset_candidates(pred):
        if not isinstance(vv, str) or not vv.strip():
            continue
        p = Path(vv).expanduser()
        if p.is_file():
            return str(p)
    return None


def build_effective_mcq_by_idx(queryset: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    samples = queryset.get("samples", []) if isinstance(queryset, dict) else []
    if not isinstance(samples, list):
        return out
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        idx = sample.get("idx", None)
        mcq = sample.get("mcq", None)
        if isinstance(idx, (int, float)) and isinstance(mcq, dict):
            out[int(idx)] = mcq
    return out
