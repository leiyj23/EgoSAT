#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small deterministic metric helpers for EgoSAT evaluation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional


def safe_div(numer: Any, denom: Any) -> Optional[float]:
    """Return numer / denom, or None when denom is zero or invalid."""
    try:
        n = float(numer)
        d = float(denom)
    except Exception:
        return None
    if d == 0.0:
        return None
    return float(n / d)


def mean_valid(values: Iterable[Any]) -> Optional[float]:
    xs: List[float] = []
    for value in values:
        if value is None:
            continue
        try:
            v = float(value)
        except Exception:
            continue
        xs.append(v)
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def accuracy_over_total(correct: Any, total: Any) -> Optional[float]:
    return safe_div(correct, total)


def precision(tp: Any, fp: Any) -> Optional[float]:
    try:
        return safe_div(int(tp), int(tp) + int(fp))
    except Exception:
        return None


def recall(tp: Any, fn: Any) -> Optional[float]:
    try:
        return safe_div(int(tp), int(tp) + int(fn))
    except Exception:
        return None


def ratio_with_counts(numer: int, denom: int, *, key: str = "value") -> Dict[str, Any]:
    return {
        key: safe_div(numer, denom),
        "numerator": int(numer),
        "denominator": int(denom),
    }


def flatten_dict_for_csv(
    data: Mapping[str, Any],
    *,
    parent_key: str = "",
    sep: str = "_",
) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_dict_for_csv(value, parent_key=new_key, sep=sep))
        elif isinstance(value, list):
            flat[new_key] = ";".join(str(x) for x in value)
        else:
            flat[new_key] = value
    return flat


def format_metric(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)

