#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Official minimal EgoSAT main-table evaluator.

This script reads raw per-GT prediction JSON files, not normalized JSONL.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.io import (  # noqa: E402
    index_samples_by_key,
    load_json,
    load_json_or_jsonl,
    parse_path_maps,
    resolve_queryset_for_prediction,
    scan_prediction_jsons,
    write_csv,
    write_json,
    write_jsonl,
    write_markdown_table,
)
from evaluation.metrics import accuracy_over_total, mean_valid, precision, recall  # noqa: E402

try:  # noqa: E402
    from evaluation.legacy.now_state_switch_score import (
        cand_state_from_clean_text,
        gt_state_from_region as _legacy_gt_state_from_region,
        normalize_state as _legacy_normalize_state,
        parse_tag as _legacy_parse_state_tag,
        required_roles_for_pair_type,
        state_token_prob_proxy,
    )
except Exception:  # pragma: no cover - fallback is for unusual import contexts.
    _STATE_TAG_RE = re.compile(r"<\s*STATE\s*>(.*?)<\s*/\s*STATE\s*>", re.IGNORECASE | re.DOTALL)

    def _legacy_parse_state_tag(text: str, which: str) -> Optional[str]:
        if which.lower() != "state":
            return None
        m = _STATE_TAG_RE.search(text or "")
        return (m.group(1) or "").strip() if m else None

    def _legacy_normalize_state(x: Optional[str]) -> str:
        s = (x or "").strip().lower().replace(" ", "_")
        if "no_interaction" in s:
            return "NO_INTERACTION"
        if "interaction" in s:
            return "INTERACTION"
        if s in {"no", "none", "0"}:
            return "NO_INTERACTION"
        if s in {"yes", "1"}:
            return "INTERACTION"
        return "UNKNOWN"

    def cand_state_from_clean_text(clean_text: str) -> Tuple[str, str]:
        s = clean_text or ""
        has_no = re.search(r"\bno[_ ]interaction\b", s, flags=re.IGNORECASE) is not None
        has_inter = re.search(r"(?<!no[_ ])\binteraction\b", s, flags=re.IGNORECASE) is not None
        if has_no and has_inter:
            return "INVALID", "clean_ambiguous"
        if has_no:
            return "NO_INTERACTION", "clean_no_interaction"
        if has_inter:
            return "INTERACTION", "clean_interaction"
        return "INVALID", "clean_none"

    def _legacy_gt_state_from_region(sample: Dict[str, Any]) -> str:
        region = str(sample.get("region", "") or "").strip().lower()
        if region == "segment":
            return "INTERACTION"
        if region == "gap":
            return "NO_INTERACTION"
        return "UNKNOWN"

    def required_roles_for_pair_type(pair_type: str) -> Optional[Tuple[str, str]]:
        pt = (pair_type or "").strip().lower()
        if pt == "segment_to_gap":
            return ("fixed_segment_mid", "scan_gap")
        if pt == "gap_to_segment":
            return ("fixed_gap_mid", "scan_segment")
        return None

    def state_token_prob_proxy(ps: Dict[str, Any]) -> Optional[float]:
        probs = ps.get("gen_token_probs")
        if isinstance(probs, list) and probs:
            try:
                return float(probs[0])
            except Exception:
                return None
        return None


LETTERS = ("A", "B", "C", "D")
SH_PRED_GROUPS = {
    "NN": "predictable",
    "PN": "branch_only",
    "NP": "surprise_only",
    "PP": "branch_and_surprise",
}
SS_POS_LABELS = {
    "ss_pos_1": "t:1.0",
    "ss_pos_2": "t:2.0",
    "ss_pos_3": "t:4.0",
}


class EvalError(RuntimeError):
    pass


def _clamp01(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None
    if v < 0.0:
        v = 0.0
    if v > 1.0:
        v = 1.0
    return float(v)


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]*>", " ", text or "")


def _parse_tag(text: str, tag: str) -> Optional[str]:
    rx = re.compile(r"<\s*" + re.escape(tag) + r"\s*>(.*?)<\s*/\s*" + re.escape(tag) + r"\s*>", re.IGNORECASE | re.DOTALL)
    m = rx.search(text or "")
    return (m.group(1) or "").strip() if m else None


def normalize_ans_letter(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip().upper()
    if re.fullmatch(r"[A-D]", s):
        return s
    hits = re.findall(r"\b([A-D])\b", s)
    uniq = []
    for hit in hits:
        if hit not in uniq:
            uniq.append(hit)
    return uniq[0] if len(uniq) == 1 else ""


def _guess_ans_from_text(text: str) -> str:
    t = _strip_tags(text)
    if not t.strip():
        return ""
    hits: List[str] = []
    hits += re.findall(r"(?:^|\n)\s*([A-D])\s*[\.\)]\s*", t, flags=re.IGNORECASE)
    hits += re.findall(r"\b(?:CHOOSE|ANSWER|ANS|OPTION)\s*[:=\-]?\s*([A-D])\b", t, flags=re.IGNORECASE)
    hits += re.findall(r"\b([A-D])\b", t, flags=re.IGNORECASE)
    starters = set(h.upper() for h in re.findall(r"(?:^|\n)\s*([A-D])\s*[\.\)\|]\s*", t, flags=re.IGNORECASE))
    if len(starters) >= 2:
        return ""
    uniq: List[str] = []
    for hit in hits:
        h = hit.upper()
        if h in LETTERS and h not in uniq:
            uniq.append(h)
    return uniq[0] if len(uniq) == 1 else ""


def extract_pred_ans(sample: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(sample, Mapping):
        return ""
    parsed = sample.get("parsed")
    if isinstance(parsed, Mapping):
        for key in ("ans", "answer", "ANS", "ANSWER"):
            ans = normalize_ans_letter(parsed.get(key))
            if ans:
                return ans

    diag = sample.get("diagnostics")
    if isinstance(diag, Mapping):
        ped = diag.get("prompt_and_encoding_debug")
        if isinstance(ped, Mapping):
            for key in ("cand_conf_final_letter", "chosen_by_p_cond"):
                ans = normalize_ans_letter(ped.get(key))
                if ans:
                    return ans
            probe = ped.get("cand_conf_probe")
            if isinstance(probe, Mapping):
                ans = normalize_ans_letter(probe.get("chosen_by_p_cond"))
                if ans:
                    return ans

    for field in ("response_text", "clean_response", "clean", "raw_response"):
        text = sample.get(field)
        if isinstance(text, str) and text.strip():
            tagged = normalize_ans_letter(_parse_tag(text, "ANS"))
            if tagged:
                return tagged
            guessed = _guess_ans_from_text(text)
            if guessed:
                return guessed

    toks = sample.get("gen_tokens")
    if isinstance(toks, list):
        joined = " ".join(str(x) for x in toks[:20])
        tagged = normalize_ans_letter(_parse_tag(joined, "ANS"))
        if tagged:
            return tagged
        guessed = _guess_ans_from_text(joined)
        if guessed:
            return guessed
    return ""


def answer_letter_from_gt(sample: Mapping[str, Any]) -> str:
    mcq = sample.get("mcq") if isinstance(sample.get("mcq"), Mapping) else {}
    for obj in (mcq, sample):
        ans = normalize_ans_letter(obj.get("answer") if isinstance(obj, Mapping) else None)
        if ans:
            return ans
        ans = normalize_ans_letter(obj.get("ans") if isinstance(obj, Mapping) else None)
        if ans:
            return ans
        idx = obj.get("answer_idx") if isinstance(obj, Mapping) else None
        try:
            i = int(idx)
        except Exception:
            i = -1
        if 0 <= i < 4:
            return LETTERS[i]
    return ""


def extract_cand_conf(sample: Optional[Mapping[str, Any]]) -> Tuple[Optional[float], str]:
    if not isinstance(sample, Mapping):
        return None, "missing_prediction"
    pred_ans = extract_pred_ans(sample)
    diag = sample.get("diagnostics")
    if isinstance(diag, Mapping):
        ped = diag.get("prompt_and_encoding_debug")
        if isinstance(ped, Mapping):
            probe = ped.get("cand_conf_probe")
            if isinstance(probe, Mapping):
                p_cond = probe.get("p_cond")
                if isinstance(p_cond, Mapping):
                    if pred_ans and pred_ans in p_cond:
                        v = _clamp01(p_cond.get(pred_ans))
                        if v is not None:
                            return v, "cand_conf_p_cond_pred"
                    vals = [_clamp01(v) for v in p_cond.values()]
                    vals = [v for v in vals if v is not None]
                    if vals:
                        return max(vals), "cand_conf_p_cond_max"
    probs = sample.get("gen_token_probs")
    if isinstance(probs, list) and probs:
        v = _clamp01(probs[0])
        if v is not None:
            return v, "gen_token_prob0"
    return None, "none"


def extract_pred_state(sample: Optional[Mapping[str, Any]]) -> Tuple[str, str]:
    if not isinstance(sample, Mapping):
        return "INVALID", "missing_prediction"
    response_text = str(sample.get("response_text", "") or "")
    clean_text = str(sample.get("clean_response", "") or sample.get("clean", "") or response_text or "")
    raw = _legacy_parse_state_tag(response_text, "state") or _legacy_parse_state_tag(clean_text, "state")
    if raw is not None:
        state = _legacy_normalize_state(raw)
        if state in {"INTERACTION", "NO_INTERACTION"}:
            return state, "tag"
    return cand_state_from_clean_text(clean_text)


def gt_state_from_sample(sample: Mapping[str, Any]) -> str:
    state = _legacy_gt_state_from_region(dict(sample))
    if state in {"INTERACTION", "NO_INTERACTION"}:
        return state
    for key in ("visible_interaction", "interaction_visible", "is_interaction"):
        value = sample.get(key)
        if isinstance(value, bool):
            return "INTERACTION" if value else "NO_INTERACTION"
        if isinstance(value, (int, float)):
            return "INTERACTION" if int(value) == 1 else "NO_INTERACTION"
    raw = sample.get("state") or sample.get("gt_state")
    norm = _legacy_normalize_state(str(raw)) if raw is not None else "UNKNOWN"
    return norm if norm in {"INTERACTION", "NO_INTERACTION"} else "UNKNOWN"


def sample_key(sample: Mapping[str, Any], pos: int) -> Any:
    key = sample.get("idx", None)
    if isinstance(key, float) and key.is_integer():
        return int(key)
    if key is not None:
        return key
    return sample.get("sample_id") or sample.get("id") or pos


def step_idx_from_sample(sample: Mapping[str, Any], *, task: str) -> Optional[int]:
    raw = sample.get("step_idx")
    try:
        step = int(raw)
    except Exception:
        step = None
    if step in {0, 1, 2}:
        return step
    key = "lead_sec" if task == "ms_pred" else "lag_sec"
    value = sample.get(key)
    try:
        sec = float(value)
    except Exception:
        return None
    for idx, ref in enumerate((8.0, 16.0, 24.0)):
        if abs(sec - ref) < 1e-3:
            return idx
    return None


def sh_pred_group(sample: Mapping[str, Any]) -> str:
    state = str(sample.get("state", "") or "").strip().upper()
    return SH_PRED_GROUPS.get(state, "unknown")


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _roots(args: argparse.Namespace) -> List[Path]:
    roots: List[Path] = [args.pred_root, args.gt_root, args.mcq_root]
    if args.effective_queryset_root is not None:
        roots.insert(0, args.effective_queryset_root)
    out: List[Path] = []
    seen = set()
    for root in roots:
        if root is None:
            continue
        r = Path(root).expanduser()
        key = str(r)
        if key not in seen:
            out.append(r)
            seen.add(key)
    return out


def iter_prediction_querysets(
    pred_dir: Path,
    *,
    roots: Sequence[Path],
    path_maps: Sequence[Tuple[str, str]],
    warnings: List[str],
) -> Iterable[Tuple[Path, Dict[str, Any], Path, Dict[str, Any], List[Dict[str, Any]], Dict[Any, Dict[str, Any]]]]:
    pred_files = scan_prediction_jsons(pred_dir)
    if not pred_files:
        raise EvalError(f"no prediction JSON files found under {pred_dir}")
    seen_qpaths = set()
    for pred_path in pred_files:
        pred = load_json(pred_path)
        if not isinstance(pred, dict):
            warnings.append(f"skip non-dict prediction JSON: {pred_path}")
            continue
        qpath = resolve_queryset_for_prediction(pred, roots=roots, path_maps=path_maps)
        if qpath is None:
            raise EvalError(f"cannot resolve source/effective queryset for prediction: {pred_path}")
        qkey = str(qpath.resolve())
        if qkey in seen_qpaths:
            warnings.append(f"skip duplicate effective queryset reference: {qpath}")
            continue
        seen_qpaths.add(qkey)
        qobj = load_json_or_jsonl(qpath)
        if isinstance(qobj, list) and len(qobj) == 1 and isinstance(qobj[0], dict):
            qobj = qobj[0]
        if not isinstance(qobj, dict):
            raise EvalError(f"effective queryset is not a JSON object: {qpath}")
        qsamples_raw = qobj.get("samples", [])
        if not isinstance(qsamples_raw, list):
            raise EvalError(f"effective queryset samples is not a list: {qpath}")
        qsamples = [s for s in qsamples_raw if isinstance(s, dict)]
        psamples = pred.get("samples", [])
        if not isinstance(psamples, list):
            psamples = []
        pred_by_key = index_samples_by_key(psamples)
        yield pred_path, pred, qpath, qobj, qsamples, pred_by_key


def require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise EvalError(f"required prediction directory missing for {label}: {path}")
    return path


def score_now_state(pred_dir: Path, args: argparse.Namespace, warnings: List[str], details_dir: Path) -> Dict[str, Any]:
    counts = {
        "samples_total": 0,
        "valid_predictions": 0,
        "invalid_or_missing_predictions": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "state_correct_full": 0,
        "unknown_gt": 0,
    }
    details: List[Dict[str, Any]] = []
    for pred_path, _pred, qpath, _qobj, qsamples, pred_by_key in iter_prediction_querysets(
        pred_dir, roots=_roots(args), path_maps=args.path_maps, warnings=warnings
    ):
        for pos, gt in enumerate(qsamples):
            key = sample_key(gt, pos)
            ps = pred_by_key.get(key)
            gt_state = gt_state_from_sample(gt)
            if gt_state not in {"INTERACTION", "NO_INTERACTION"}:
                counts["unknown_gt"] += 1
                continue
            pred_state, pred_method = extract_pred_state(ps)
            pred_valid = pred_state in {"INTERACTION", "NO_INTERACTION"}
            gt_pos = gt_state == "INTERACTION"
            pred_pos = pred_state == "INTERACTION"
            counts["samples_total"] += 1
            if pred_valid:
                counts["valid_predictions"] += 1
            else:
                counts["invalid_or_missing_predictions"] += 1
            if gt_pos and pred_pos:
                counts["tp"] += 1
            elif (not gt_pos) and pred_pos:
                counts["fp"] += 1
            elif gt_pos and (not pred_pos):
                counts["fn"] += 1
            else:
                counts["tn"] += 1
            if pred_valid and pred_state == gt_state:
                counts["state_correct_full"] += 1
            if args.write_details:
                details.append(
                    {
                        "pred_file": str(pred_path),
                        "queryset_path": str(qpath),
                        "idx": key,
                        "gt_state": gt_state,
                        "pred_state": pred_state,
                        "pred_method": pred_method,
                        "valid": pred_valid,
                    }
                )
    out = {
        "rec": recall(counts["tp"], counts["fn"]),
        "prec": precision(counts["tp"], counts["fp"]),
        "state_accuracy_full": accuracy_over_total(counts["state_correct_full"], counts["samples_total"]),
        "counts": counts,
    }
    write_json(details_dir / "summary.json", out)
    if args.write_details:
        write_jsonl(details_dir / "details.jsonl", details)
    return out


def score_mcq_accuracy(
    pred_dir: Path,
    args: argparse.Namespace,
    warnings: List[str],
    details_dir: Path,
    *,
    bucket_fn: Optional[Callable[[Mapping[str, Any]], str]] = None,
    step_task: Optional[str] = None,
) -> Dict[str, Any]:
    buckets: Dict[str, Dict[str, int]] = {}
    details: List[Dict[str, Any]] = []

    def bucket_for(sample: Mapping[str, Any]) -> str:
        if step_task:
            step = step_idx_from_sample(sample, task=step_task)
            return f"step_{step}" if step is not None else "unknown"
        if bucket_fn is not None:
            return bucket_fn(sample)
        return "all"

    for pred_path, _pred, qpath, _qobj, qsamples, pred_by_key in iter_prediction_querysets(
        pred_dir, roots=_roots(args), path_maps=args.path_maps, warnings=warnings
    ):
        for pos, gt in enumerate(qsamples):
            key = sample_key(gt, pos)
            gt_ans = answer_letter_from_gt(gt)
            if not gt_ans:
                warnings.append(f"missing GT answer in {qpath} idx={key}; sample skipped")
                continue
            bucket = bucket_for(gt)
            b = buckets.setdefault(bucket, {"samples_total": 0, "samples_valid": 0, "samples_invalid": 0, "correct": 0})
            ps = pred_by_key.get(key)
            pred_ans = extract_pred_ans(ps)
            valid = bool(pred_ans)
            correct = bool(valid and pred_ans == gt_ans)
            b["samples_total"] += 1
            if valid:
                b["samples_valid"] += 1
            else:
                b["samples_invalid"] += 1
            if correct:
                b["correct"] += 1
            if args.write_details:
                details.append(
                    {
                        "pred_file": str(pred_path),
                        "queryset_path": str(qpath),
                        "idx": key,
                        "bucket": bucket,
                        "gt_ans": gt_ans,
                        "pred_ans": pred_ans or None,
                        "valid": valid,
                        "correct": correct,
                    }
                )

    summary: Dict[str, Any] = {"buckets": {}}
    for name, b in buckets.items():
        summary["buckets"][name] = {
            **b,
            "accuracy_over_total": accuracy_over_total(b["correct"], b["samples_total"]),
            "accuracy_over_valid": accuracy_over_total(b["correct"], b["samples_valid"]),
        }
    all_counts = {"samples_total": 0, "samples_valid": 0, "samples_invalid": 0, "correct": 0}
    for b in buckets.values():
        for key in all_counts:
            all_counts[key] += int(b.get(key, 0))
    summary["all"] = {
        **all_counts,
        "accuracy_over_total": accuracy_over_total(all_counts["correct"], all_counts["samples_total"]),
        "accuracy_over_valid": accuracy_over_total(all_counts["correct"], all_counts["samples_valid"]),
    }
    write_json(details_dir / "summary.json", summary)
    if args.write_details:
        write_jsonl(details_dir / "details.jsonl", details)
    return summary


def score_confidence(
    pred_dir: Path,
    args: argparse.Namespace,
    warnings: List[str],
    details_dir: Path,
    *,
    bucket_fn: Optional[Callable[[Mapping[str, Any]], str]] = None,
    step_task: Optional[str] = None,
) -> Dict[str, Any]:
    buckets: Dict[str, Dict[str, Any]] = {}
    details: List[Dict[str, Any]] = []

    def bucket_for(sample: Mapping[str, Any]) -> str:
        if step_task:
            step = step_idx_from_sample(sample, task=step_task)
            return f"step_{step}" if step is not None else "unknown"
        if bucket_fn is not None:
            return bucket_fn(sample)
        return "all"

    for pred_path, _pred, qpath, _qobj, qsamples, pred_by_key in iter_prediction_querysets(
        pred_dir, roots=_roots(args), path_maps=args.path_maps, warnings=warnings
    ):
        for pos, gt in enumerate(qsamples):
            key = sample_key(gt, pos)
            bucket = bucket_for(gt)
            b = buckets.setdefault(bucket, {"samples_total": 0, "conf_values": [], "conf_sources": {}})
            ps = pred_by_key.get(key)
            conf, source = extract_cand_conf(ps)
            b["samples_total"] += 1
            b["conf_sources"][source] = int(b["conf_sources"].get(source, 0)) + 1
            if conf is not None:
                b["conf_values"].append(float(conf))
            if args.write_details:
                details.append(
                    {
                        "pred_file": str(pred_path),
                        "queryset_path": str(qpath),
                        "idx": key,
                        "bucket": bucket,
                        "conf": conf,
                        "conf_source": source,
                    }
                )
    summary: Dict[str, Any] = {"buckets": {}}
    for name, b in buckets.items():
        values = [float(x) for x in b["conf_values"]]
        summary["buckets"][name] = {
            "samples_total": int(b["samples_total"]),
            "conf_count_over_valid": int(len(values)),
            "conf_mean_over_valid": mean_valid(values),
            "conf_sources": dict(b["conf_sources"]),
        }
    write_json(details_dir / "summary.json", summary)
    if args.write_details:
        write_jsonl(details_dir / "details.jsonl", details)
    return summary


def score_state_switch(pred_dir: Path, args: argparse.Namespace, warnings: List[str], details_dir: Path) -> Dict[str, Any]:
    pairs: Dict[str, Dict[str, Any]] = {}
    details: List[Dict[str, Any]] = []
    sample_counts = {"samples_total": 0, "valid_predictions": 0, "invalid_or_missing_predictions": 0, "unknown_gt": 0}

    for pred_path, _pred, qpath, _qobj, qsamples, pred_by_key in iter_prediction_querysets(
        pred_dir, roots=_roots(args), path_maps=args.path_maps, warnings=warnings
    ):
        for pos, gt in enumerate(qsamples):
            key = sample_key(gt, pos)
            gt_state = gt_state_from_sample(gt)
            if gt_state not in {"INTERACTION", "NO_INTERACTION"}:
                sample_counts["unknown_gt"] += 1
                continue
            ps = pred_by_key.get(key)
            pred_state, pred_method = extract_pred_state(ps)
            valid = pred_state in {"INTERACTION", "NO_INTERACTION"}
            conf = state_token_prob_proxy(dict(ps)) if isinstance(ps, Mapping) else None
            sample_counts["samples_total"] += 1
            if valid:
                sample_counts["valid_predictions"] += 1
            else:
                sample_counts["invalid_or_missing_predictions"] += 1

            pair_type = str(gt.get("state_switch_pair_type", "") or "").strip().lower()
            role = str(gt.get("state_switch_role", "") or "").strip().lower()
            raw_pair_id = str(gt.get("state_switch_pair_id", "") or "").strip()
            if raw_pair_id:
                pair_id = f"{qpath.as_posix()}::{raw_pair_id}"
                rec = pairs.setdefault(pair_id, {"pair_type": pair_type, "roles": {}})
                rec["pair_type"] = rec.get("pair_type") or pair_type
                rec["roles"][role] = {
                    "idx": key,
                    "gt_state": gt_state,
                    "pred_state": pred_state,
                    "valid": valid,
                    "conf": conf,
                }
            if args.write_details:
                details.append(
                    {
                        "pred_file": str(pred_path),
                        "queryset_path": str(qpath),
                        "idx": key,
                        "pair_id": raw_pair_id or None,
                        "pair_type": pair_type or None,
                        "role": role or None,
                        "gt_state": gt_state,
                        "pred_state": pred_state,
                        "pred_method": pred_method,
                        "valid": valid,
                        "conf_proxy_tokenprob_first_state_token": conf,
                    }
                )

    direction_counts = {
        "FG_to_BG": {
            "pair_type": "segment_to_gap",
            "pairs_total_seen": 0,
            "pairs_missing_required_roles": 0,
            "pairs_invalid_due_to_invalid_probe": 0,
            "pairs_valid": 0,
            "pairs_success_both_probes_correct": 0,
            "after_conf_success_values": [],
        },
        "BG_to_FG": {
            "pair_type": "gap_to_segment",
            "pairs_total_seen": 0,
            "pairs_missing_required_roles": 0,
            "pairs_invalid_due_to_invalid_probe": 0,
            "pairs_valid": 0,
            "pairs_success_both_probes_correct": 0,
            "after_conf_success_values": [],
        },
    }

    for rec in pairs.values():
        pair_type = str(rec.get("pair_type", "") or "").strip().lower()
        req = required_roles_for_pair_type(pair_type)
        if req is None:
            continue
        direction = "FG_to_BG" if pair_type == "segment_to_gap" else "BG_to_FG"
        dc = direction_counts[direction]
        dc["pairs_total_seen"] += 1
        roles = rec.get("roles") if isinstance(rec.get("roles"), Mapping) else {}
        before_role, after_role = req
        if before_role not in roles or after_role not in roles:
            dc["pairs_missing_required_roles"] += 1
            continue
        before = roles[before_role]
        after = roles[after_role]
        if not (before.get("valid") and after.get("valid")):
            dc["pairs_invalid_due_to_invalid_probe"] += 1
            continue
        dc["pairs_valid"] += 1
        ok = before.get("pred_state") == before.get("gt_state") and after.get("pred_state") == after.get("gt_state")
        if ok:
            dc["pairs_success_both_probes_correct"] += 1
            if after.get("conf") is not None:
                dc["after_conf_success_values"].append(float(after["conf"]))

    switch_summary: Dict[str, Any] = {"samples": sample_counts, "switch": {}}
    for direction, dc in direction_counts.items():
        valid = int(dc["pairs_valid"])
        total = int(dc["pairs_total_seen"])
        success = int(dc["pairs_success_both_probes_correct"])
        values = [float(x) for x in dc["after_conf_success_values"]]
        switch_summary["switch"][direction] = {
            "pair_type": dc["pair_type"],
            "pairs_total_seen": total,
            "pairs_missing_required_roles": int(dc["pairs_missing_required_roles"]),
            "pairs_invalid_due_to_invalid_probe": int(dc["pairs_invalid_due_to_invalid_probe"]),
            "pairs_valid": valid,
            "pairs_success_both_probes_correct": success,
            "success_rate_over_valid_pairs": accuracy_over_total(success, valid),
            "success_rate_over_total_pairs": accuracy_over_total(success, total),
            "after_transition_conf_mean_over_success": mean_valid(values),
            "after_transition_conf_count_over_success": len(values),
        }
    write_json(details_dir / "summary.json", switch_summary)
    if args.write_details:
        write_jsonl(details_dir / "details.jsonl", details)
    return switch_summary


def parse_ss_pos(items: Optional[Sequence[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise EvalError(f"--ss-pos must be KEY=DIR, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key not in SS_POS_LABELS:
            raise EvalError(f"unknown --ss-pos key {key}; expected one of {sorted(SS_POS_LABELS)}")
        if not value:
            raise EvalError(f"empty directory value for --ss-pos {key}")
        out[key] = value
    return out


def sanitize_ss_pos(pos: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", pos or "").strip("_")
    while "__" in s:
        s = s.replace("__", "_")
    return s or "empty"


def find_ss_pos_dir(base: Path, key: str, explicit: Mapping[str, str]) -> Path:
    if key in explicit:
        value = explicit[key]
        p = Path(value).expanduser()
        if p.is_absolute() and p.is_dir():
            return p
        p2 = base / value
        if p2.is_dir():
            return p2
        raise EvalError(f"--ss-pos {key} points to missing directory: {value}")
    pos = SS_POS_LABELS[key]
    sanitized = sanitize_ss_pos(pos)
    variants = [
        f"sspos_{sanitized}",
        f"sspos_{sanitized.replace('.', '_')}",
        f"sspos_{sanitized.replace('.', 'p')}",
        sanitized,
        sanitized.replace(".", "_"),
        sanitized.replace(".", "p"),
    ]
    for name in variants:
        p = base / name
        if p.is_dir():
            return p
    existing = [p.name for p in sorted(base.iterdir()) if p.is_dir()] if base.is_dir() else []
    raise EvalError(f"cannot auto-resolve {key} ({pos}) under {base}; existing dirs={existing}")


def confidence_missing_result(warnings: List[str], label: str, path: Path, allow: bool) -> Optional[Dict[str, Any]]:
    if path.is_dir():
        return None
    msg = f"missing confidence prediction directory for {label}: {path}"
    if allow:
        warnings.append(msg)
        return {"buckets": {}}
    raise EvalError(msg + " (use --allow-missing-confidence to write null confidence metrics)")


def flatten_main_row(model: str, split: str, metrics: Mapping[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"model": model, "split": split}
    now = metrics.get("now_narration", {})
    row.update(
        {
            "now_rec": now.get("rec"),
            "now_prec": now.get("prec"),
            "now_mcq_acc": now.get("mcq_acc"),
        }
    )
    ss = metrics.get("now_state_switch", {})
    for i, key in enumerate(("ss_pos_1", "ss_pos_2", "ss_pos_3"), 1):
        block = ss.get(key, {})
        row[f"ss{i}_fg2bg_acc"] = block.get("fg_to_bg_acc")
        row[f"ss{i}_bg2fg_acc"] = block.get("bg_to_fg_acc")
        row[f"ss{i}_fg2bg_conf"] = block.get("fg_to_bg_conf_success_after_mean")
        row[f"ss{i}_bg2fg_conf"] = block.get("bg_to_fg_conf_success_after_mean")
    sh = metrics.get("sh_pred", {})
    for name in ("predictable", "branch_only", "surprise_only", "branch_and_surprise"):
        row[f"sh_pred_{name}_acc"] = sh.get(f"{name}_acc")
    for name in ("predictable", "branch_only", "surprise_only", "branch_and_surprise"):
        row[f"sh_pred_{name}_conf"] = sh.get(f"{name}_conf")
    for task in ("ms_pred", "ms_rtrv"):
        block = metrics.get(task, {})
        for step in range(3):
            row[f"{task}_step{step}_acc"] = block.get(f"step_{step}_acc")
        row[f"{task}_multi_avg_acc"] = block.get("multi_avg_acc")
        for step in range(3):
            row[f"{task}_step{step}_conf"] = block.get(f"step_{step}_conf")
        row[f"{task}_multi_avg_conf"] = block.get("multi_avg_conf")
    row["sh_rtrv_acc"] = metrics.get("sh_rtrv", {}).get("acc")
    return row


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    warnings: List[str] = []
    out_dir: Path = args.out_dir
    details_root = out_dir / "task_details"
    _mkdir(details_root)

    pred_root: Path = args.pred_root
    path_maps = parse_path_maps(args.path_map)
    args.path_maps = path_maps

    for label, root in (("pred-root", args.pred_root), ("gt-root", args.gt_root), ("mcq-root", args.mcq_root)):
        if not root.is_dir():
            raise EvalError(f"--{label} does not exist or is not a directory: {root}")
    if args.effective_queryset_root is not None and not args.effective_queryset_root.is_dir():
        raise EvalError(f"--effective-queryset-root does not exist: {args.effective_queryset_root}")

    metrics: Dict[str, Any] = {}

    now_state_summary = score_now_state(
        require_dir(pred_root / "now_narration" / "cand_state", "now_narration/cand_state"),
        args,
        warnings,
        details_root / "now_narration_cand_state",
    )
    now_mcq_summary = score_mcq_accuracy(
        require_dir(pred_root / "now_narration" / "cand_mcq", "now_narration/cand_mcq"),
        args,
        warnings,
        details_root / "now_narration_cand_mcq",
    )
    metrics["now_narration"] = {
        "rec": now_state_summary["rec"],
        "prec": now_state_summary["prec"],
        "mcq_acc": now_mcq_summary["all"]["accuracy_over_total"],
        "state_accuracy_full": now_state_summary["state_accuracy_full"],
        "counts": {
            "state": now_state_summary["counts"],
            "mcq": now_mcq_summary["all"],
        },
    }

    ss_base = require_dir(pred_root / "now_state_switch" / "cand_state", "now_state_switch/cand_state")
    explicit_ss = parse_ss_pos(args.ss_pos)
    ss_metrics: Dict[str, Any] = {}
    for key, pos in SS_POS_LABELS.items():
        ss_dir = find_ss_pos_dir(ss_base, key, explicit_ss)
        ss_summary = score_state_switch(ss_dir, args, warnings, details_root / f"now_state_switch_{key}")
        fg = ss_summary["switch"]["FG_to_BG"]
        bg = ss_summary["switch"]["BG_to_FG"]
        ss_metrics[key] = {
            "pos": pos,
            "pred_dir": str(ss_dir),
            "fg_to_bg_acc": fg["success_rate_over_valid_pairs"],
            "bg_to_fg_acc": bg["success_rate_over_valid_pairs"],
            "fg_to_bg_conf_success_after_mean": fg["after_transition_conf_mean_over_success"],
            "bg_to_fg_conf_success_after_mean": bg["after_transition_conf_mean_over_success"],
            "counts": {
                "samples": ss_summary["samples"],
                "FG_to_BG": fg,
                "BG_to_FG": bg,
            },
        }
    metrics["now_state_switch"] = ss_metrics

    sh_acc = score_mcq_accuracy(
        require_dir(pred_root / "sh_pred" / "cand_full", "sh_pred/cand_full"),
        args,
        warnings,
        details_root / "sh_pred_cand_full",
        bucket_fn=sh_pred_group,
    )
    sh_conf_dir = pred_root / "sh_pred" / "cand_conf"
    sh_conf_missing = confidence_missing_result(warnings, "sh_pred/cand_conf", sh_conf_dir, args.allow_missing_confidence)
    sh_conf = sh_conf_missing or score_confidence(
        sh_conf_dir,
        args,
        warnings,
        details_root / "sh_pred_cand_conf",
        bucket_fn=sh_pred_group,
    )
    sh_metrics: Dict[str, Any] = {"counts": {"accuracy": sh_acc, "confidence": sh_conf}}
    for name in ("predictable", "branch_only", "surprise_only", "branch_and_surprise"):
        sh_metrics[f"{name}_acc"] = sh_acc["buckets"].get(name, {}).get("accuracy_over_total")
        sh_metrics[f"{name}_conf"] = sh_conf["buckets"].get(name, {}).get("conf_mean_over_valid")
    metrics["sh_pred"] = sh_metrics

    for task in ("ms_pred", "ms_rtrv"):
        acc = score_mcq_accuracy(
            require_dir(pred_root / task / "cand", f"{task}/cand"),
            args,
            warnings,
            details_root / f"{task}_cand",
            step_task=task,
        )
        conf_dir = pred_root / task / "cand_conf"
        conf_missing = confidence_missing_result(warnings, f"{task}/cand_conf", conf_dir, args.allow_missing_confidence)
        conf = conf_missing or score_confidence(
            conf_dir,
            args,
            warnings,
            details_root / f"{task}_cand_conf",
            step_task=task,
        )
        block: Dict[str, Any] = {"counts": {"accuracy": acc, "confidence": conf}}
        acc_values: List[Optional[float]] = []
        conf_values: List[Optional[float]] = []
        for step in range(3):
            key = f"step_{step}"
            a = acc["buckets"].get(key, {}).get("accuracy_over_total")
            c = conf["buckets"].get(key, {}).get("conf_mean_over_valid")
            block[f"step_{step}_acc"] = a
            block[f"step_{step}_conf"] = c
            acc_values.append(a)
            conf_values.append(c)
        block["multi_avg_acc"] = mean_valid(acc_values)
        block["multi_avg_conf"] = mean_valid(conf_values)
        metrics[task] = block

    sh_rtrv = score_mcq_accuracy(
        require_dir(pred_root / "sh_rtrv" / "cand", "sh_rtrv/cand"),
        args,
        warnings,
        details_root / "sh_rtrv_cand",
    )
    metrics["sh_rtrv"] = {
        "acc": sh_rtrv["all"]["accuracy_over_total"],
        "counts": sh_rtrv,
    }

    result = {
        "model": args.model,
        "split": args.split,
        "pred_root": str(args.pred_root),
        "created_at_unix": time.time(),
        "metrics": metrics,
        "warnings": warnings,
    }

    _mkdir(out_dir)
    write_json(out_dir / "main_table_metrics.json", result)
    row = flatten_main_row(args.model, args.split, metrics)
    columns = list(row.keys())
    write_csv(out_dir / "main_table_metrics.csv", [row], columns)
    write_markdown_table(out_dir / "main_table_metrics.md", [row], columns)
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-root", required=True, type=Path, help="Root containing per-task raw prediction directories for one model.")
    parser.add_argument("--model", required=True, help="Model name to write in output files.")
    parser.add_argument("--mcq-root", required=True, type=Path, help="Released MCQ metadata root, used for fallback metadata resolution.")
    parser.add_argument("--gt-root", required=True, type=Path, help="Released GT metadata root, used for fallback metadata resolution.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for main_table_metrics.{json,csv,md}.")
    parser.add_argument("--effective-queryset-root", type=Path, default=None, help="Optional root containing effective querysets referenced by raw predictions.")
    parser.add_argument("--split", default="val", help="Split label written to outputs; default: val.")
    parser.add_argument("--path-map", action="append", default=[], help="Rewrite old absolute paths in raw JSON: OLD=NEW. May be repeated.")
    parser.add_argument("--ss-pos", action="append", default=[], help="Map state-switch position key to directory name, e.g. ss_pos_1=sspos_t_1_0. May be repeated.")
    parser.add_argument("--allow-missing-confidence", action="store_true", help="Write null confidence metrics instead of failing when cand_conf dirs are absent.")
    parser.add_argument("--write-details", action="store_true", help="Write per-sample details JSONL under <out-dir>/task_details/.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = evaluate(args)
    except EvalError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    print(f"[OK] wrote evaluation for {result['model']} -> {args.out_dir}")
    if result.get("warnings"):
        print(f"[WARN] {len(result['warnings'])} warning(s); see main_table_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

