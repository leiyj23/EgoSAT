#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
now_state_switch_score.py

Evaluate NOW_STATE_SWITCH (cand_state) across multiple sspos directories.

We scan:
  ~/benchmark_val/testllm/<MODEL_NAME>/now_state_switch/cand_state/sspos_*/   (e.g., sspos_t_1.0)

Each sspos dir contains prediction files:
  <MODEL_NAME>__<video_uid>__<clip_id>__<clip_uid>__pred__cand_state.json

We DO NOT rely on GT_DIR for scoring by default.
Instead we use pred["video_metadata"]["queryset_path"] to load the derived queryset, which contains:
  - region: segment/gap -> GT state
  - state_switch_pair_type: segment_to_gap / gap_to_segment
  - state_switch_role: fixed_segment_mid / scan_gap / fixed_gap_mid / scan_segment
  - state_switch_pair_id: pair grouping id

Metrics per sspos dir:
  1) Confusion matrix (GT state from queryset.region vs pred_state) over VALID pred states only.
  2) Switch success rate:
       - FG->BG (segment_to_gap): require (fixed_segment_mid + scan_gap) both correct.
       - BG->FG (gap_to_segment): require (fixed_gap_mid + scan_segment) both correct.
     Report over valid pairs; also report missing/invalid pair counts.
  3) Confidence proxy means for 4 point types (using token prob proxy):
       - FG2BG_fixed   : segment_to_gap + fixed_segment_mid
       - FG2BG_sspos   : segment_to_gap + scan_gap
       - BG2FG_fixed   : gap_to_segment + fixed_gap_mid
       - BG2FG_sspos   : gap_to_segment + scan_segment
     Proxy conf = the FIRST gen_token_prob within <STATE>...</STATE> span.
  4) NEW: Confidence proxy means GIVEN SUCCESSFUL SWITCH ONLY (pair-level success must hold):
       - same 4 point types, but only counted if the corresponding pair is a successful switch.

Outputs:
  ~/benchmark_val/score/<MODEL_NAME>/now_state_switch_cand_state/<sspos_dir>/{summary.json, details.jsonl}
  ~/benchmark_val/score/<MODEL_NAME>/now_state_switch_cand_state/summary_all.json

Run:
  MODEL_NAME=qwen2_5_vl_7b python3 now_state_switch_score.py
"""

from __future__ import annotations

import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -------------------------
# Paths / knobs
# -------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini_pro").strip() or "gemini_pro"

PRED_BASE_DIR = (
    Path.home() / "benchmark_val" / "testllm" / MODEL_NAME / "now_state_switch" / "cand_state"
)

OUT_ROOT = Path.home() / "benchmark_val" / "score" / MODEL_NAME / "now_state_switch_cand_state"


# -------------------------
# NEW: queryset field mapping by MODEL_NAME
# -------------------------
# Per your rule:
#   - qwen2_5_vl_7b / video_llava_7b / timechat_online_7b use: pred["source_queryset"]
#   - gemini_pro / claude_sonnet use: pred["video_metadata"]["queryset_path"]
_QUERYSET_FIELD_BY_MODEL: Dict[str, str] = {
    "qwen2_5_vl_7b": "source_queryset",
    "video_llava_7b": "source_queryset",
    "timechat_online_7b": "source_queryset",
    "gemini_pro": "video_metadata.queryset_path",
    "claude_sonnet": "video_metadata.queryset_path",
    "roi_timechat_online_7b_ft_step1484":"source_queryset",
    "timechat_online_7b_ft_step1484":"source_queryset",
}


def _get_queryset_path_from_pred(pred: Dict[str, Any]) -> Optional[str]:
    """
    Resolve derived queryset path from pred file, using MODEL_NAME mapping.
    Also includes a safe fallback to the other field family.
    """
    m = (MODEL_NAME or "").strip().lower()
    mode = _QUERYSET_FIELD_BY_MODEL.get(m, "video_metadata.queryset_path")

    qpath: Optional[str] = None
    if mode == "source_queryset":
        v = pred.get("source_queryset", None)
        qpath = str(v).strip() if isinstance(v, str) and v.strip() else None
        # fallback (just in case)
        if not qpath:
            vm = pred.get("video_metadata", {}) if isinstance(pred.get("video_metadata"), dict) else {}
            v2 = vm.get("queryset_path", None)
            qpath = str(v2).strip() if isinstance(v2, str) and v2.strip() else None
        return qpath

    # mode == video_metadata.queryset_path
    vm = pred.get("video_metadata", {}) if isinstance(pred.get("video_metadata"), dict) else {}
    v = vm.get("queryset_path", None)
    qpath = str(v).strip() if isinstance(v, str) and v.strip() else None
    # fallback (just in case)
    if not qpath:
        v2 = pred.get("source_queryset", None)
        qpath = str(v2).strip() if isinstance(v2, str) and v2.strip() else None
    return qpath


# -------------------------
# Regex / parsing helpers (same spirit as now_narration cand_state)
# -------------------------
_TAG_RE = {
    "state": re.compile(r"<\s*STATE\s*>(.*?)<\s*/\s*STATE\s*>", re.IGNORECASE | re.DOTALL),
}
_NO_INTER_RE = re.compile(r"\bno[_ ]interaction\b", re.IGNORECASE)
# "interaction" not preceded by "no_" or "no "
_INTER_ONLY_RE = re.compile(r"(?<!no[_ ])\binteraction\b", re.IGNORECASE)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_load_json_or_one_jsonl(path: Path) -> Dict[str, Any]:
    """
    Accept:
      - .json: dict
      - .jsonl: must contain exactly ONE JSON object
                (either single-line JSONL or pretty JSON saved as .jsonl).
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8").strip()
    if not raw:
        raw = p.read_text(encoding="utf-8-sig").strip()
    if not raw:
        raise ValueError(f"Empty file: {p}")

    # try full json first
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
            return obj[0]
    except Exception:
        pass

    # jsonl fallback
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    recs = [json.loads(ln) for ln in lines]
    if len(recs) != 1 or not isinstance(recs[0], dict):
        raise ValueError(f"Expected exactly 1 JSON object in JSONL: {p}, got {len(recs)}")
    return recs[0]


def parse_tag(text: str, which: str) -> Optional[str]:
    rx = _TAG_RE.get(which)
    if not rx:
        return None
    m = rx.search(text or "")
    if not m:
        return None
    return (m.group(1) or "").strip()


def normalize_state(x: Optional[str]) -> str:
    """
    Same as now_narration_eval:
      - accept variants / spaces / underscores
      - outputs: INTERACTION / NO_INTERACTION / UNKNOWN
    """
    s = (x or "").strip().lower().replace(" ", "_")
    if not s:
        return "UNKNOWN"
    if "no_interaction" in s:
        return "NO_INTERACTION"
    if "interaction" in s:
        return "INTERACTION"
    if s in {"no", "none", "null", "0"}:
        return "NO_INTERACTION"
    if s in {"yes", "1"}:
        return "INTERACTION"
    return "UNKNOWN"


def cand_state_from_clean_text(clean_text: str) -> Tuple[str, str]:
    """
    Same rule as now_narration_eval:
      - if ONLY 'interaction' appears -> INTERACTION
      - if ONLY 'no interaction/no_interaction' appears -> NO_INTERACTION
      - if both appear or neither appear -> INVALID
    Returns (state, method)
    """
    s = clean_text or ""
    has_no = _NO_INTER_RE.search(s) is not None
    has_inter = _INTER_ONLY_RE.search(s) is not None

    if has_no and has_inter:
        return "INVALID", "clean_ambiguous"
    if has_no:
        return "NO_INTERACTION", "clean_no_interaction"
    if has_inter:
        return "INTERACTION", "clean_interaction"
    return "INVALID", "clean_none"


def gt_state_from_region(sample: Dict[str, Any]) -> str:
    r = (sample.get("region") or "").strip().lower()
    if r == "segment":
        return "INTERACTION"
    if r == "gap":
        return "NO_INTERACTION"
    vi = sample.get("visible_interaction", None)
    if isinstance(vi, (int, float)):
        return "INTERACTION" if int(vi) == 1 else "NO_INTERACTION"
    return "UNKNOWN"


def build_pred_file_list(pred_dir: Path) -> List[Path]:
    """
    Keep ONLY prediction files (*__pred*.json) and skip manifest/derived/queryset files.
    """
    if not pred_dir.exists():
        return []
    files = sorted([p for p in pred_dir.rglob("*.json") if p.is_file()], key=lambda x: x.as_posix())
    out: List[Path] = []
    for p in files:
        name = p.name.lower()
        if "manifest" in name:
            continue
        if name.endswith("__pred.json") or name.endswith("_pred.json") or "__pred" in name:
            out.append(p)
    return out


def pretty_confusion_2x2(cm: Dict[str, Dict[str, int]]) -> str:
    rows = ["INTERACTION", "NO_INTERACTION"]
    cols = ["INTERACTION", "NO_INTERACTION"]
    lines = []
    header = "GT\\PRED".ljust(14) + "".join([c.rjust(16) for c in cols])
    lines.append(header)
    for r in rows:
        line = r.ljust(14)
        for c in cols:
            line += str(cm.get(r, {}).get(c, 0)).rjust(16)
        lines.append(line)
    return "\n".join(lines)


# -------------------------
# Token-prob confidence proxy
# -------------------------
def _norm_tok(t: str) -> str:
    return (t or "").strip()


def _tok_has_alnum(t: str) -> bool:
    return re.search(r"[A-Za-z0-9]", t or "") is not None


def _find_tag_span(tokens: List[str], tag: str) -> Optional[Tuple[int, int]]:
    """
    Return (content_start_idx, content_end_idx) for the FIRST <TAG>...</TAG> span.
    Works with either:
      - "<STATE>" in one token
      - "<", "STATE", ">" split tokens (also tolerates tokens like "<", "STATE>")
    Similar for closing: "</STATE>" or "</", "STATE", ">"
    """
    if not tokens:
        return None
    T = [str(x) for x in tokens]
    U = [x.upper() for x in T]
    tagU = tag.upper()

    start_close = None  # index of the token that contains/ends the start tag
    end_open = None     # index of the token that begins the end tag

    # ---- find start tag ----
    for i in range(len(U)):
        ti = U[i]
        if f"<{tagU}>" in ti:
            start_close = i
            break
        # pattern: "<", "STATE", ">" (3 tokens)
        if i + 2 < len(U):
            if "<" in U[i] and tagU in U[i + 1] and ">" in U[i + 2]:
                start_close = i + 2
                break
        # pattern: "<", "STATE>" (2 tokens)
        if i + 1 < len(U):
            if "<" in U[i] and (tagU in U[i + 1] and ">" in U[i + 1]):
                start_close = i + 1
                break
        # pattern: "<STATE", ">" (2 tokens)
        if i + 1 < len(U):
            if (f"<{tagU}" in U[i]) and ">" in U[i + 1]:
                start_close = i + 1
                break

    if start_close is None:
        return None

    content_start = start_close + 1

    # ---- find end tag ----
    for j in range(content_start, len(U)):
        tj = U[j]
        if f"</{tagU}>" in tj:
            end_open = j
            break
        # pattern: "</", "STATE", ">"
        if j + 2 < len(U):
            if "</" in U[j] and tagU in U[j + 1] and ">" in U[j + 2]:
                end_open = j
                break
        # pattern: "</STATE", ">"
        if j + 1 < len(U):
            if f"</{tagU}" in U[j] and ">" in U[j + 1]:
                end_open = j
                break

    if end_open is None:
        return None

    content_end = end_open
    if content_end < content_start:
        return None
    return content_start, content_end


def state_token_prob_proxy(ps: Dict[str, Any]) -> Optional[float]:
    """
    Proxy confidence:
      - find <STATE>...</STATE> span in gen_tokens
      - take the FIRST token prob within that span that has alnum
    """
    toks = ps.get("gen_tokens", None)
    probs = ps.get("gen_token_probs", None)
    if not isinstance(toks, list) or not isinstance(probs, list):
        return None
    if not toks or not probs or len(toks) != len(probs):
        return None

    span = _find_tag_span([_norm_tok(x) for x in toks], tag="STATE")
    if span is None:
        return None
    a, b = span
    if a >= b:
        return None

    for i in range(a, b):
        if _tok_has_alnum(str(toks[i])):
            try:
                return float(probs[i])
            except Exception:
                return None
    return None


# -------------------------
# Point typing (your 4 classes)
# -------------------------
def point_type_from_pair_role(pair_type: str, role: str) -> Optional[str]:
    pt = (pair_type or "").strip().lower()
    rr = (role or "").strip().lower()

    # FG->BG
    if pt == "segment_to_gap" and rr == "fixed_segment_mid":
        return "FG2BG_fixed"
    if pt == "segment_to_gap" and rr == "scan_gap":
        return "FG2BG_sspos"

    # BG->FG
    if pt == "gap_to_segment" and rr == "fixed_gap_mid":
        return "BG2FG_fixed"
    if pt == "gap_to_segment" and rr == "scan_segment":
        return "BG2FG_sspos"

    return None


def required_roles_for_pair_type(pair_type: str) -> Optional[Tuple[str, str]]:
    pt = (pair_type or "").strip().lower()
    if pt == "segment_to_gap":
        return ("fixed_segment_mid", "scan_gap")
    if pt == "gap_to_segment":
        return ("fixed_gap_mid", "scan_segment")
    return None


# -------------------------
# Core evaluation for one sspos dir
# -------------------------
def eval_one_sspos_dir(pred_dir: Path, out_dir: Path) -> Dict[str, Any]:
    _ensure_dir(out_dir)
    details_path = out_dir / "details.jsonl"
    summary_path = out_dir / "summary.json"
    if details_path.exists():
        details_path.unlink(missing_ok=True)

    pred_files = build_pred_file_list(pred_dir)

    # confusion over VALID answers only
    cm_valid: Dict[str, Dict[str, int]] = {}

    def cm_add(gt_s: str, pr_s: str) -> None:
        if gt_s not in {"INTERACTION", "NO_INTERACTION"}:
            return
        if pr_s not in {"INTERACTION", "NO_INTERACTION"}:
            return
        cm_valid.setdefault(gt_s, {})
        cm_valid[gt_s][pr_s] = int(cm_valid[gt_s].get(pr_s, 0)) + 1

    # confidence sums for 4 point types
    conf_sum: Dict[str, float] = {}
    conf_cnt: Dict[str, int] = {}

    def conf_add(k: str, v: Optional[float]) -> None:
        if not k:
            return
        if v is None:
            return
        conf_sum[k] = float(conf_sum.get(k, 0.0)) + float(v)
        conf_cnt[k] = int(conf_cnt.get(k, 0)) + 1

    # NEW: confidence sums GIVEN successful switch only
    conf_sum_ok: Dict[str, float] = {}
    conf_cnt_ok: Dict[str, int] = {}

    def conf_add_ok(k: str, v: Optional[float]) -> None:
        if not k:
            return
        if v is None:
            return
        conf_sum_ok[k] = float(conf_sum_ok.get(k, 0.0)) + float(v)
        conf_cnt_ok[k] = int(conf_cnt_ok.get(k, 0)) + 1

    # pairs accumulation
    pairs: Dict[str, Dict[str, Any]] = {}  # pair_id -> {"pair_type":..., "roles": {role: {...}}}

    # queryset cache
    queryset_cache: Dict[str, Dict[str, Any]] = {}

    # counters
    num_files_total = 0
    num_files_loaded = 0
    skipped_empty_pred_files = 0  # NEW
    missing_queryset_path = 0
    missing_queryset_file = 0
    missing_queryset_sample = 0

    total_samples = 0
    valid_samples = 0
    invalid_samples = 0

    for pf in pred_files:
        num_files_total += 1
        try:
            pred = json.loads(pf.read_text(encoding="utf-8"))
        except Exception:
            continue
        num_files_loaded += 1

        # NEW: skip empty pred files early
        psamples = pred.get("samples", [])
        if (not isinstance(psamples, list)) or (len(psamples) == 0):
            skipped_empty_pred_files += 1
            continue

        # NEW: queryset path mapping by MODEL_NAME
        queryset_path = _get_queryset_path_from_pred(pred)
        if not isinstance(queryset_path, str) or not queryset_path.strip():
            missing_queryset_path += 1
            continue
        queryset_path = str(queryset_path).strip()

        qp = Path(queryset_path).expanduser()
        if not qp.is_file():
            missing_queryset_file += 1
            continue

        if queryset_path in queryset_cache:
            qobj = queryset_cache[queryset_path]
        else:
            print(f"[DEBUG] loading queryset: {qp}")
            try:
                qobj = safe_load_json_or_one_jsonl(qp)
            except Exception as e:
                print(f"[WARN] bad queryset file, skip: {qp}")
                print(f"[WARN] error: {repr(e)}")
                continue
            queryset_cache[queryset_path] = qobj

        qsamples = qobj.get("samples", [])
        if not isinstance(qsamples, list):
            qsamples = []
        qs_by_idx: Dict[int, Dict[str, Any]] = {}
        for s in qsamples:
            if not isinstance(s, dict):
                continue
            idx = s.get("idx", None)
            if isinstance(idx, (int, float)):
                qs_by_idx[int(idx)] = s

        for ps in psamples:
            if not isinstance(ps, dict):
                continue
            total_samples += 1

            p_idx = ps.get("idx", None)
            if not isinstance(p_idx, (int, float)):
                missing_queryset_sample += 1
                continue
            idxi = int(p_idx)
            gt_s = qs_by_idx.get(idxi)
            if gt_s is None:
                missing_queryset_sample += 1
                continue

            gt_state = gt_state_from_region(gt_s)
            pair_type = str(gt_s.get("state_switch_pair_type", "") or "").strip()
            role = str(gt_s.get("state_switch_role", "") or "").strip()
            pair_id = str(gt_s.get("state_switch_pair_id", "") or "").strip()

            resp_text = str(ps.get("response_text", "") or "")
            clean_text = str(ps.get("clean_response", "") or ps.get("clean", "") or resp_text or "")

            pred_state_raw = parse_tag(resp_text, "state")
            pred_state = "INVALID"
            pred_method = "invalid"

            if pred_state_raw is not None:
                ns = normalize_state(pred_state_raw)
                if ns in {"INTERACTION", "NO_INTERACTION"}:
                    pred_state = ns
                    pred_method = "tag"
                else:
                    ps2, m2 = cand_state_from_clean_text(clean_text)
                    pred_state = ps2
                    pred_method = m2
            else:
                ps2, m2 = cand_state_from_clean_text(clean_text)
                pred_state = ps2
                pred_method = m2

            is_valid = pred_state in {"INTERACTION", "NO_INTERACTION"}
            if is_valid:
                valid_samples += 1
                cm_add(gt_state, pred_state)
            else:
                invalid_samples += 1

            # conf proxy (token prob)
            conf_proxy = state_token_prob_proxy(ps)

            # 4-class confidence means (unconditional)
            ptype = point_type_from_pair_role(pair_type, role)
            if ptype:
                conf_add(ptype, conf_proxy)

            # pair accumulation
            if pair_id:
                rec = pairs.setdefault(pair_id, {"pair_type": pair_type, "roles": {}})
                rec["pair_type"] = rec.get("pair_type") or pair_type
                rec["roles"][role] = {
                    "idx": idxi,
                    "gt_state": gt_state,
                    "pred_state": pred_state,
                    "valid": bool(is_valid),
                    "conf_proxy": conf_proxy,
                }

            detail = {
                "pred_file": pf.name,
                "queryset_path": queryset_path,
                "idx": idxi,
                "t_eval": float(ps.get("t_eval")) if isinstance(ps.get("t_eval"), (int, float)) else None,
                "gt_state": gt_state,
                "pred_state_raw": pred_state_raw,
                "pred_state": pred_state,
                "pred_state_method": pred_method,
                "valid": bool(is_valid),
                "conf_proxy_tokenprob_first_state_token": conf_proxy,
                "pair_id": pair_id or None,
                "pair_type": pair_type or None,
                "role": role or None,
                "point_type": ptype,
            }
            with open(details_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(detail, ensure_ascii=False) + "\n")

    # ---- compute switch success ----
    fg2bg_total = 0
    fg2bg_valid = 0
    fg2bg_success = 0
    fg2bg_missing = 0
    fg2bg_invalid_pair = 0

    bg2fg_total = 0
    bg2fg_valid = 0
    bg2fg_success = 0
    bg2fg_missing = 0
    bg2fg_invalid_pair = 0

    for pid, rec in pairs.items():
        pt = str(rec.get("pair_type", "") or "").strip().lower()
        roles = rec.get("roles", {}) if isinstance(rec.get("roles"), dict) else {}
        req = required_roles_for_pair_type(pt)
        if req is None:
            continue
        r_fixed, r_scan = req

        if pt == "segment_to_gap":
            fg2bg_total += 1
        elif pt == "gap_to_segment":
            bg2fg_total += 1

        if (r_fixed not in roles) or (r_scan not in roles):
            if pt == "segment_to_gap":
                fg2bg_missing += 1
            else:
                bg2fg_missing += 1
            continue

        a = roles[r_fixed]
        b = roles[r_scan]

        if not (a.get("valid", False) and b.get("valid", False)):
            if pt == "segment_to_gap":
                fg2bg_invalid_pair += 1
            else:
                bg2fg_invalid_pair += 1
            continue

        # valid pair (both probes valid)
        if pt == "segment_to_gap":
            fg2bg_valid += 1
        else:
            bg2fg_valid += 1

        ok = (a.get("pred_state") == a.get("gt_state")) and (b.get("pred_state") == b.get("gt_state"))
        if ok:
            if pt == "segment_to_gap":
                fg2bg_success += 1
            else:
                bg2fg_success += 1

            # NEW: confidence GIVEN successful switch
            # add both probes' conf into the corresponding 4 point types
            ptype_a = point_type_from_pair_role(pt, r_fixed)
            ptype_b = point_type_from_pair_role(pt, r_scan)
            if ptype_a:
                conf_add_ok(ptype_a, a.get("conf_proxy", None))
            if ptype_b:
                conf_add_ok(ptype_b, b.get("conf_proxy", None))

    # ---- finalize confidence means ----
    conf_mean: Dict[str, Optional[float]] = {}
    for k in ["FG2BG_fixed", "FG2BG_sspos", "BG2FG_fixed", "BG2FG_sspos"]:
        c = int(conf_cnt.get(k, 0))
        conf_mean[k] = (float(conf_sum.get(k, 0.0)) / c) if c > 0 else None

    # NEW: finalize confidence means given successful switch
    conf_mean_ok: Dict[str, Optional[float]] = {}
    for k in ["FG2BG_fixed", "FG2BG_sspos", "BG2FG_fixed", "BG2FG_sspos"]:
        c = int(conf_cnt_ok.get(k, 0))
        conf_mean_ok[k] = (float(conf_sum_ok.get(k, 0.0)) / c) if c > 0 else None

    # ---- finalize confusion stats ----
    tp = int(cm_valid.get("INTERACTION", {}).get("INTERACTION", 0))
    tn = int(cm_valid.get("NO_INTERACTION", {}).get("NO_INTERACTION", 0))
    total_valid_for_cm = 0
    for g in ("INTERACTION", "NO_INTERACTION"):
        for p in ("INTERACTION", "NO_INTERACTION"):
            total_valid_for_cm += int(cm_valid.get(g, {}).get(p, 0))
    acc_cm = (tp + tn) / total_valid_for_cm if total_valid_for_cm > 0 else 0.0

    # ---- pack summary ----
    summary = {
        "model_name": MODEL_NAME,
        "pred_dir": str(pred_dir),
        "num_pred_files_total": int(num_files_total),
        "num_pred_files_loaded": int(num_files_loaded),
        "skipped_empty_pred_files": int(skipped_empty_pred_files),  # NEW
        "missing_queryset_path_in_pred": int(missing_queryset_path),
        "missing_queryset_file": int(missing_queryset_file),
        "missing_queryset_sample_match": int(missing_queryset_sample),
        "samples_total_seen": int(total_samples),
        "valid_samples": int(valid_samples),
        "invalid_samples": int(invalid_samples),
        "valid_ratio": float(valid_samples / total_samples) if total_samples > 0 else 0.0,

        "confusion_valid_only": cm_valid,
        "accuracy_over_valid_for_cm": float(acc_cm),
        "confusion_pretty": pretty_confusion_2x2(cm_valid),

        "switch": {
            "FG_to_BG": {
                "pair_type": "segment_to_gap",
                "pairs_total_seen": int(fg2bg_total),
                "pairs_missing_required_roles": int(fg2bg_missing),
                "pairs_invalid_due_to_invalid_probe": int(fg2bg_invalid_pair),
                "pairs_valid": int(fg2bg_valid),
                "pairs_success_both_probes_correct": int(fg2bg_success),
                "success_rate_over_valid_pairs": float(fg2bg_success / fg2bg_valid) if fg2bg_valid > 0 else 0.0,
            },
            "BG_to_FG": {
                "pair_type": "gap_to_segment",
                "pairs_total_seen": int(bg2fg_total),
                "pairs_missing_required_roles": int(bg2fg_missing),
                "pairs_invalid_due_to_invalid_probe": int(bg2fg_invalid_pair),
                "pairs_valid": int(bg2fg_valid),
                "pairs_success_both_probes_correct": int(bg2fg_success),
                "success_rate_over_valid_pairs": float(bg2fg_success / bg2fg_valid) if bg2fg_valid > 0 else 0.0,
            },
        },

        "confidence_proxy_tokenprob_first_state_token": {
            "definition": "first gen_token_prob within <STATE>...</STATE> span (first token with alnum)",
            "means": conf_mean,
            "counts": {k: int(conf_cnt.get(k, 0)) for k in ["FG2BG_fixed", "FG2BG_sspos", "BG2FG_fixed", "BG2FG_sspos"]},
        },

        # NEW: confidence means given pair-level successful switch
        "confidence_proxy_tokenprob_first_state_token_given_successful_switch": {
            "definition": "same proxy, but only for probes that belong to pairs where BOTH probes are correct (successful switch)",
            "means": conf_mean_ok,
            "counts": {k: int(conf_cnt_ok.get(k, 0)) for k in ["FG2BG_fixed", "FG2BG_sspos", "BG2FG_fixed", "BG2FG_sspos"]},
        },

        "timestamp_unix": time.time(),
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# -------------------------
# Main: scan sspos dirs
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_base_dir",
        type=str,
        default=str(PRED_BASE_DIR),
        help="Base dir containing sspos_* subdirs (default: ~/benchmark_val/testllm/<MODEL_NAME>/now_state_switch/cand_state)",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=str(OUT_ROOT),
        help="Output root (default: ~/benchmark_val/score/<MODEL_NAME>/now_state_switch_cand_state)",
    )
    args = parser.parse_args()

    pred_base = Path(args.pred_base_dir).expanduser()
    out_root = Path(args.out_root).expanduser()
    _ensure_dir(out_root)

    if not pred_base.exists():
        raise FileNotFoundError(f"pred_base_dir not found: {pred_base}")

    # collect sspos_* subdirs; if none, evaluate the base dir directly
    sspos_dirs = sorted([p for p in pred_base.iterdir() if p.is_dir() and p.name.startswith("sspos_")], key=lambda x: x.name)
    if not sspos_dirs:
        sspos_dirs = [pred_base]

    summaries: Dict[str, Any] = {
        "model_name": MODEL_NAME,
        "pred_base_dir": str(pred_base),
        "evaluated_dirs": [],
        "timestamp_unix": time.time(),
    }

    for d in sspos_dirs:
        tag = d.name
        out_dir = out_root / tag
        print("\n" + "=" * 70)
        print(f"[EVAL] sspos_dir: {d}")
        s = eval_one_sspos_dir(d, out_dir)

        # pretty console
        sw_fg2bg = s["switch"]["FG_to_BG"]
        sw_bg2fg = s["switch"]["BG_to_FG"]
        conf = s["confidence_proxy_tokenprob_first_state_token"]
        means = conf["means"]
        counts = conf["counts"]

        conf_ok = s["confidence_proxy_tokenprob_first_state_token_given_successful_switch"]
        means_ok = conf_ok["means"]
        counts_ok = conf_ok["counts"]

        print(f"[OUT] {out_dir / 'summary.json'}")
        print(f"[OUT] {out_dir / 'details.jsonl'}")
        print("")
        print("---- Confusion (VALID only) ----")
        print(s["confusion_pretty"])
        print(f"Acc(valid-only CM): {s['accuracy_over_valid_for_cm']:.4f}")
        print("")
        print("---- Switch success (both probes correct) ----")
        print(f"FG->BG: {sw_fg2bg['success_rate_over_valid_pairs']:.4f} "
              f"({sw_fg2bg['pairs_success_both_probes_correct']}/{sw_fg2bg['pairs_valid']}) "
              f"missing={sw_fg2bg['pairs_missing_required_roles']}, invalid_pair={sw_fg2bg['pairs_invalid_due_to_invalid_probe']}")
        print(f"BG->FG: {sw_bg2fg['success_rate_over_valid_pairs']:.4f} "
              f"({sw_bg2fg['pairs_success_both_probes_correct']}/{sw_bg2fg['pairs_valid']}) "
              f"missing={sw_bg2fg['pairs_missing_required_roles']}, invalid_pair={sw_bg2fg['pairs_invalid_due_to_invalid_probe']}")
        print("")
        print("---- Confidence proxy means (token prob) ----")
        for k in ["FG2BG_fixed", "FG2BG_sspos", "BG2FG_fixed", "BG2FG_sspos"]:
            m = means.get(k)
            c = counts.get(k, 0)
            if m is None:
                print(f"{k:12s}: None  (n={c})")
            else:
                print(f"{k:12s}: {m:.6f}  (n={c})")

        print("")
        print("---- Confidence proxy means GIVEN successful switch (clean) ----")
        for k in ["FG2BG_fixed", "FG2BG_sspos", "BG2FG_fixed", "BG2FG_sspos"]:
            m = means_ok.get(k)
            c = counts_ok.get(k, 0)
            if m is None:
                print(f"{k:12s}: None  (n={c})")
            else:
                print(f"{k:12s}: {m:.6f}  (n={c})")

        summaries["evaluated_dirs"].append({tag: s})

    # write top-level summary
    (out_root / "summary_all.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 70)
    print(f"[WROTE] {out_root / 'summary_all.json'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()