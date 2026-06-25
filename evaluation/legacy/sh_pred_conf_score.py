#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sh_pred_cand_conf_eval.py

Eval for sh_pred in cand_conf mode.

We report metrics for:
- Predictable group: NN
- Unpredictable group: NP/PN/PP
Plus:
- mean confidence for WRONG predictions within Unpredictable group

Inputs:
  ~/benchmark_val/testllm/<MODEL_NAME>/sh_pred/cand_conf/*.json

Each pred file should contain:
  - source_queryset (preferred) pointing to derived queryset
We load that queryset and match samples by idx.

Confidence source priority:
  1) diagnostics.prompt_and_encoding_debug.cand_conf_probe.p_cond[ANS] if available
  2) gen_token_probs[0] fallback

Outputs:
  ~/benchmark_val/score/<MODEL_NAME>/sh_pred_cand_conf/{summary.json, details.jsonl}

Run:
  MODEL_NAME=qwen2_5_vl_7b python3 sh_pred_cand_conf_eval.py
"""

from __future__ import annotations

import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

try:
    from .mcq_source_stats import (
        init_source_stats,
        source_stats_report_lines,
        summarize_source_stats,
        update_source_stats,
    )
except ImportError:
    from mcq_source_stats import (
        init_source_stats,
        source_stats_report_lines,
        summarize_source_stats,
        update_source_stats,
    )


# -------------------------
# Paths / knobs
# -------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini_pro").strip() or "gemini_pro"

PRED_DIR = Path.home() / "benchmark_val" / "testllm" / MODEL_NAME / "sh_pred" / "cand_conf"
OUT_DIR = Path.home() / "benchmark_val" / "score" / MODEL_NAME / "sh_pred_cand_conf"

OUT_SUMMARY = OUT_DIR / "summary.json"
OUT_DETAILS = OUT_DIR / "details.jsonl"

PRED_LOG_EVERY_FILES = 10
SAMPLE_LOG_EVERY = 800

RESP_SNIPPET = 240


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _snippet(s: str, n: int) -> str:
    s = (s or "").replace("\n", "\\n")
    return s if len(s) <= n else (s[:n] + "...")


def safe_load_json_or_one_jsonl(path: Path) -> Dict[str, Any]:
    """
    Accept:
      - JSON dict
      - JSONL with exactly ONE JSON object (or pretty JSON mistakenly saved as .jsonl)
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8").strip()
    if not raw:
        raw = p.read_text(encoding="utf-8-sig").strip()
    if not raw:
        raise ValueError(f"Empty file: {p}")

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
            return obj[0]
    except Exception:
        pass

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    recs = [json.loads(ln) for ln in lines]
    if len(recs) != 1 or not isinstance(recs[0], dict):
        raise ValueError(f"Expected exactly 1 JSON object in JSONL: {p}, got {len(recs)}")
    return recs[0]


def build_pred_file_list(pred_dir: Path) -> List[Path]:
    if not pred_dir.exists():
        return []
    files = sorted([p for p in pred_dir.rglob("*.json") if p.is_file()], key=lambda x: x.as_posix())
    out: List[Path] = []
    skipped = 0
    for p in files:
        name = p.name.lower()
        if "manifest" in name:
            skipped += 1
            continue
        out.append(p)
    _log(f"Pred list: {pred_dir} -> {len(out)} files (skipped_manifest={skipped})")
    return out


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
    """
    Priority:
      1) pred["source_queryset"]
      2) pred["video_metadata"]["queryset_path"]
      3) any nested key/value containing "queryset" (choose first existing file)
    """
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

    cands = _collect_queryset_candidates(pred)
    for _kp, vv in cands:
        if not isinstance(vv, str) or not vv.strip():
            continue
        p = Path(vv).expanduser()
        if p.is_file():
            return str(p)

    return None


def normalize_ans_letter(x: str) -> str:
    s = (x or "").strip().upper()
    s = re.sub(r"[^A-D]", "", s)
    return s[0] if s and s[0] in "ABCD" else ""


def _extract_ans_from_pred_sample(ps: Dict[str, Any]) -> str:
    # prefer parsed.ans
    parsed = ps.get("parsed", {}) if isinstance(ps.get("parsed"), dict) else {}
    a = parsed.get("ans", None)
    if isinstance(a, str):
        aa = normalize_ans_letter(a)
        if aa:
            return aa

    # try clean_response / response_text
    for k in ["clean_response", "clean", "response_text"]:
        v = ps.get(k, None)
        if isinstance(v, str):
            aa = normalize_ans_letter(v)
            if aa:
                return aa

    return ""


def _extract_conf_from_pred_sample(ps: Dict[str, Any], ans: str) -> Tuple[Optional[float], str]:
    """
    Return (confidence, source_str).
    Priority:
      1) diagnostics.prompt_and_encoding_debug.cand_conf_probe.p_cond[ans]
      2) gen_token_probs[0]
    """
    # (1) p_cond from cand_conf_probe
    diag = ps.get("diagnostics", {}) if isinstance(ps.get("diagnostics"), dict) else {}
    ped = diag.get("prompt_and_encoding_debug", {}) if isinstance(diag.get("prompt_and_encoding_debug"), dict) else {}
    probe = ped.get("cand_conf_probe", {}) if isinstance(ped.get("cand_conf_probe"), dict) else {}
    if probe.get("ok") and isinstance(probe.get("p_cond"), dict):
        pc = probe["p_cond"]
        if isinstance(ans, str) and ans in pc:
            try:
                v = float(pc[ans])
                if v < 0.0:
                    v = 0.0
                if v > 1.0:
                    v = 1.0
                return v, "cand_conf_p_cond"
            except Exception:
                pass

    # (2) fallback gen_token_probs[0]
    probs = ps.get("gen_token_probs", None)
    if isinstance(probs, list) and len(probs) >= 1:
        try:
            v = float(probs[0])
            if v < 0.0:
                v = 0.0
            if v > 1.0:
                v = 1.0
            return v, "gen_token_prob0"
        except Exception:
            pass

    return None, "none"


def _mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def eval_sh_pred_cand_conf() -> Dict[str, Any]:
    _ensure_dir(OUT_DIR)
    if OUT_DETAILS.exists():
        OUT_DETAILS.unlink(missing_ok=True)

    pred_files = build_pred_file_list(PRED_DIR)
    _log(f"SH_PRED(CAND_CONF): start processing {len(pred_files)} pred files")

    queryset_cache: Dict[str, Dict[str, Any]] = {}

    # groups
    # - predictable: NN
    # - branch_only: PN
    # - surprise_only: NP
    # - branch_and_surprise: PP
    # Keep UNP as a legacy aggregate for compatibility.
    groups = {
        "NN": {"total": 0, "valid": 0, "correct": 0, "conf_valid": [], "conf_correct": [], "conf_wrong": []},
        "PN": {"total": 0, "valid": 0, "correct": 0, "conf_valid": [], "conf_correct": [], "conf_wrong": []},
        "NP": {"total": 0, "valid": 0, "correct": 0, "conf_valid": [], "conf_correct": [], "conf_wrong": []},
        "PP": {"total": 0, "valid": 0, "correct": 0, "conf_valid": [], "conf_correct": [], "conf_wrong": []},
        "UNP": {"total": 0, "valid": 0, "correct": 0, "conf_valid": [], "conf_correct": [], "conf_wrong": []},
        "UNK": {"total": 0, "valid": 0, "correct": 0, "conf_valid": [], "conf_correct": [], "conf_wrong": []},
        "ALL": {"total": 0, "valid": 0, "correct": 0, "conf_valid": [], "conf_correct": [], "conf_wrong": []},
    }
    source_stats_by_group: Dict[str, Dict[str, Any]] = {
        "NN": init_source_stats(),
        "PN": init_source_stats(),
        "NP": init_source_stats(),
        "PP": init_source_stats(),
        "UNP": init_source_stats(),
        "UNK": init_source_stats(),
        "ALL": init_source_stats(),
    }

    # file counters
    num_files_total = 0
    num_files_loaded = 0
    missing_queryset_path = 0
    missing_queryset_file = 0
    queryset_load_fail = 0

    total_samples = 0
    invalid_samples = 0

    def _which_group(state: str) -> str:
        st = (state or "").strip().upper()
        if st in {"NN", "NP", "PN", "PP"}:
            return st
        return "UNK"

    for fi, pf in enumerate(pred_files, 1):
        num_files_total += 1
        if fi == 1 or (fi % PRED_LOG_EVERY_FILES == 0):
            _log(f"SH_PRED: file {fi}/{len(pred_files)} -> {pf.name} (samples={total_samples})")

        try:
            pred = json.loads(pf.read_text(encoding="utf-8"))
        except Exception as e:
            _log(f"SH_PRED: pred json load failed: {pf.name} err={repr(e)}")
            continue
        num_files_loaded += 1

        qpath = resolve_queryset_path_from_pred(pred)
        if not qpath:
            missing_queryset_path += 1
            continue
        qp = Path(qpath).expanduser()
        if not qp.is_file():
            missing_queryset_file += 1
            continue

        if qpath in queryset_cache:
            qs = queryset_cache[qpath]
        else:
            try:
                qs = safe_load_json_or_one_jsonl(qp)
            except Exception as e:
                queryset_load_fail += 1
                _log(f"SH_PRED: queryset load failed: {qp.name} err={repr(e)}")
                continue
            queryset_cache[qpath] = qs

        qsamples = qs.get("samples", [])
        if not isinstance(qsamples, list):
            qsamples = []
        qs_by_idx: Dict[int, Dict[str, Any]] = {}
        for s in qsamples:
            if not isinstance(s, dict):
                continue
            idx = s.get("idx", None)
            if isinstance(idx, (int, float)):
                qs_by_idx[int(idx)] = s

        psamples = pred.get("samples", [])
        if not isinstance(psamples, list):
            continue

        for ps in psamples:
            if not isinstance(ps, dict):
                continue

            total_samples += 1
            if total_samples == 1 or (total_samples % SAMPLE_LOG_EVERY == 0):
                _log(f"SH_PRED: samples processed={total_samples}, invalid={invalid_samples}")

            # idx + gt
            p_idx = ps.get("idx", None)
            if not isinstance(p_idx, (int, float)):
                invalid_samples += 1
                continue
            idxi = int(p_idx)

            gt_s = qs_by_idx.get(idxi)
            if gt_s is None:
                invalid_samples += 1
                continue

            # state
            state = str(gt_s.get("state", "") or "").strip().upper()
            grp = _which_group(state)

            # mcq answer
            mcq = gt_s.get("mcq", {}) if isinstance(gt_s.get("mcq"), dict) else {}
            gt_ans = normalize_ans_letter(str(mcq.get("answer", "") or ""))
            answer_idx = mcq.get("answer_idx", None)
            options = mcq.get("options", [])
            if not isinstance(options, list):
                options = []
            options4 = [str(x) for x in options[:4]]
            option_sources = mcq.get("option_sources", [])
            if not isinstance(option_sources, list):
                option_sources = []
            option_sources4 = [str(x) for x in option_sources[:4]] if option_sources else []

            pred_ans = _extract_ans_from_pred_sample(ps)
            is_valid = bool(pred_ans) and (pred_ans in {"A", "B", "C", "D"}) and bool(gt_ans)

            conf, conf_src = _extract_conf_from_pred_sample(ps, pred_ans) if is_valid else (None, "none")

            hit = bool(is_valid and (pred_ans == gt_ans))

            # update buckets
            update_groups = ["ALL", grp]
            if grp in {"NP", "PN", "PP"}:
                update_groups.append("UNP")
            for gname in update_groups:
                groups[gname]["total"] += 1
                if is_valid:
                    groups[gname]["valid"] += 1
                    if conf is not None:
                        groups[gname]["conf_valid"].append(float(conf))
                    if hit:
                        groups[gname]["correct"] += 1
                        if conf is not None:
                            groups[gname]["conf_correct"].append(float(conf))
                    else:
                        # wrong among valid
                        if conf is not None:
                            groups[gname]["conf_wrong"].append(float(conf))

            source_info_all = update_source_stats(
                source_stats_by_group["ALL"],
                pred_ans,
                gt_ans,
                answer_idx,
                option_sources,
                bool(is_valid),
            )
            source_info_group = update_source_stats(
                source_stats_by_group[grp],
                pred_ans,
                gt_ans,
                answer_idx,
                option_sources,
                bool(is_valid),
            )
            if grp in {"NP", "PN", "PP"}:
                update_source_stats(
                    source_stats_by_group["UNP"],
                    pred_ans,
                    gt_ans,
                    answer_idx,
                    option_sources,
                    bool(is_valid),
                )
            source_info = source_info_group or source_info_all

            # details
            resp_text = str(ps.get("response_text", "") or "")
            detail = {
                "file": pf.name,
                "queryset_path": qpath,
                "idx": idxi,
                "state": state if state else "UNK",
                "group": grp,

                "gt_ans": gt_ans,
                "pred_ans": pred_ans if pred_ans else None,
                "answer_letter": gt_ans if gt_ans else None,
                "pred_letter": pred_ans if pred_ans else None,
                "valid": bool(is_valid),
                "wrong": bool(is_valid and not hit),
                "hit": bool(hit),
                "options": options4,
                "option_sources": option_sources4 if option_sources4 else None,
                "gt_source": source_info.get("gt_source"),
                "pred_source": source_info.get("pred_source"),
                "is_disturb": source_info.get("is_disturb"),
                "is_strong_distractor": source_info.get("is_strong_distractor"),
                "mcq_source": "effective_queryset",
                "mcp_file": mcq.get("mcp_file", None),

                "conf": conf,
                "conf_source": conf_src,

                "resp_snippet": _snippet(resp_text, RESP_SNIPPET),
            }
            with open(OUT_DETAILS, "a", encoding="utf-8") as f:
                f.write(json.dumps(detail, ensure_ascii=False) + "\n")

    def _pack(name: str, g: Dict[str, Any]) -> Dict[str, Any]:
        total = int(g["total"])
        valid = int(g["valid"])
        correct = int(g["correct"])
        invalid = int(max(0, total - valid))
        acc_valid = float(correct / valid) if valid > 0 else 0.0
        out = {
            "samples_total": total,
            "samples_valid": valid,
            "samples_invalid": invalid,
            "correct": correct,
            "accuracy_over_valid": acc_valid,

            "conf_mean_over_valid": _mean([float(x) for x in g["conf_valid"]]),
            "conf_mean_over_correct": _mean([float(x) for x in g["conf_correct"]]),
            "conf_mean_over_wrong": _mean([float(x) for x in g["conf_wrong"]]),

            "conf_count_over_valid": int(len(g["conf_valid"])),
            "conf_count_over_correct": int(len(g["conf_correct"])),
            "conf_count_over_wrong": int(len(g["conf_wrong"])),
        }
        st = source_stats_by_group.get(name, init_source_stats())
        source_summary = summarize_source_stats(
            total_count=total,
            valid_count=valid,
            invalid_count=invalid,
            mapped_count=int(st.get("mapped_count", 0)),
            correct_mapped=int(st.get("correct_mapped", 0)),
            selected_source_counts=st.get("selected_source_counts", {}),
            wrong_source_counts=st.get("wrong_source_counts", {}),
            gt_source_counts=st.get("gt_source_counts", {}),
            option_source_missing_count=int(st.get("option_source_missing_count", 0)),
        )
        for k, v in source_summary.items():
            out.setdefault(k, v)
        return out

    summary = {
        "model_name": MODEL_NAME,
        "task": "sh_pred_cand_conf",
        "pred_dir": str(PRED_DIR),
        "out_dir": str(OUT_DIR),

        "num_pred_files_total": int(num_files_total),
        "num_pred_files_loaded": int(num_files_loaded),
        "missing_queryset_path": int(missing_queryset_path),
        "missing_queryset_file": int(missing_queryset_file),
        "queryset_load_fail": int(queryset_load_fail),

        "samples_total_seen": int(total_samples),
        "samples_invalid": int(invalid_samples),

        "groups": {
            "NN_predictable": _pack("NN", groups["NN"]),
            "PN_branch_only": _pack("PN", groups["PN"]),
            "NP_surprise_only": _pack("NP", groups["NP"]),
            "PP_branch_and_surprise": _pack("PP", groups["PP"]),
            "UNP_unpredictable_NP_PN_PP": _pack("UNP", groups["UNP"]),
            "UNK": _pack("UNK", groups["UNK"]),
            "ALL": _pack("ALL", groups["ALL"]),
        },

        "notes": {
            "confidence_priority": [
                "diagnostics.prompt_and_encoding_debug.cand_conf_probe.p_cond[ANS]",
                "gen_token_probs[0]",
            ],
            "unpredictable_definition": ["NP", "PN", "PP"],
            "predictable_definition": ["NN"],
            "four_group_definition": {
                "NN": "predictable",
                "PN": "branch_only",
                "NP": "surprise_only",
                "PP": "branch_and_surprise"
            },
            "extra_report": "mean confidence over WRONG predictions within UNP group is groups.UNP.conf_mean_over_wrong",
        },

        "timestamp_unix": time.time(),
    }
    for k, v in summary["groups"]["ALL"].items():
        summary.setdefault(k, v)

    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    global MODEL_NAME, PRED_DIR, OUT_DIR, OUT_SUMMARY, OUT_DETAILS

    parser = argparse.ArgumentParser(description="Evaluate sh_pred cand_conf and source statistics.")
    parser.add_argument("--model_name", default=None, help="Override MODEL_NAME without changing the environment.")
    parser.add_argument("--runs_root", type=Path, default=None, help="Infer pred_dir as <runs_root>/<model_name>/sh_pred/<flavor>.")
    parser.add_argument("--pred_dir", type=Path, default=None, help="Override prediction directory.")
    parser.add_argument("--out_dir", type=Path, default=None, help="Override output directory.")
    parser.add_argument("--flavor", default="cand_conf", help="Prediction flavor for --runs_root inference; default: cand_conf.")
    args = parser.parse_args()

    if args.model_name:
        MODEL_NAME = str(args.model_name).strip()
    flavor = str(args.flavor or "cand_conf").strip()
    if args.pred_dir is not None:
        PRED_DIR = args.pred_dir.expanduser()
    elif args.runs_root is not None:
        PRED_DIR = args.runs_root.expanduser() / MODEL_NAME / "sh_pred" / flavor
    else:
        PRED_DIR = Path.home() / "benchmark_val" / "testllm" / MODEL_NAME / "sh_pred" / "cand_conf"

    if args.out_dir is not None:
        OUT_DIR = args.out_dir.expanduser()
    else:
        OUT_DIR = Path.home() / "benchmark_val" / "score" / MODEL_NAME / "sh_pred_cand_conf"

    OUT_SUMMARY = OUT_DIR / "summary.json"
    OUT_DETAILS = OUT_DIR / "details.jsonl"

    if not PRED_DIR.exists():
        raise FileNotFoundError(f"PRED_DIR not found: {PRED_DIR}")
    _ensure_dir(OUT_DIR)

    print("\n================ SH_PRED CAND_CONF EVAL ================\n")
    print(f"MODEL_NAME   : {MODEL_NAME}")
    print(f"PRED_DIR     : {PRED_DIR}")
    print(f"OUT_DIR      : {OUT_DIR}")
    print("")

    s = eval_sh_pred_cand_conf()

    print(f"\n[WROTE] {OUT_SUMMARY}")
    print(f"[WROTE] {OUT_DETAILS}")
    print("")

    gNN = s["groups"]["NN_predictable"]
    gU = s["groups"]["UNP_unpredictable_NP_PN_PP"]

    print("---- NN (predictable) ----")
    print(f"Valid : {gNN['samples_valid']} / {gNN['samples_total']} | Acc(valid)={gNN['accuracy_over_valid']:.4f}")
    print(f"Conf(valid mean)   : {gNN['conf_mean_over_valid']} (n={gNN['conf_count_over_valid']})")
    print(f"Conf(correct mean) : {gNN['conf_mean_over_correct']} (n={gNN['conf_count_over_correct']})")
    print("")

    print("---- NP/PN/PP (unpredictable) ----")
    print(f"Valid : {gU['samples_valid']} / {gU['samples_total']} | Acc(valid)={gU['accuracy_over_valid']:.4f}")
    print(f"Conf(valid mean)   : {gU['conf_mean_over_valid']} (n={gU['conf_count_over_valid']})")
    print(f"Conf(correct mean) : {gU['conf_mean_over_correct']} (n={gU['conf_count_over_correct']})")
    print(f"Conf(wrong mean)   : {gU['conf_mean_over_wrong']} (n={gU['conf_count_over_wrong']})")
    print("")

    print("---- Source stats overall (valid-only) ----")
    for line in source_stats_report_lines(s):
        print(line)
    print("")

    print("[DONE]\n")


if __name__ == "__main__":
    main()
