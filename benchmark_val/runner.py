#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runner for Ego4D NOW-style tasks (GT-driven).

Pipeline:
  GT (.json/.jsonl, single record) ->
  find source video: <video_uid>.mp4 under VIDEO_ROOT ->
  CUT interval clip using interval_start_sec/interval_end_sec ->
  run strict-online task on the interval clip (t_eval is RELATIVE to interval unless helper decides otherwise).

Default Paths (open-source wrapper):
  GT root:
    <PROJECT_ROOT>/examples/gt/<TASK>/{train,val}/
  Video root:
    provided by VIDEO_ROOT, usually an Ego4D full_scale directory containing <video_uid>.mp4
  Runs root:
    <PROJECT_ROOT>/outputs/runs/<MODEL_NAME>/<TASK>/<flavor>/

Env:
  MODEL_NAME: required (used for output folder naming)
  TASK: task name (default: now_narration). Example: now_narration, now_state_switch, sh_rtrv, ms_rtrv, ms_rtrv, ms_pred, sh_pred
  SPLIT: all|train|val (default: val)
  GT_ROOT: GT root containing {train,val}/ (default: <PROJECT_ROOT>/examples/gt/<TASK>)
  VIDEO_ROOT: Ego4D video root containing <video_uid>.mp4 (wrapper should set this)
  RUNS_ROOT: runs root (default: <PROJECT_ROOT>/outputs/runs)
  ADAPTER_PY: path to adapter python file (default: <PROJECT_ROOT>/benchmark_val/llms/llm_adapter.py)
  HELPER_PY: path to helper python file (default: <PROJECT_ROOT>/benchmark_val/helper.py)

  DRY_RUN: 1 => print only
  MAX_ITEMS: optional int

IMPORTANT:
  This runner forces NOW_TIME_MODE=clip unless you explicitly set it.

Update (per request):
  - Always use HELPER to build the effective queryset.
    * In open mode: helper provides the open-template prompt.
    * In candidate mode (sh_rtrv/ms_rtrv/ms_pred/sh_pred): helper reads mcq_shuffled and constructs prompts from its options.
  - Runner no longer builds prompts by itself and no longer resolves MCQ here.

Additional update (state-switch scan support):
  - Add CLI --ss_pos (REQUIRED only when TASK==now_state_switch).
  - Pass it to helper via env (SS_POS / NOW_SS_POS / NOW_STATE_SWITCH_POS).
  - Encode ss_pos into output directory for reproducibility and to allow multiple scans.

Update (per request - stop-after-this-file):
  - After each successful inference, read effective queryset's `helper.stop_after_this_file`.
  - If True, break (stop entering next GT, but current GT has already finished).

NEW (per request - now_narration cand mode):
  - If TASK==now_narration and --mode cand:
      run TWO passes per GT file:
        1) cand_state  (STATE-only)
        2) cand_mcq    (ACTION MCQ, segment-only)
    outputs go to:
      <RUNS_ROOT>/<MODEL_NAME>/now_narration/cand_state/
      <RUNS_ROOT>/<MODEL_NAME>/now_narration/cand_mcq/
    Stop-after-this-file is decided by the STATE pass, but runner will still finish the MCQ pass for the same GT.

NEW (per request - now_state_switch cand_state):
  - If TASK==now_state_switch:
      --mode open       -> ["open"]
      --mode cand/state -> ["cand_state"]
    (still requires --ss_pos)

[MODIFICATION for debugging]
  - Print full traceback on failure (including adapter stack) and record it in manifest.
"""

import os
import sys
import json
import time
import shutil
import subprocess
import importlib.util
import argparse
import hashlib
import random
import traceback  # <<< NEW
from pathlib import Path
from typing import Any, Dict, List, Tuple


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _normalize_task_name(task: str) -> str:
    t = (task or "").strip().lower()
    if not t:
        return "now_narration"
    # aliases
    if t in {"now-narration", "now"}:
        return "now_narration"
    if t in {"now-state-switch", "state_switch", "now_switch"}:
        return "now_state_switch"
    if t in {"sh_rtrv", "sh-rtrv", "past_retrieval", "past-retrieval"}:
        return "sh_rtrv"
    if t in {"ms_rtrv", "ms-rtrv", "multistep_past_retrieval", "multistep-past-retrieval"}:
        return "ms_rtrv"
    # NEW: ms_pred aliases
    if t in {"ms_pred", "ms-pred", "multistep_prediction", "multistep-prediction", "multistep_future_prediction", "multistep-future-prediction"}:
        return "ms_pred"
    # NEW: sh_pred aliases
    if t in {
        "sh_pred", "sh-pred",
        "short_horizon_prediction", "short-horizon-prediction",
        "short_horizon_pred", "short-horizon-pred",
        "short_prediction", "short-prediction",
        "sh_pred_full", "sh-pred-full",
    }:
        return "sh_pred"
    return t


def safe_load_json_or_one_jsonl(path: Path) -> Dict[str, Any]:
    """
    Accept:
      - .json: single dict
      - .jsonl: must contain exactly ONE JSON object (either single-line or multi-line but 1 record)
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"GT not found: {p}")

    raw = p.read_text(encoding="utf-8").strip()
    if not raw:
        # try utf-8-sig
        raw = p.read_text(encoding="utf-8-sig").strip()
    if not raw:
        raise ValueError(f"Empty file: {p}")

    if p.suffix.lower() == ".jsonl":
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        recs = [json.loads(ln) for ln in lines]
        if len(recs) != 1 or not isinstance(recs[0], dict):
            raise ValueError(f"Expected exactly 1 JSON object in JSONL: {p}, got {len(recs)}")
        return recs[0]

    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict JSON: {p}")
    return obj


def load_adapter_from_py(adapter_py: Path):
    adapter_py = adapter_py.expanduser().resolve()
    if not adapter_py.exists():
        raise FileNotFoundError(f"ADAPTER_PY not found: {adapter_py}")
    spec = importlib.util.spec_from_file_location("adapter_module", str(adapter_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load adapter spec: {adapter_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "create_adapter"):
        raise RuntimeError(f"Adapter must define create_adapter(): {adapter_py}")
    adapter = mod.create_adapter()
    if not hasattr(adapter, "run"):
        raise RuntimeError(f"Adapter instance must have .run(...), got {type(adapter)}")
    return adapter


def load_helper_from_py(helper_py: Path):
    helper_py = helper_py.expanduser().resolve()
    if not helper_py.exists():
        raise FileNotFoundError(f"HELPER_PY not found: {helper_py}")
    spec = importlib.util.spec_from_file_location("helper_module", str(helper_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load helper spec: {helper_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "create_helper"):
        raise RuntimeError(f"Helper must define create_helper(): {helper_py}")
    helper = mod.create_helper()
    if not hasattr(helper, "prepare_queryset"):
        raise RuntimeError(f"Helper instance must have .prepare_queryset(...), got {type(helper)}")
    return helper


def _run_ffmpeg(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (rc={p.returncode}).\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDERR(last 4000 chars):\n{p.stderr[-4000:]}"
        )


def cut_interval_clip(
    *,
    src_video: Path,
    dst_clip: Path,
    start_sec: float,
    end_sec: float,
) -> None:
    """
    Make an interval clip [start_sec, end_sec] from src_video.
    Re-encode to avoid keyframe seeking issues.
    """
    ensure_dir(dst_clip.parent)

    start_sec = float(max(0.0, start_sec))
    end_sec = float(max(start_sec, end_sec))
    dur = end_sec - start_sec

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.6f}",
        "-i", str(src_video),
        "-t", f"{dur:.6f}",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(dst_clip),
    ]
    _run_ffmpeg(cmd)


def resolve_source_video(video_root: Path, video_uid: str) -> Path:
    """
    Primary: VIDEO_ROOT/<video_uid>.mp4
    Fallback: search recursively under VIDEO_ROOT.
    """
    p = video_root / f"{video_uid}.mp4"
    if p.exists():
        return p

    hits = list(video_root.rglob(f"{video_uid}.mp4"))
    if hits:
        hits.sort(key=lambda x: x.as_posix())
        return hits[0]

    raise FileNotFoundError(f"Cannot find source video for uid={video_uid} under {video_root}")


def list_gt_files(gt_root: Path, split: str) -> List[Path]:
    split = (split or "val").lower()
    if split == "all":
        bases = [gt_root / "train", gt_root / "val"]
    elif split in {"train", "val"}:
        bases = [gt_root / split]
    else:
        raise ValueError(f"Unsupported SPLIT: {split}")

    files: List[Path] = []
    for b in bases:
        if b.exists():
            files += [p for p in b.rglob("*") if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}]
    files = sorted(files, key=lambda x: x.as_posix())
    return files


# ---------------------------
# sh_rtrv additions (kept; runner no longer uses them to build prompts)
# ---------------------------

def _normalize_mode(x: str) -> str:
    t = (x or "").strip().lower()
    if t in {"cand", "candidate", "mcq"}:
        return "cand"
    return "open"


def _stable_seed(*parts: str) -> int:
    s = "||".join([p or "" for p in parts]).encode("utf-8")
    h = hashlib.md5(s).hexdigest()
    return int(h[:8], 16)


# ---------------------------
# NEW: skip empty GT instead of failing
# ---------------------------

def _gt_is_empty(gt: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Return (is_empty, reason).
    Empty means: num_samples==0 OR samples is an empty list.
    """
    try:
        ns = gt.get("num_samples", None)
        if isinstance(ns, (int, float)) and int(ns) == 0:
            return True, "num_samples==0"
    except Exception:
        pass

    s = gt.get("samples", None)
    if isinstance(s, list) and len(s) == 0:
        return True, "samples==[]"
    return False, ""


# ---------------------------
# NEW: ss_pos helpers (state-switch scan)
# ---------------------------

def _sanitize_for_path(x: str) -> str:
    """
    Make a safe, stable folder suffix. Keep [a-zA-Z0-9._-], map others to '_'.
    """
    s = (x or "").strip()
    if not s:
        return "empty"
    out = []
    for ch in s:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    y = "".join(out)
    while "__" in y:
        y = y.replace("__", "_")
    return y.strip("_") or "empty"


# ---------------------------
# NEW: stop-after-this-file flag (read from effective queryset)
# ---------------------------

def _should_stop_after_file(effective_qs_path: Path) -> bool:
    """
    Read effective queryset JSON and check:
      obj["helper"]["stop_after_this_file"] (preferred)
    Fallbacks:
      obj["stop_after_this_file"]
    Accept bool/int/str.
    """
    try:
        obj = safe_load_json_or_one_jsonl(effective_qs_path)
    except Exception:
        return False

    v = None
    if isinstance(obj, dict):
        h = obj.get("helper", None)
        if isinstance(h, dict) and "stop_after_this_file" in h:
            v = h.get("stop_after_this_file", None)
        elif "stop_after_this_file" in obj:
            v = obj.get("stop_after_this_file", None)

    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return int(v) != 0
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "t"}
    return False


# ---------------------------
# NEW: runner-side now_narration cand plan
# ---------------------------

def _now_narration_plan_from_mode(raw_mode: str) -> List[str]:
    """
    Runner interpretation:
      --mode open        -> ["open"]
      --mode cand        -> ["cand_state", "cand_mcq"]
      --mode cand_state  -> ["cand_state"]
      --mode cand_mcq    -> ["cand_mcq"]
    """
    m = (raw_mode or "").strip().lower()
    if not m or m == "open":
        return ["open"]
    if m in {"cand", "candidate"}:
        return ["cand_state", "cand_mcq"]
    if m in {"cand_state", "state", "state_only", "state-only", "candstate"}:
        return ["cand_state"]
    if m in {"cand_mcq", "mcq", "action", "action_mcq", "action-mcq", "candmcq"}:
        return ["cand_mcq"]
    # fallback: keep old normalize
    if _normalize_mode(m) == "cand":
        return ["cand_state", "cand_mcq"]
    return ["open"]


# ---------------------------
# NEW: runner-side now_state_switch plan (open / cand_state)
# ---------------------------

def _now_state_switch_plan_from_mode(raw_mode: str) -> List[str]:
    """
    Runner interpretation for now_state_switch:
      --mode open                 -> ["open"]
      --mode cand / cand_state    -> ["cand_state"]
    """
    m = (raw_mode or "").strip().lower()
    if not m or m == "open":
        return ["open"]
    if m in {"cand_state", "state", "state_only", "state-only", "candstate", "cand", "candidate", "mcq"}:
        return ["cand_state"]
    # fallback: if legacy normalizer says cand, map to cand_state
    if _normalize_mode(m) == "cand":
        return ["cand_state"]
    return ["open"]


def main():
    # -------- minimal CLI mode selector (as requested) --------
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", type=str, default=os.environ.get("PRED_FLAVOR", "open"))
    # REQUIRED only when TASK==now_state_switch
    parser.add_argument("--ss_pos", type=str, default="")
    args, _unknown = parser.parse_known_args()

    raw_mode = args.mode
    mode_norm = _normalize_mode(args.mode)  # open/cand (legacy)

    model_name = os.environ.get("MODEL_NAME", "").strip()
    if not model_name:
        raise RuntimeError("Please set env MODEL_NAME, e.g. export MODEL_NAME=openrouter_gpt4o")

    task = _normalize_task_name(os.environ.get("TASK", "now_narration"))
    split = os.environ.get("SPLIT", "val").strip().lower()
    dry_run = os.environ.get("DRY_RUN", "0").strip() == "1"
    max_items = os.environ.get("MAX_ITEMS", "").strip()
    max_items = int(max_items) if max_items else None

    # ---------------------------
    # NEW: support nonstandard candidate flavors (e.g. cand_conf) WITHOUT being skipped.
    # - For ms_* candidate tasks, output flavor can be cand_conf, but helper should still build candidate prompts.
    # ---------------------------
    candidate_tasks = {"sh_rtrv", "ms_rtrv", "ms_pred", "sh_pred"}
    env_pred_flavor = os.environ.get("PRED_FLAVOR", "").strip().lower()

    # NEW: choose per-task plan
    if task == "now_narration":
        plan_flavors = _now_narration_plan_from_mode(raw_mode)
    elif task == "now_state_switch":
        plan_flavors = _now_state_switch_plan_from_mode(raw_mode)
    elif task in candidate_tasks:
        # If user set PRED_FLAVOR to a nonstandard flavor (e.g. cand_conf) AND is running cand mode,
        # use that as the output flavor to avoid colliding with existing /cand/ files.
        if mode_norm == "cand" and env_pred_flavor and env_pred_flavor not in {"open", "cand"}:
            plan_flavors = [env_pred_flavor]
        else:
            plan_flavors = [mode_norm]
    else:
        plan_flavors = ["open"]

    # NEW: enforce --ss_pos only for now_state_switch, and pass via env for helper to read.
    ss_pos_raw = ""
    ss_pos_sanitized = ""
    if task == "now_state_switch":
        ss_pos_raw = (args.ss_pos or "").strip()
        if not ss_pos_raw:
            raise RuntimeError(
                "TASK==now_state_switch requires CLI --ss_pos.\n"
                "Examples:\n"
                "  python runner.py --ss_pos r:0.25\n"
                "  python runner.py --ss_pos t:2.0\n"
            )
        ss_pos_sanitized = _sanitize_for_path(ss_pos_raw)

        os.environ["SS_POS"] = ss_pos_raw
        os.environ["NOW_SS_POS"] = ss_pos_raw
        os.environ["NOW_STATE_SWITCH_POS"] = ss_pos_raw

    here = Path(__file__).resolve().parent
    project_root = here.parent
    default_gt_root = str(project_root / "examples" / "gt" / task)
    gt_root = Path(os.environ.get("GT_ROOT", default_gt_root)).expanduser()
    video_root = Path(os.environ.get("VIDEO_ROOT", str(project_root / "videos"))).expanduser()
    runs_root = Path(os.environ.get("RUNS_ROOT", str(project_root / "outputs" / "runs"))).expanduser()

    default_helper_py = here / "helper.py"
    default_adapter_py = here / "llms" / "llm_adapter.py"

    helper_py = Path(os.environ.get("HELPER_PY", str(default_helper_py))).expanduser()
    adapter_py = Path(os.environ.get("ADAPTER_PY", str(default_adapter_py))).expanduser()

    # FORCE clip time mode by default (because runner CUTS interval clip)
    if not os.environ.get("NOW_TIME_MODE", "").strip():
        os.environ["NOW_TIME_MODE"] = "clip"

    gt_files = list_gt_files(gt_root, split)
    if max_items is not None:
        gt_files = gt_files[:max_items]

    if not gt_files:
        eprint(f"[WARN] No GT files found under {gt_root} (split={split})")
        return

    # NEW: multi-output dirs by flavor
    task_root = runs_root / model_name / task
    ensure_dir(task_root)

    out_dirs: Dict[str, Path] = {}
    manifests: Dict[str, Path] = {}

    for flav in plan_flavors:
        out_dir = task_root / flav
        if task == "now_state_switch":
            out_dir = out_dir / f"sspos_{ss_pos_sanitized}"
        ensure_dir(out_dir)
        out_dirs[flav] = out_dir
        manifests[flav] = out_dir / "manifest.jsonl"

    # staging shared across flavors (avoid duplicating temp)
    tmp_dir = task_root / "_staging"
    ensure_dir(tmp_dir)

    video_cache_dir = runs_root / "_video_cache" / task / split
    ensure_dir(video_cache_dir)

    helper = load_helper_from_py(helper_py)
    adapter = load_adapter_from_py(adapter_py)

    eprint(f"[INFO] MODEL_NAME={model_name}")
    eprint(f"[INFO] TASK={task}")
    eprint(f"[INFO] SPLIT={split}")
    eprint(f"[INFO] MODE(raw)={raw_mode}")
    eprint(f"[INFO] PLAN={plan_flavors}")
    if task == "now_state_switch":
        eprint(f"[INFO] SS_POS={ss_pos_raw} (sanitized={ss_pos_sanitized})")
        eprint(f"[INFO] SS_POS env -> SS_POS / NOW_SS_POS / NOW_STATE_SWITCH_POS")
    eprint(f"[INFO] GT_ROOT={gt_root}")
    eprint(f"[INFO] VIDEO_ROOT={video_root}")
    eprint(f"[INFO] RUNS_ROOT={runs_root}")
    eprint(f"[INFO] HELPER_PY={helper_py}")
    eprint(f"[INFO] ADAPTER_PY={adapter_py}")
    eprint(f"[INFO] NOW_TIME_MODE={os.environ.get('NOW_TIME_MODE')}")
    eprint(f"[INFO] Total GT files: {len(gt_files)}")
    for flav in plan_flavors:
        eprint(f"[INFO] OUTPUT_DIR[{flav}]={out_dirs[flav]}")

    processed = skipped = failed = 0

    for i, gt_path in enumerate(gt_files, 1):
        stop_after_this_gt = False

        try:
            gt = safe_load_json_or_one_jsonl(gt_path)

            video_uid = str(gt.get("video_uid", "")).strip()
            meta = gt.get("video_metadata", {}) if isinstance(gt.get("video_metadata"), dict) else {}
            clip_uid = str(meta.get("clip_uid", "")).strip()
            clip_id = str(meta.get("clip_id", "")).strip()
            interval_start = float(meta.get("interval_start_sec", 0.0))
            interval_end = float(meta.get("interval_end_sec", interval_start))

            if not video_uid:
                raise ValueError(f"Missing video_uid in GT: {gt_path}")

            is_empty, empty_reason = _gt_is_empty(gt)
            if is_empty:
                skipped += 1
                for flav in plan_flavors:
                    rec = {
                        "idx": i,
                        "gt": str(gt_path),
                        "task": task,
                        "mode": flav,
                        "model": model_name,
                        "video_uid": video_uid,
                        "interval_start_sec": interval_start,
                        "interval_end_sec": interval_end,
                        "status": "skipped_empty_gt",
                        "reason": empty_reason,
                    }
                    if task == "now_state_switch":
                        rec["ss_pos"] = ss_pos_raw
                        rec["ss_pos_sanitized"] = ss_pos_sanitized
                    with open(manifests[flav], "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                eprint(f"[{i}/{len(gt_files)}] SKIP_EMPTY_GT ({empty_reason}): {gt_path.name}")
                continue

            src_video = resolve_source_video(video_root, video_uid)

            clip_tag = f"{video_uid}__{clip_id or 'NA'}__{clip_uid or 'NA'}_{interval_start:.3f}-{interval_end:.3f}.mp4"
            interval_clip = video_cache_dir / clip_tag

            if not interval_clip.exists():
                if dry_run:
                    eprint(f"[{i}/{len(gt_files)}] DRY_RUN: would cut interval clip -> {interval_clip.name}")
                else:
                    cut_interval_clip(
                        src_video=src_video,
                        dst_clip=interval_clip,
                        start_sec=interval_start,
                        end_sec=interval_end,
                    )

            for flav in plan_flavors:
                out_dir = out_dirs[flav]
                manifest_path = manifests[flav]

                # ---------------------------
                # NEW: In cand_conf (and other nonstandard cand flavors), helper must still run in "cand" mode
                # so that it constructs candidate prompts, while adapter sees the real flavor (cand_conf).
                # ---------------------------
                helper_flav = flav
                if task in candidate_tasks and mode_norm == "cand" and flav not in {"open", "cand"}:
                    helper_flav = "cand"

                # ---- helper pass ----
                os.environ["PRED_FLAVOR"] = helper_flav

                derived_qs = tmp_dir / f"{gt_path.stem}__derived_{task}__{flav}.json"
                effective_qs = Path(
                    helper.prepare_queryset(
                        queryset_path=gt_path,
                        out_path=derived_qs,
                        task=task,
                    )
                )
                if not effective_qs.exists():
                    raise FileNotFoundError(f"Helper returned path but file not found: {effective_qs}")

                staged_out = tmp_dir / f"{gt_path.stem}__pred__{flav}.json"
                final_out = out_dir / f"{model_name}__{staged_out.name}"

                if task == "now_narration" and flav in {"open", "cand_state", "cand_mcq"}:
                    if _should_stop_after_file(effective_qs):
                        stop_after_this_gt = True

                if final_out.exists():
                    skipped += 1
                    rec = {
                        "idx": i,
                        "gt": str(gt_path),
                        "effective_queryset": str(effective_qs),
                        "video_uid": video_uid,
                        "interval_start_sec": interval_start,
                        "interval_end_sec": interval_end,
                        "interval_clip": str(interval_clip),
                        "task": f"{task}_{flav}",
                        "model": model_name,
                        "status": "skipped_exists",
                        "output": str(final_out),
                    }
                    if task == "now_state_switch":
                        rec["ss_pos"] = ss_pos_raw
                        rec["ss_pos_sanitized"] = ss_pos_sanitized
                    with open(manifest_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    eprint(f"[{i}/{len(gt_files)}] SKIP (exists) [{flav}]: {final_out.name}")
                    continue

                if dry_run:
                    eprint(f"[{i}/{len(gt_files)}] DRY_RUN [{flav}]:")
                    eprint(f"  gt: {gt_path}")
                    eprint(f"  video_uid: {video_uid}")
                    eprint(f"  src_video: {src_video}")
                    eprint(f"  interval: [{interval_start}, {interval_end}] -> {interval_clip}")
                    eprint(f"  effective_qs: {effective_qs}")
                    if task == "now_state_switch":
                        eprint(f"  ss_pos: {ss_pos_raw} (sanitized={ss_pos_sanitized})")
                    eprint(f"  staged_out: {staged_out}")
                    eprint(f"  final_out:  {final_out}")
                    continue

                # ---- adapter pass ----
                os.environ["PRED_FLAVOR"] = flav

                t0 = time.time()
                produced = adapter.run(
                    video_path=interval_clip,
                    queryset_path=effective_qs,
                    pred_flavor=flav,
                    out_json_path=staged_out,
                )
                dt = time.time() - t0
                produced = Path(produced)

                if not produced.exists():
                    raise FileNotFoundError(f"Adapter returned output but missing: {produced}")

                shutil.move(str(produced), str(final_out))

                rec = {
                    "idx": i,
                    "gt": str(gt_path),
                    "effective_queryset": str(effective_qs),
                    "video_uid": video_uid,
                    "src_video": str(src_video),
                    "interval_start_sec": interval_start,
                    "interval_end_sec": interval_end,
                    "interval_clip": str(interval_clip),
                    "task": f"{task}_{flav}",
                    "model": model_name,
                    "helper_py": str(helper_py),
                    "adapter_py": str(adapter_py),
                    "output": str(final_out),
                    "status": "ok",
                    "elapsed_sec": round(dt, 3),
                }
                if task == "now_state_switch":
                    rec["ss_pos"] = ss_pos_raw
                    rec["ss_pos_sanitized"] = ss_pos_sanitized
                with open(manifest_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                processed += 1
                eprint(f"[{i}/{len(gt_files)}] OK [{flav}] -> {final_out.name} ({dt:.2f}s)")

                if task != "now_narration" and _should_stop_after_file(effective_qs):
                    eprint(f"[INFO] helper.stop_after_this_file=True -> STOP AFTER THIS FILE: {gt_path.name}")
                    stop_after_this_gt = True
                    break

            if stop_after_this_gt:
                eprint(f"[INFO] STOP AFTER THIS FILE (after finishing planned passes): {gt_path.name}")
                break

        except Exception as e:
            # <<< NEW: capture full traceback (includes adapter stack) >>>
            tb = traceback.format_exc()

            msg = str(e)

            if "No usable samples found in GT" in msg or "no usable samples" in msg.lower():
                skipped += 1
                for flav in plan_flavors:
                    rec = {
                        "idx": i,
                        "gt": str(gt_path),
                        "task": task,
                        "mode": flav,
                        "model": model_name,
                        "status": "skipped_empty_gt",
                        "reason": "helper_reported_no_usable_samples",
                        "error": repr(e),
                        "traceback": tb,  # <<< NEW (still useful)
                    }
                    if task == "now_state_switch":
                        rec["ss_pos"] = ss_pos_raw
                        rec["ss_pos_sanitized"] = ss_pos_sanitized
                    with open(manifests[flav], "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                eprint(f"[{i}/{len(gt_files)}] SKIP_EMPTY_GT (helper): {gt_path.name}\n  {e}\n")
                eprint(tb)  # <<< NEW
                continue

            failed += 1
            for flav in plan_flavors:
                rec = {
                    "idx": i,
                    "gt": str(gt_path),
                    "task": task,
                    "mode": flav,
                    "model": model_name,
                    "status": "failed",
                    "error": repr(e),
                    "traceback": tb,  # <<< NEW
                }
                if task == "now_state_switch":
                    rec["ss_pos"] = ss_pos_raw
                    rec["ss_pos_sanitized"] = ss_pos_sanitized
                with open(manifests[flav], "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # <<< NEW: print full traceback to stderr >>>
            eprint(f"[{i}/{len(gt_files)}] FAIL: {gt_path.name}\n  {e}\n")
            eprint(tb)

    eprint(f"\n[SUMMARY] processed={processed}, skipped={skipped}, failed={failed}")
    for flav in plan_flavors:
        eprint(f"[SUMMARY] manifest[{flav}]={manifests[flav]}")


if __name__ == "__main__":
    main()
