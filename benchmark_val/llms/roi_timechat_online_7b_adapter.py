#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI-TimeChat-Online adapter (Scheme A: image-based) for runner+helper pipeline.

Key idea (to guarantee ROI alignment):
  - Do NOT use "video" item + process_vision_info(...) because internal frame sampling is opaque.
  - Instead:
      1) Downsample the *interval clip* to 1FPS once (ffmpeg) -> cached mp4
      2) For each sample, compute explicit 1FPS frame indices in [ws_rel, we_rel] (strict-online)
      3) Load frames as PIL images in the SAME order as indices
      4) Build roi_cache["frames"] in the SAME order as indices
      5) Call model.generate(..., roi_cache=roi_cache, drop_method=roi_*, ...)

This matches your standalone ego_streaming_timechatonline.py behavior.

Contract:
  - create_adapter() -> instance with .run(...)
  - run(video_path, queryset_path, out_json_path, ...) writes a unified JSON:
      {
        dataset, task, video_uid, model_name, model_id, source_queryset,
        video_clip, generated_at_unix, params, samples:[...]
      }

Env knobs (compatible with your existing runner command; plus ROI ones):
  MODEL_NAME
  TIMECHAT_REPO_ROOT                  # MUST be set to TimeChat-Online-main root
  TIMECHAT_HF_MODEL_ID                # default wyccccc/TimeChatOnline-7B
  TIMECHAT_DEVICE                     # cuda|cpu
  TIMECHAT_ATTN_IMPL                  # flash_attention_2|sdpa|eager (try, fallback sdpa)
  TIMECHAT_MAX_FRAMES                 # default 64
  TIMECHAT_SDPA_MAX_FRAMES            # default 6
  MAX_NEW_TOKENS                      # default 128
  TEMPERATURE / TOP_P
  RETURN_LOGPROBS                     # default 1
  STOP_ON_CLOSE_TAGS                  # default 1
  DEANCHOR_ANS                        # default 1
  CAND_BAD_WORDS
  ENABLE_RETRY                        # default 1
  MAX_RETRIES                         # default 1
  RESUME                              # default 1

  # DTD knobs (your ROI-aware DTD accepts these)
  TIMECHAT_DROP_METHOD                # default roi_feature
  TIMECHAT_DROP_THRESHOLD             # default 0.85
  TIMECHAT_DROP_ABSOLUTE              # default 1
  TIMECHAT_REQUIRE_DTD                # default 1 (raise if generate() doesn't accept drop_*/roi_cache)
  TIMECHAT_SAVE_DR                    # default 0
  TIMECHAT_DR_SAVE_DIR                # default cache/timechat_dr

  # ROI cache locating
  ROI_CACHE_PATH                      # explicit path (highest priority)
  ROI_CACHE_ROOT                      # required unless ROI_CACHE_PATH is set
  ROI_COORD                           # norm|pixel (default norm)
  GAZE_R_DEFAULT                       # default 0.08 (norm radius fallback)

  # NEW (ablation): if set, overwrite gaze to a center circle (radius in normalized coord)
  ROI_GAZE_CENTER_R                   # if not set -> keep original gaze; if set -> use center-circle gaze

  # 1FPS cache
  TIMECHAT_FPS1_CACHE_DIR             # default cache/timechat_fps1

  # NEW: finetune loading switch
  TIMECHAT_FT_DIR                     # if set, load LoRA adapter from this dir (e.g., .../final/step_0000706)
  TIMECHAT_MERGE_LORA                 # default 0; if 1, merge LoRA into base for faster inference
  TIMECHAT_FT_VERBOSE                 # default 1; print load summary
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from peft import PeftModel  # already imported; now actually used

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, StoppingCriteria, StoppingCriteriaList

from decord import VideoReader, cpu  # type: ignore
from PIL import Image


# -------------------------
# Small utilities
# -------------------------
def _env(key: str, default: str) -> str:
    v = os.environ.get(key, "").strip()
    return v if v else default


def _env_int(key: str, default: int) -> int:
    try:
        return int(float(_env(key, str(default))))
    except Exception:
        return int(default)


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except Exception:
        return float(default)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _load_json_or_one_jsonl(path: Path) -> Dict[str, Any]:
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

    # fallback jsonl: exactly one record
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    recs = [json.loads(ln) for ln in lines]
    if len(recs) != 1 or not isinstance(recs[0], dict):
        raise ValueError(f"Expected exactly 1 JSON object in JSONL: {p}, got {len(recs)}")
    return recs[0]


def _dump_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _guess_task(qs: Dict[str, Any]) -> str:
    t = (qs.get("task_name") or qs.get("task") or "").strip().lower()
    return t or "unknown"


def _first_tag_block(text: str) -> str:
    if not text:
        return text
    s = text.strip()
    for tag in ("NOW", "PAST", "FUTURE"):
        m = re.search(rf"(<{tag}>.*?</{tag}>)", s, flags=re.DOTALL | re.IGNORECASE)
        if m:
            blk = m.group(1)
            blk = re.sub(r"\s+", " ", blk).strip()
            return blk
    return s.splitlines()[0].strip() if s.splitlines() else s


def _parse_xmlish(block: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not block:
        return out
    for k in ("STATE", "VERB", "NOUN", "DESC", "CONF", "ANS"):
        m = re.search(rf"<{k}>(.*?)</{k}>", block, flags=re.IGNORECASE | re.DOTALL)
        if m:
            out[k.lower()] = m.group(1).strip()
    mroot = re.search(r"<(NOW|PAST|FUTURE)>", block, flags=re.IGNORECASE)
    if mroot:
        out["root"] = mroot.group(1).upper()
    return out


def _ensure_repo_on_syspath() -> None:
    """
    Ensure TimeChat-Online-main is importable as a Python package root,
    so `import eval.qwen2_5_vl...` works even if adapter is outside repo.
    """
    repo_root = os.environ.get("TIMECHAT_REPO_ROOT", "").strip()
    if not repo_root:
        return
    rr = str(Path(repo_root).expanduser().resolve())
    if rr not in sys.path:
        sys.path.insert(0, rr)


# -------------------------
# Stop-on-close-tags
# -------------------------
class _StopOnCloseTags(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len: int, stop_strs_upper: List[str], max_decode_tokens: int = 96):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_len = int(max(0, prompt_len))
        self.stop_strs_upper = [s.upper() for s in (stop_strs_upper or []) if s]
        self.max_decode_tokens = int(max(16, max_decode_tokens))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.tokenizer is None or not self.stop_strs_upper:
            return False
        try:
            seq = input_ids[0]
            if seq.numel() <= self.prompt_len:
                return False
            gen = seq[self.prompt_len:]
            if gen.numel() > self.max_decode_tokens:
                gen = gen[-self.max_decode_tokens:]
            txt = self.tokenizer.decode(gen.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            up = (txt or "").upper()
            for s in self.stop_strs_upper:
                if s in up:
                    return True
            return False
        except Exception:
            return False


def _build_bad_words_ids(tokenizer, csv: str) -> Optional[List[List[int]]]:
    if tokenizer is None:
        return None
    parts = [p.strip() for p in (csv or "").split(",") if p.strip()]
    if not parts:
        return None
    bad: List[List[int]] = []
    for s in parts:
        try:
            ids = tokenizer.encode(s, add_special_tokens=False)
        except Exception:
            ids = []
        if ids:
            bad.append([int(x) for x in ids])
    return bad if bad else None


def _deanchor_ans_template(prompt: str) -> Tuple[str, bool]:
    if not prompt:
        return prompt, False
    p2 = re.sub(r"<\s*ANS\s*>\s*A\s*<\s*/\s*ANS\s*>", "<ANS>choice_here</ANS>", prompt, flags=re.IGNORECASE)
    return p2, (p2 != prompt)


def _infer_kind_and_schema(prompt: str) -> Tuple[str, str, bool]:
    s = (prompt or "")
    up = s.upper()

    cand = ("<ANS>" in up) or bool(re.search(r"^\s*[A-D]\.\s+", s, flags=re.MULTILINE))

    if "<NOW>" in up or "</NOW>" in up:
        has_ans = "<ANS>" in up and "</ANS>" in up
        has_state = "<STATE>" in up
        has_verb = "<VERB>" in up
        has_noun = "<NOUN>" in up
        has_desc = "<DESC>" in up
        has_conf = "<CONF>" in up
        if has_ans and has_verb and has_noun and has_conf and (not has_state) and (not has_desc):
            return "now", "now_action_mcq", True
        if has_state and has_conf and (not has_verb) and (not has_noun) and (not has_desc):
            return "now", "now_state_only", False
        return "now", "now_full", cand

    if "<PAST>" in up or "</PAST>" in up:
        return "past", "now_full", cand

    if "<FUTURE>" in up or "</FUTURE>" in up:
        return "pred", "now_full", cand

    return "other", "now_full", cand


def _has_all_tags(text: str, tags: List[str]) -> bool:
    up = (text or "").upper()
    return all(f"<{t}>" in up and f"</{t}>" in up for t in tags)


def _schema_invalid(kind: str, now_schema: str, cand: bool, text: str) -> Tuple[bool, str]:
    s = (text or "").strip()
    if not s:
        return True, "empty"

    up = s.upper()
    if kind == "now":
        if "</NOW>" not in up:
            return True, "missing_close_now"
        if now_schema == "now_state_only":
            need = ["STATE", "CONF"]
        elif now_schema == "now_action_mcq":
            need = ["ANS", "VERB", "NOUN", "CONF"]
        else:
            need = ["STATE", "VERB", "NOUN", "DESC", "CONF"]
        if not _has_all_tags(s, need):
            return True, "missing_now_subtags"
        if now_schema in {"now_state_only", "now_full"}:
            m = re.search(r"<STATE>(.*?)</STATE>", s, flags=re.IGNORECASE | re.DOTALL)
            if not m:
                return True, "missing_state_value"
            st = m.group(1).strip().upper()
            if st not in {"INTERACTION", "NO_INTERACTION"}:
                return True, "bad_state_enum"
        return False, ""

    if kind == "past":
        if "</PAST>" not in up:
            return True, "missing_close_past"
        need = ["VERB", "NOUN", "DESC", "CONF"]
        if cand:
            need = ["ANS"] + need
        if not _has_all_tags(s, need):
            return True, "missing_past_subtags"
        return False, ""

    if kind == "pred":
        if "</FUTURE>" not in up:
            return True, "missing_close_future"
        need = ["VERB", "NOUN", "DESC", "CONF"]
        if cand:
            need = ["ANS"] + need
        if not _has_all_tags(s, need):
            return True, "missing_future_subtags"
        return False, ""

    return False, ""


# -------------------------
# 1FPS cache + frame IO
# -------------------------
def _ffmpeg_fps1(src_video: Path, dst_video: Path) -> None:
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_video.with_name(dst_video.stem + ".tmp" + dst_video.suffix)
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src_video),
        "-vf", "fps=1",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(tmp),
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    if not tmp.exists() or tmp.stat().st_size < 1024:
        raise RuntimeError(f"ffmpeg fps=1 produced invalid file: {tmp}")
    os.replace(str(tmp), str(dst_video))


def _get_vr(video_path: Path) -> VideoReader:
    return VideoReader(str(video_path), ctx=cpu(0))


def _uniform_sample_indices(all_indices: List[int], max_frames: int) -> List[int]:
    if len(all_indices) <= max_frames:
        return all_indices
    step = len(all_indices) / float(max_frames)
    sampled = []
    for i in range(max_frames):
        j = int(i * step)
        if j >= len(all_indices):
            j = len(all_indices) - 1
        sampled.append(all_indices[j])
    sampled = sorted(set(sampled))
    return sampled if sampled else all_indices[:1]


def _select_indices_1fps(ws_rel: float, we_rel: float, n_total: int, max_frames: int) -> List[int]:
    # 1FPS timeline: frame index ~ second
    if n_total <= 0:
        return [0]
    start_idx = int(max(0, math.floor(float(ws_rel))))
    end_idx = int(min(n_total - 1, math.floor(float(we_rel))))
    if end_idx < start_idx:
        end_idx = start_idx
    all_idx = list(range(start_idx, end_idx + 1))
    if not all_idx:
        all_idx = [max(0, min(n_total - 1, start_idx))]
    return _uniform_sample_indices(all_idx, max_frames=max_frames)


def _frames_to_images(vr: VideoReader, indices: List[int]) -> List[Image.Image]:
    imgs: List[Image.Image] = []
    for idx in indices:
        fr = vr[idx].asnumpy()  # HWC RGB
        imgs.append(Image.fromarray(fr))
    return imgs


# -------------------------
# ROI cache helpers (same idea as your standalone script)
# -------------------------
def _load_roi_cache_any(path: str) -> Dict[int, Dict[str, Any]]:
    """
    Supports:
      - JSONL: each line a dict with frame_idx
      - JSON: {"frames":[...]} or {"0":{...}, "1":{...}}
    Return: dict[int->frame_dict]
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}

    by_idx: Dict[int, Dict[str, Any]] = {}

    if p.suffix.lower() == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fr = json.loads(line)
                if "frame_idx" in fr:
                    by_idx[int(fr["frame_idx"])] = fr
        return by_idx

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "frames" in data and isinstance(data["frames"], list):
        for i, fr in enumerate(data["frames"]):
            by_idx[i] = fr
        return by_idx

    if isinstance(data, dict):
        ok = True
        for k, v in data.items():
            try:
                ki = int(k)
            except Exception:
                ok = False
                break
            by_idx[ki] = v
        if ok:
            return by_idx

    return {}


def _build_roi_cache_for_indices(
    roi_by_frameidx: Dict[int, Dict[str, Any]],
    indices: List[int],
    frame_w: Optional[int],
    frame_h: Optional[int],
    coord: str,
    gaze_r_default: float,
    gaze_center_r_norm: Optional[float] = None,  # <<< NEW (ablation switch + radius)
) -> Dict[str, Any]:
    frames: List[Dict[str, Any]] = []
    for idx in indices:
        fr = roi_by_frameidx.get(int(idx), {}) or {}
        fr = dict(fr)
        fr.setdefault("coord", coord)

        # ---------------- NEW: overwrite gaze to a center circle if enabled ----------------
        if gaze_center_r_norm is not None:
            r0 = float(gaze_center_r_norm)
            r0 = max(1e-6, min(0.5, r0))  # safety clamp

            old_gz = fr.get("gaze", None)
            if isinstance(old_gz, dict):
                gz = dict(old_gz)
            else:
                gz = {}

            if coord == "pixel":
                if frame_w is not None and frame_h is not None:
                    gz["x"] = float(frame_w) * 0.5
                    gz["y"] = float(frame_h) * 0.5
                    gz["r"] = float(r0) * float(min(frame_w, frame_h))
                else:
                    gz["x"] = 0.0
                    gz["y"] = 0.0
                    gz["r"] = float(r0)
            else:
                gz["x"] = 0.5
                gz["y"] = 0.5
                gz["r"] = float(r0)

            if "conf" not in gz:
                if isinstance(old_gz, dict) and ("conf" in old_gz):
                    gz["conf"] = old_gz.get("conf")
                else:
                    gz["conf"] = 1.0

            fr["gaze"] = gz
        else:
            # ---------------- original behavior (unchanged) ----------------
            if "gaze" in fr and isinstance(fr["gaze"], dict):
                fr["gaze"].setdefault("r", gaze_r_default)

        if coord == "pixel":
            if frame_w is not None:
                fr.setdefault("frame_w", frame_w)
            if frame_h is not None:
                fr.setdefault("frame_h", frame_h)

        frames.append(fr)

    return {"frames": frames, "frame_w": frame_w, "frame_h": frame_h}


def _resolve_roi_cache_path(qs: Dict[str, Any], video_path: Path) -> Optional[Path]:
    """
    Priority:
      1) env ROI_CACHE_PATH if exists
      2) env ROI_CACHE_ROOT + canonical filename derived from qs.video_metadata
      3) glob under ROI_CACHE_ROOT for best match
    """
    p0 = os.environ.get("ROI_CACHE_PATH", "").strip()
    if p0:
        p = Path(p0).expanduser()
        if p.exists():
            return p

    root_value = _env("ROI_CACHE_ROOT", "").strip()
    if not root_value:
        return None, {
            "status": "miss",
            "reason": "ROI_CACHE_ROOT is not set",
        }
    root = Path(root_value).expanduser()
    if not root.exists():
        return None

    vm = qs.get("video_metadata", {}) if isinstance(qs.get("video_metadata"), dict) else {}
    video_uid = str(vm.get("video_uid", qs.get("video_uid", ""))).strip()
    clip_uid = str(vm.get("clip_uid", "")).strip()
    clip_id = str(vm.get("clip_id", "")).strip()

    cand_names = []
    if video_uid and clip_id and clip_uid:
        cand_names += [
            f"{video_uid}__{clip_id}__{clip_uid}.roi_cache_merged_fps1.jsonl",
            f"{video_uid}__{clip_id}__{clip_uid}.roi_cache_merged_fps1.json",
            f"{video_uid}__{clip_id}__{clip_uid}.roi_cache.jsonl",
        ]

    for nm in cand_names:
        p = root / nm
        if p.exists():
            return p

    # glob fallback (lightweight)
    pats: List[str] = []
    if video_uid and clip_id and clip_uid:
        pats += [f"{video_uid}__{clip_id}__{clip_uid}*.jsonl", f"{video_uid}*{clip_uid}*.jsonl"]
    elif video_uid:
        pats += [f"{video_uid}*.jsonl"]

    hits: List[Path] = []
    for pat in pats:
        hits += list(root.glob(pat))
        if hits:
            break

    # if still none, try recursive one level (some people store subdirs)
    if not hits:
        for pat in pats[:1]:
            hits += list(root.rglob(pat))
            if hits:
                break

    if hits:
        hits = sorted(hits, key=lambda x: (len(x.as_posix()), x.as_posix()))
        return hits[0]

    return None


# -------------------------
# Adapter
# -------------------------
class _ROI_TimeChatOnlineAdapter:
    def __init__(self) -> None:
        # Ensure repo root is importable, so eval.qwen2_5_vl... works
        _ensure_repo_on_syspath()

        # MUST import your ROI-DTD model class (local code under TimeChat-Online-main/eval/qwen2_5_vl/)
        try:
            from eval.qwen2_5_vl.modeling_qwen2_5_vl_DTD_ROI import (  # type: ignore
                Qwen2_5_VLForConditionalGeneration,
            )
            import inspect
            print("[ROI_IMPORT_OK]", inspect.getfile(Qwen2_5_VLForConditionalGeneration))
        except Exception as e:
            rr = os.environ.get("TIMECHAT_REPO_ROOT", "").strip()
            raise ImportError(
                "Failed to import ROI DTD model class:\n"
                "  from eval.qwen2_5_vl.modeling_qwen2_5_vl_DTD_ROI import Qwen2_5_VLForConditionalGeneration\n\n"
                "Fix:\n"
                "  - Set env TIMECHAT_REPO_ROOT to your TimeChat-Online-main root (it must contain `eval/`).\n"
                "  - Ensure the file exists at: $TIMECHAT_REPO_ROOT/eval/qwen2_5_vl/modeling_qwen2_5_vl_DTD_ROI.py\n"
                f"Current TIMECHAT_REPO_ROOT={rr!r}\n"
                f"Original error: {repr(e)}"
            ) from e

        self._ModelCls = Qwen2_5_VLForConditionalGeneration

        self.model_id = _env("TIMECHAT_HF_MODEL_ID", "wyccccc/TimeChatOnline-7B")
        self.model_name = _env("MODEL_NAME", "roi_timechat_online")

        dev = _env("TIMECHAT_DEVICE", "cuda").lower()
        if dev == "cpu" or not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = "cuda"

        self.attn_impl = _env("TIMECHAT_ATTN_IMPL", "flash_attention_2").strip()
        self.max_frames = _env_int("TIMECHAT_MAX_FRAMES", 64)
        self.sdpa_max_frames = _env_int("TIMECHAT_SDPA_MAX_FRAMES", 6)

        self.max_new_tokens = _env_int("MAX_NEW_TOKENS", 128)
        self.temperature = _env_float("TEMPERATURE", 0.0)
        self.top_p = _env_float("TOP_P", 1.0)

        self.return_logprobs = _env("RETURN_LOGPROBS", "1") != "0"
        self.stop_on_close_tags = _env("STOP_ON_CLOSE_TAGS", "1") != "0"
        self.deanchor_ans = _env("DEANCHOR_ANS", "1") != "0"

        self.enable_retry = _env("ENABLE_RETRY", "1") != "0"
        self.max_retries = max(0, _env_int("MAX_RETRIES", 1))
        self.resume = _env("RESUME", "1") != "0"

        self.close_tag_strs = ["</NOW>", "</PAST>", "</FUTURE>"]

        self.cand_bad_words = _env(
            "CAND_BAD_WORDS",
            "choice_here,verb_here,noun_here_or_none,short_concise_sentence,STATE_HERE,STATE_HERE</STATE>",
        )

        # DTD + ROI knobs (match your standalone defaults)
        self.drop_method = _env("TIMECHAT_DROP_METHOD", "roi_feature").strip()
        self.drop_threshold = _env_float("TIMECHAT_DROP_THRESHOLD", 0.85)
        self.drop_absolute = _env("TIMECHAT_DROP_ABSOLUTE", "1") != "0"
        self.require_dtd = _env("TIMECHAT_REQUIRE_DTD", "1") != "0"
        self.save_dr = _env("TIMECHAT_SAVE_DR", "0") != "0"
        self.dr_save_dir = Path(_env("TIMECHAT_DR_SAVE_DIR", "cache/timechat_dr")).expanduser()

        self.roi_coord = _env("ROI_COORD", "norm").strip().lower()
        self.gaze_r_default = _env_float("GAZE_R_DEFAULT", 0.08)

        # ---------------- NEW: gaze center-circle ablation switch ----------------
        # If ROI_GAZE_CENTER_R is set to a positive float, gaze will be overwritten to center circle with this radius (norm).
        self.gaze_center_r_norm: Optional[float] = None
        _gzr = os.environ.get("ROI_GAZE_CENTER_R", "").strip()
        if _gzr:
            try:
                rr = float(_gzr)
                if rr > 0:
                    self.gaze_center_r_norm = max(1e-6, min(0.5, rr))
            except Exception:
                self.gaze_center_r_norm = None
        # ---------------- NEW block ends ----------------

        # cand_conf controls (kept compatible)
        self._active_pred_flavor: str = ""
        self._active_task: str = ""
        self.cand_conf_max_new_tokens = _env_int("CAND_CONF_MAX_NEW_TOKENS", 4)
        self.cand_conf_force_fallback = _env("CAND_CONF_FORCE_FALLBACK", "1") != "0"

        device_map = "auto" if self.device == "cuda" else None

        load_err: Optional[Exception] = None
        model = None
        for impl in [self.attn_impl, "sdpa"]:
            try:
                model = self._ModelCls.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    device_map=device_map,
                    trust_remote_code=True,
                    attn_implementation=impl,
                )
                self.attn_impl_effective = impl
                load_err = None
                break
            except Exception as e:
                load_err = e
                model = None

        if model is None:
            raise RuntimeError(f"Failed to load model {self.model_id}: {repr(load_err)}")

        self.model = model
        self.model.eval()

        # -------------------------
        # NEW: OPTIONAL finetune loading switch via env var
        #   - TIMECHAT_FT_DIR: directory that contains PEFT adapter (adapter_model.safetensors, adapter_config.json, ...)
        #   - also loads extra_trainable.pt if present
        #   - optional merge: TIMECHAT_MERGE_LORA=1
        # -------------------------
        self.ft_dir = _env("TIMECHAT_FT_DIR", "").strip()
        self.ft_verbose = _env("TIMECHAT_FT_VERBOSE", "1") != "0"
        self.merge_lora = _env("TIMECHAT_MERGE_LORA", "0") != "0"
        if self.ft_dir:
            ft = Path(self.ft_dir).expanduser()
            if not ft.exists():
                raise FileNotFoundError(f"TIMECHAT_FT_DIR does not exist: {ft}")

            if self.ft_verbose:
                print(f"[FT] loading PEFT adapter from TIMECHAT_FT_DIR={str(ft)}")

            try:
                # Attach LoRA adapter onto the already-loaded base model
                self.model = PeftModel.from_pretrained(self.model, str(ft), is_trainable=False)
            except Exception as e:
                raise RuntimeError(f"Failed to load PEFT adapter from {str(ft)!r}: {repr(e)}") from e

            # Load extra unfrozen weights (projector/merger etc.) if present
            extra_path = ft / "extra_trainable.pt"
            if extra_path.exists():
                try:
                    extra = torch.load(str(extra_path), map_location="cpu")
                except Exception as e:
                    raise RuntimeError(f"Failed to torch.load extra_trainable.pt at {str(extra_path)!r}: {repr(e)}") from e

                name2param = dict(self.model.named_parameters())
                loaded, skipped = 0, 0
                for n, w in (extra or {}).items():
                    p = name2param.get(n, None)
                    if p is None:
                        skipped += 1
                        continue
                    try:
                        with torch.no_grad():
                            p.copy_(w.to(device=p.device, dtype=p.dtype))
                        loaded += 1
                    except Exception:
                        skipped += 1

                if self.ft_verbose:
                    print(f"[FT] loaded extra_trainable.pt: loaded={loaded} skipped={skipped} path={str(extra_path)}")
            else:
                if self.ft_verbose:
                    print(f"[FT] extra_trainable.pt not found under {str(ft)} (this is OK if you didn't save any extra)")

            # Optional: merge LoRA for faster inference
            if self.merge_lora:
                if self.ft_verbose:
                    print("[FT] merging LoRA into base model (merge_and_unload)")
                try:
                    self.model = self.model.merge_and_unload()
                except Exception as e:
                    print(f"[FT][WARN] merge_and_unload failed: {repr(e)}")

            self.model.eval()
        # ------------------------- NEW block ends -------------------------

        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.tokenizer = getattr(self.processor, "tokenizer", None)

        if self.attn_impl_effective == "sdpa":
            self.effective_max_frames = min(self.max_frames, self.sdpa_max_frames)
        else:
            self.effective_max_frames = self.max_frames

        self.bad_words_ids_cand = _build_bad_words_ids(self.tokenizer, self.cand_bad_words)

        # 1FPS cache dir
        self.fps1_cache_dir = Path(_env("TIMECHAT_FPS1_CACHE_DIR", "cache/timechat_fps1")).expanduser()
        self.fps1_cache_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # NEW: output path rewrite for cand_conf to avoid overwriting /cand/
    # (same spirit as your video_llava adapter)
    # -------------------------
    def _rewrite_out_path_for_cand_conf(self, out_path: Path) -> Path:
        s = out_path.as_posix()

        # rewrite directory segment if runner passed /cand/
        if "/sh_pred/cand/" in s:
            s2 = s.replace("/sh_pred/cand/", "/sh_pred/cand_conf/")
        elif "/sh_pred_full/cand/" in s:
            s2 = s.replace("/sh_pred_full/cand/", "/sh_pred_full/cand_conf/")
        elif "/ms_pred/cand/" in s:
            s2 = s.replace("/ms_pred/cand/", "/ms_pred/cand_conf/")
        elif "/ms_rtrv/cand/" in s:
            s2 = s.replace("/ms_rtrv/cand/", "/ms_rtrv/cand_conf/")
        else:
            s2 = s

        p2 = Path(s2)
        name = p2.name

        # try common suffix rewrites
        if "__pred__cand.json" in name:
            p2 = p2.with_name(name.replace("__pred__cand.json", "__pred__cand_conf.json"))
        elif "__derived_sh_pred__cand.json" in name:
            p2 = p2.with_name(name.replace("__derived_sh_pred__cand.json", "__derived_sh_pred__cand_conf.json"))
        elif "__derived_sh_pred_full__cand.json" in name:
            p2 = p2.with_name(name.replace("__derived_sh_pred_full__cand.json", "__derived_sh_pred_full__cand_conf.json"))
        elif name.endswith("__pred.json"):
            p2 = p2.with_name(name[:-9] + "__pred__cand_conf.json")
        elif name.endswith(".json") and ("cand_conf" not in name):
            p2 = p2.with_name(name[:-5] + "__cand_conf.json")

        return p2

    def _resolve_call_signature(self, *args, **kwargs) -> Tuple[Path, Path, Path]:
        vp = kwargs.get("video_path") or kwargs.get("clip_path") or kwargs.get("video") or None
        qp = kwargs.get("queryset_path") or kwargs.get("qs_path") or kwargs.get("queryset") or None
        op = kwargs.get("out_json_path") or kwargs.get("out_path") or kwargs.get("output_path") or kwargs.get("out") or None

        if vp is None or qp is None or op is None:
            pos = [a for a in args if a is not None]
            paths = [Path(p) for p in pos if isinstance(p, (str, Path))]
            mp4s = [p for p in paths if p.suffix.lower() in {".mp4", ".mkv", ".webm"}]
            jsons = [p for p in paths if p.suffix.lower() in {".json", ".jsonl"}]
            if vp is None and mp4s:
                vp = mp4s[0]
            if qp is None and len(jsons) >= 1:
                qp = jsons[0]
            if op is None and len(jsons) >= 2:
                op = jsons[1]
            if op is None:
                for p in reversed(paths):
                    if p.suffix.lower() in {".json", ".jsonl"}:
                        op = p
                        break

        if vp is None or qp is None or op is None:
            raise ValueError(f"Cannot resolve adapter.run signature.\nargs={args}\nkwargs keys={list(kwargs.keys())}")

        return Path(vp).expanduser(), Path(qp).expanduser(), Path(op).expanduser()

    def _infer_window_rel(
        self,
        qs: Dict[str, Any],
        sample: Dict[str, Any],
        *,
        clip_dur_sec: float,
    ) -> Tuple[float, float, float, float]:
        params = qs.get("params", {}) if isinstance(qs.get("params"), dict) else {}
        time_offset = float(_safe_float(params.get("time_offset_sec", 0.0), 0.0))

        t_eval_rel = sample.get("t_eval_rel", None)
        if t_eval_rel is None:
            t_eval = float(_safe_float(sample.get("t_eval", 0.0), 0.0))
            t_eval_rel = t_eval - time_offset
        t_eval_rel = float(max(0.0, float(t_eval_rel)))

        ws_sched = sample.get("window_start_sec", None)
        we_sched = sample.get("window_end_sec", None)

        if ws_sched is not None and we_sched is not None:
            ws_rel = float(_safe_float(ws_sched, 0.0) - time_offset)
            we_rel = float(_safe_float(we_sched, t_eval_rel + time_offset) - time_offset)
        else:
            lookback = float(_safe_float(sample.get("lookback_sec", params.get("lookback_sec", 20.0)), 20.0))
            ws_rel = float(max(0.0, t_eval_rel - lookback))
            we_rel = float(t_eval_rel)

        # strict-online clamp
        we_rel = min(float(we_rel), float(t_eval_rel))
        ws_rel = max(0.0, min(float(ws_rel), float(we_rel)))

        if clip_dur_sec > 0:
            ws_rel = float(max(0.0, min(ws_rel, clip_dur_sec)))
            we_rel = float(max(0.0, min(we_rel, clip_dur_sec)))

        lookback_used = float(max(0.0, we_rel - ws_rel))
        return t_eval_rel, ws_rel, we_rel, lookback_used

    def _prepare_interval_fps1_vr(self, interval_clip: Path) -> Tuple[Path, VideoReader, int]:
        """
        Build/read 1FPS cached mp4 for the interval clip, then open decord VideoReader.
        """
        stem = interval_clip.stem
        out = self.fps1_cache_dir / f"{stem}__fps1.mp4"
        if not out.exists():
            _ffmpeg_fps1(interval_clip, out)
        vr = _get_vr(out)
        n = len(vr)
        if n <= 0:
            raise RuntimeError(f"1FPS video has no frames: {out}")
        return out, vr, n

    def _make_inputs_from_images(self, images: List[Image.Image], prompt: str) -> Tuple[Dict[str, Any], int, str]:
        """
        Build Qwen2.5-VL chat input with N images + text.
        Return (inputs_dict, prompt_len, chat_text_preview)
        """
        messages = [{
            "role": "user",
            "content": ([{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]),
        }]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt")

        # move tensors to model device (Scheme A fix: keep second_per_grid_ts on CPU)
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.model.device)
        else:
            inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        try:
            if "second_per_grid_ts" in inputs and hasattr(inputs["second_per_grid_ts"], "to"):
                inputs["second_per_grid_ts"] = inputs["second_per_grid_ts"].to("cpu")
        except Exception:
            pass

        prompt_len = int(inputs["input_ids"].shape[1])
        return inputs, prompt_len, text[:4000]

    def _call_generate(
        self,
        inputs: Dict[str, Any],
        prompt_len: int,
        *,
        kind: str,
        now_schema: str,
        cand: bool,
        roi_cache: Optional[Dict[str, Any]],
        stop_on_close: bool,
    ) -> Tuple[str, List[str], List[float], Optional[float], Optional[float]]:
        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=int(self.max_new_tokens),
            do_sample=False,
            temperature=0.0,
            top_p=None,
            return_dict_in_generate=True,
            output_scores=bool(self.return_logprobs),
        )

        if self.temperature and self.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(self.temperature)
            gen_kwargs["top_p"] = float(self.top_p)

        if kind == "now":
            if now_schema == "now_state_only":
                gen_kwargs["max_new_tokens"] = min(int(gen_kwargs["max_new_tokens"]), 32)
            elif now_schema == "now_action_mcq":
                gen_kwargs["max_new_tokens"] = min(int(gen_kwargs["max_new_tokens"]), 48)

        if cand and self.bad_words_ids_cand:
            gen_kwargs["bad_words_ids"] = self.bad_words_ids_cand

        if stop_on_close and self.tokenizer is not None and kind in {"now", "past", "pred"}:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([
                _StopOnCloseTags(self.tokenizer, prompt_len=prompt_len, stop_strs_upper=self.close_tag_strs)
            ])

        if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token_id", None) is not None:
            gen_kwargs["pad_token_id"] = int(self.tokenizer.eos_token_id)

        dtd_kwargs: Dict[str, Any] = dict(
            drop_method=str(self.drop_method),
            drop_threshold=float(self.drop_threshold),
            drop_absolute=bool(self.drop_absolute),
        )
        if roi_cache is not None:
            dtd_kwargs["roi_cache"] = roi_cache

        dr_save_path = None
        if self.save_dr:
            self.dr_save_dir.mkdir(parents=True, exist_ok=True)
            dr_save_path = str(self.dr_save_dir / f"drop_{int(time.time())}_pid{os.getpid()}.jsonl")
            dtd_kwargs["dr_save_path"] = dr_save_path

        try:
            out = self.model.generate(**inputs, **gen_kwargs, **dtd_kwargs)
        except TypeError as e:
            if self.require_dtd:
                raise TypeError(
                    "model.generate() does not accept ROI-DTD kwargs (drop_method/drop_threshold/drop_absolute/roi_cache).\n"
                    "This means you are NOT running the ROI-DTD-enabled model class.\n"
                    "Check TIMECHAT_REPO_ROOT and that you import Qwen2_5_VLForConditionalGeneration from\n"
                    "  eval.qwen2_5_vl.modeling_qwen2_5_vl_DTD_ROI\n"
                    f"Original error: {repr(e)}"
                ) from e
            out = self.model.generate(**inputs, **gen_kwargs)

        seq = out.sequences[0]
        gen_ids = seq[prompt_len:]
        tok = self.tokenizer
        response_text = tok.decode(gen_ids.tolist(), skip_special_tokens=True).strip() if tok is not None else ""

        gen_tokens: List[str] = []
        gen_token_probs: List[float] = []
        sent_logp: Optional[float] = None
        mean_logp: Optional[float] = None

        if self.return_logprobs and getattr(out, "scores", None) is not None and tok is not None:
            scores = out.scores
            Tn = min(len(scores), int(gen_ids.shape[0]))
            logps: List[float] = []
            for t in range(Tn):
                logits = scores[t][0].float()
                tid = int(gen_ids[t].item())
                lp = float(torch.log_softmax(logits, dim=-1)[tid].item())
                logps.append(lp)
                gen_token_probs.append(float(math.exp(lp)))
                gen_tokens.append(tok.decode([tid], skip_special_tokens=True))
            if logps:
                sent_logp = float(sum(logps))
                mean_logp = float(sent_logp / max(1, len(logps)))

        return response_text, gen_tokens, gen_token_probs, sent_logp, mean_logp

    @torch.inference_mode()
    def _generate_once(
        self,
        *,
        images: List[Image.Image],
        prompt: str,
        kind: str,
        now_schema: str,
        cand: bool,
        roi_cache: Optional[Dict[str, Any]],
    ) -> Tuple[str, List[str], List[float], Optional[float], Optional[float], Dict[str, Any]]:
        diag: Dict[str, Any] = {}

        if self.deanchor_ans and cand:
            prompt, changed = _deanchor_ans_template(prompt)
            diag["deanchor_ans_applied"] = bool(changed)
        else:
            diag["deanchor_ans_applied"] = False

        # ---- cand_conf mode (image-based) ----
        cand_conf_enabled = (str(getattr(self, "_active_pred_flavor", "") or "").strip().lower() == "cand_conf")
        # NEW: extend cand_conf support to sh_pred/ms_pred/ms_rtrv (and sh_pred_full)
        if cand_conf_enabled and str(getattr(self, "_active_task", "") or "").strip().lower() in {"sh_pred", "sh_pred_full", "ms_pred", "ms_rtrv"}:
            has_opts = bool(re.search(r"^\s*[A-D]\.\s+", prompt, flags=re.MULTILINE))
            if has_opts and self.tokenizer is not None:
                def _extract_options_raw_map(text: str) -> Dict[str, str]:
                    outm: Dict[str, str] = {}
                    for ln in (text or "").splitlines():
                        m = re.match(r"^\s*([A-D])\.\s+(.*)$", ln.strip(), flags=re.IGNORECASE)
                        if not m:
                            continue
                        k = m.group(1).upper()
                        v = m.group(2).strip()
                        if v:
                            outm[k] = v
                    return outm

                def _build_ms_cand_conf_prompt(orig: str, opts_raw: Dict[str, str]) -> str:
                    first = ""
                    try:
                        first = (orig.splitlines()[0] or "").strip()
                    except Exception:
                        first = ""
                    title = first if first else "[candidate]"
                    lines: List[str] = []
                    lines.append(title)
                    lines.append("")
                    lines.append("Choose exactly ONE option (A/B/C/D) based on the egocentric video up to now.")
                    lines.append("Output ONLY the single uppercase letter A, B, C, or D. No other text.")
                    lines.append("")
                    lines.append("Options:")
                    for k in ["A", "B", "C", "D"]:
                        if k in opts_raw and str(opts_raw[k]).strip():
                            lines.append(f"{k}. {str(opts_raw[k]).strip()}")
                    return "\n".join(lines).strip()

                def _choice_token_ids(tokenizer, letter: str) -> List[int]:
                    cand_texts = [letter, " " + letter, "\n" + letter, "\n\n" + letter, "\t" + letter]
                    ids: List[int] = []
                    for t in cand_texts:
                        try:
                            enc = tokenizer.encode(t, add_special_tokens=False)
                        except Exception:
                            enc = []
                        if isinstance(enc, list) and len(enc) == 1:
                            tid = int(enc[0])
                            if tid not in ids:
                                ids.append(tid)
                    if not ids:
                        try:
                            enc0 = tokenizer.encode(letter, add_special_tokens=False)
                        except Exception:
                            enc0 = []
                        if isinstance(enc0, list) and len(enc0) >= 1:
                            ids = [int(enc0[0])]
                    return ids

                class _StopOnFirstABCD(StoppingCriteria):
                    def __init__(self, tokenizer, prompt_len: int, max_decode_tokens: int = 16):
                        super().__init__()
                        self.tokenizer = tokenizer
                        self.prompt_len = int(max(0, prompt_len))
                        self.max_decode_tokens = int(max(8, max_decode_tokens))

                    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                        if self.tokenizer is None:
                            return False
                        try:
                            seq = input_ids[0]
                            if seq.numel() <= self.prompt_len:
                                return False
                            gen = seq[self.prompt_len:]
                            if gen.numel() > self.max_decode_tokens:
                                gen = gen[-self.max_decode_tokens:]
                            txt = self.tokenizer.decode(gen.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
                            up = (txt or "").upper()
                            return bool(re.search(r"[A-D]", up))
                        except Exception:
                            return False

                opts_raw = _extract_options_raw_map(prompt)
                prompt_conf = _build_ms_cand_conf_prompt(prompt, opts_raw)
                diag["cand_conf_enabled"] = True
                diag["cand_conf_task"] = str(getattr(self, "_active_task", "") or "")
                diag["cand_conf_prompt_preview"] = prompt_conf[:1200] if len(prompt_conf) > 1200 else prompt_conf

                inputs, prompt_len, _ = self._make_inputs_from_images(images, prompt_conf)

                # probe next-token distribution at the first step (same as video_llava style)
                probe: Dict[str, Any] = {"ok": False}
                try:
                    out0 = self.model(**inputs)
                    logits = getattr(out0, "logits", None)
                    if isinstance(logits, torch.Tensor) and logits.ndim == 3:
                        last = logits[0, -1, :].float()
                        probs = torch.softmax(last, dim=-1)

                        p_raw: Dict[str, float] = {}
                        token_ids_map: Dict[str, List[int]] = {}
                        for L in ["A", "B", "C", "D"]:
                            ids = _choice_token_ids(self.tokenizer, L)
                            token_ids_map[L] = ids
                            if ids:
                                idx_t = torch.tensor(ids, device=probs.device, dtype=torch.long)
                                pv = float(probs.index_select(0, idx_t).sum().item())
                            else:
                                pv = 0.0
                            p_raw[L] = float(pv)

                        mass = float(sum(p_raw.values()))
                        p_cond = {k: (p_raw[k] / mass if mass > 0 else None) for k in p_raw.keys()}
                        chosen = max(["A", "B", "C", "D"], key=lambda x: float(p_cond[x] or 0.0)) if mass > 0 else None

                        ent = None
                        if mass > 0:
                            ee = 0.0
                            for L in ["A", "B", "C", "D"]:
                                v = float(p_cond[L] or 0.0)
                                if v > 0:
                                    ee += -v * math.log(v + 1e-12)
                            ent = float(ee)

                        probe = {
                            "ok": True,
                            "p_raw": p_raw,
                            "mass_abcd": mass,
                            "p_cond": p_cond,
                            "chosen_by_p_cond": chosen,
                            "entropy_p_cond": ent,
                            "token_ids_map": token_ids_map,
                        }
                except Exception as e:
                    probe = {"ok": False, "error": repr(e)}
                diag["cand_conf_probe"] = probe

                # small constrained generation (mainly sanity)
                gen_kwargs: Dict[str, Any] = dict(
                    max_new_tokens=int(max(1, min(int(self.cand_conf_max_new_tokens), 8))),
                    do_sample=False,
                    temperature=0.0,
                    top_p=None,
                    return_dict_in_generate=True,
                    output_scores=bool(self.return_logprobs),
                    stopping_criteria=StoppingCriteriaList([_StopOnFirstABCD(self.tokenizer, prompt_len=prompt_len)]),
                )

                if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token_id", None) is not None:
                    gen_kwargs["pad_token_id"] = int(self.tokenizer.eos_token_id)

                dtd_kwargs: Dict[str, Any] = dict(
                    drop_method=str(self.drop_method),
                    drop_threshold=float(self.drop_threshold),
                    drop_absolute=bool(self.drop_absolute),
                )
                if roi_cache is not None:
                    dtd_kwargs["roi_cache"] = roi_cache

                try:
                    out = self.model.generate(**inputs, **gen_kwargs, **dtd_kwargs)
                except TypeError as e:
                    if self.require_dtd:
                        raise
                    out = self.model.generate(**inputs, **gen_kwargs)

                seq = out.sequences[0]
                gen_ids = seq[prompt_len:]
                tok = self.tokenizer
                resp_text_raw = tok.decode(gen_ids.tolist(), skip_special_tokens=True).strip() if tok is not None else ""

                letter = ""
                m = re.search(r"[A-D]", (resp_text_raw or "").upper())
                if m:
                    letter = m.group(0).upper()
                if (not letter) and probe.get("ok") and self.cand_conf_force_fallback:
                    ch = probe.get("chosen_by_p_cond", None)
                    if isinstance(ch, str) and ch in {"A", "B", "C", "D"}:
                        letter = ch

                diag["cand_conf_generated_raw"] = resp_text_raw[:500] if len(resp_text_raw) > 500 else resp_text_raw
                diag["cand_conf_final_letter"] = letter if letter else None

                # We still return logprobs list optionally (best-effort)
                gen_tokens: List[str] = []
                gen_token_probs: List[float] = []
                sent_logp: Optional[float] = None
                mean_logp: Optional[float] = None
                if self.return_logprobs and getattr(out, "scores", None) is not None and tok is not None:
                    scores = out.scores
                    Tn = min(len(scores), int(gen_ids.shape[0]))
                    logps: List[float] = []
                    for t in range(Tn):
                        logits_t = scores[t][0].float()
                        tid = int(gen_ids[t].item())
                        lp = float(torch.log_softmax(logits_t, dim=-1)[tid].item())
                        logps.append(lp)
                        gen_token_probs.append(float(math.exp(lp)))
                        gen_tokens.append(tok.decode([tid], skip_special_tokens=True))
                    if logps:
                        sent_logp = float(sum(logps))
                        mean_logp = float(sent_logp / max(1, len(logps)))

                return (letter if letter else resp_text_raw.strip()), gen_tokens, gen_token_probs, sent_logp, mean_logp, diag

        # ---- normal generation (image-based) ----
        inputs, prompt_len, _ = self._make_inputs_from_images(images, prompt)

        t0 = time.time()
        resp_text, gen_tokens, gen_token_probs, sent_logp, mean_logp = self._call_generate(
            inputs,
            prompt_len,
            kind=kind,
            now_schema=now_schema,
            cand=cand,
            roi_cache=roi_cache,
            stop_on_close=self.stop_on_close_tags,
        )
        diag["latency_generate_sec"] = float(time.time() - t0)
        return resp_text, gen_tokens, gen_token_probs, sent_logp, mean_logp, diag

    def _retry_suffix(self, kind: str, now_schema: str) -> str:
        if kind == "now":
            if now_schema == "now_state_only":
                return (
                    "Output EXACTLY one line:\n"
                    "<NOW><STATE>INTERACTION</STATE><CONF>0.00</CONF></NOW>\n"
                    "Do NOT output anything outside the tags."
                )
            if now_schema == "now_action_mcq":
                return (
                    "Output EXACTLY one line:\n"
                    "<NOW><ANS>A</ANS><VERB>verb_here</VERB><NOUN>noun_here</NOUN><CONF>0.00</CONF></NOW>\n"
                    "ANS must be one of A/B/C/D and VERB/NOUN must be copied EXACTLY from the chosen option."
                )
            return (
                "Output EXACTLY one line:\n"
                "<NOW><STATE>INTERACTION</STATE><VERB>verb_here</VERB><NOUN>noun_here_or_none</NOUN>"
                "<DESC>short_sentence</DESC><CONF>0.00</CONF></NOW>\n"
                "Do NOT output anything outside the tags."
            )

        if kind == "past":
            return (
                "Output EXACTLY one line:\n"
                "<PAST><VERB>none</VERB><NOUN>none</NOUN><DESC>YOU do nothing</DESC><CONF>0.00</CONF></PAST>\n"
                "Do NOT output anything outside the tags."
            )

        if kind == "pred":
            return (
                "Output EXACTLY one line:\n"
                "<FUTURE><VERB>none</VERB><NOUN>none</NOUN><DESC>YOU do nothing</DESC><CONF>0.00</CONF></FUTURE>\n"
                "Do NOT output anything outside the tags."
            )

        return "Output only the required one-line tags."

    def run(self, *args, **kwargs) -> Path:
        video_path, queryset_path, out_path = self._resolve_call_signature(*args, **kwargs)

        qs = _load_json_or_one_jsonl(queryset_path)
        task = _guess_task(qs)

        params = qs.get("params", {}) if isinstance(qs.get("params"), dict) else {}
        env_flavor = os.environ.get("PRED_FLAVOR", "").strip().lower()
        qs_flavor = str(params.get("pred_flavor", "") or "").strip().lower()
        self._active_pred_flavor = env_flavor if env_flavor else qs_flavor
        self._active_task = str(task or "").strip().lower()

        # NEW: reroute output path for cand_conf to avoid overwriting /cand/
        if self._active_pred_flavor == "cand_conf" and self._active_task in {"sh_pred", "sh_pred_full", "ms_pred", "ms_rtrv"}:
            out_path = self._rewrite_out_path_for_cand_conf(out_path)

        vm = qs.get("video_metadata", {}) if isinstance(qs.get("video_metadata"), dict) else {}
        video_uid = str(vm.get("video_uid", qs.get("video_uid", ""))).strip()

        samples = qs.get("samples", [])
        if not isinstance(samples, list):
            raise ValueError(f"Effective queryset 'samples' must be list: {queryset_path}")

        # resume
        done_by_idx: Dict[int, Dict[str, Any]] = {}
        if self.resume and out_path.exists():
            try:
                prev = _load_json_or_one_jsonl(out_path)
                prev_samples = prev.get("samples", []) if isinstance(prev, dict) else []
                if isinstance(prev_samples, list):
                    for r in prev_samples:
                        if isinstance(r, dict) and "idx" in r:
                            done_by_idx[_safe_int(r["idx"], -1)] = r
            except Exception:
                done_by_idx = {}

        # Prepare interval 1FPS VR once
        fps1_path, vr1, n_total_1fps = self._prepare_interval_fps1_vr(video_path)
        clip_dur_sec = float(n_total_1fps - 1)  # ~ seconds range; used only for clamping

        # Resolve ROI cache once per interval clip
        roi_path = _resolve_roi_cache_path(qs, video_path)
        roi_by_frameidx: Optional[Dict[int, Dict[str, Any]]] = None
        if str(self.drop_method).lower().startswith("roi") and roi_path is not None and roi_path.exists():
            roi_by_frameidx = _load_roi_cache_any(str(roi_path))
        else:
            roi_by_frameidx = None

        out_obj: Dict[str, Any] = {
            "dataset": qs.get("dataset", "Ego4D"),
            "task": task,
            "video_uid": video_uid,
            "model_name": self.model_name,
            "model_id": self.model_id,
            "source_queryset": str(queryset_path),
            "video_clip": str(video_path),
            "generated_at_unix": float(time.time()),
            "params": qs.get("params", {}),
            "roi": {
                "drop_method": str(self.drop_method),
                "drop_threshold": float(self.drop_threshold),
                "drop_absolute": bool(self.drop_absolute),
                "roi_cache_path": str(roi_path) if roi_path is not None else None,
                "roi_cache_loaded_frames": int(len(roi_by_frameidx)) if isinstance(roi_by_frameidx, dict) else 0,
                "roi_coord": self.roi_coord,
                "gaze_r_default": float(self.gaze_r_default),
                # NEW: record ablation config (does not affect default behavior)
                "gaze_center_circle_enabled": bool(self.gaze_center_r_norm is not None),
                "gaze_center_circle_r_norm": float(self.gaze_center_r_norm) if self.gaze_center_r_norm is not None else None,
            },
            "fps1_cache": {
                "fps1_video": str(fps1_path),
                "fps1_total_frames": int(n_total_1fps),
            },
            "samples": [],
        }

        for s in samples:
            if not isinstance(s, dict):
                continue
            idx = _safe_int(s.get("idx", -1), -1)

            if idx >= 0 and idx in done_by_idx:
                out_obj["samples"].append(done_by_idx[idx])
                continue

            prompt = str(s.get("prompt", "")).strip()
            if not prompt:
                out_obj["samples"].append({"idx": idx, "error": "missing prompt"})
                _dump_json(out_path, out_obj)
                continue

            kind, now_schema, cand = _infer_kind_and_schema(prompt)

            t_eval_rel, ws_rel, we_rel, lookback_used = self._infer_window_rel(qs, s, clip_dur_sec=clip_dur_sec)

            # Explicit 1FPS indices on interval timeline
            indices = _select_indices_1fps(ws_rel, we_rel, n_total_1fps, max_frames=int(self.effective_max_frames))
            images = _frames_to_images(vr1, indices)

            # Build aligned roi_cache with SAME indices order
            roi_cache = None
            roi_hit = 0
            if roi_by_frameidx is not None and str(self.drop_method).lower().startswith("roi"):
                w, h = images[0].size
                roi_cache = _build_roi_cache_for_indices(
                    roi_by_frameidx=roi_by_frameidx,
                    indices=indices,
                    frame_w=w,
                    frame_h=h,
                    coord=self.roi_coord,
                    gaze_r_default=float(self.gaze_r_default),
                    gaze_center_r_norm=self.gaze_center_r_norm,  # <<< NEW
                )
                # hit rate diagnostics
                for ii in indices:
                    if int(ii) in roi_by_frameidx:
                        roi_hit += 1

            t0 = time.time()
            resp_text, gen_tokens, gen_token_probs, sent_logp, mean_logp, gen_diag = self._generate_once(
                images=images,
                prompt=prompt,
                kind=kind,
                now_schema=now_schema,
                cand=cand,
                roi_cache=roi_cache,
            )
            latency_first = float(time.time() - t0)

            final_text = resp_text
            retry_triggered = False
            retry_reason = ""
            attempts: List[Dict[str, Any]] = [{
                "attempt": 1,
                "latency_sec": latency_first,
                "diag": gen_diag,
            }]

            # no schema retry for cand_conf
            if self.enable_retry and self.max_retries > 0 and (self._active_pred_flavor != "cand_conf"):
                bad, reason = _schema_invalid(kind, now_schema, cand, _first_tag_block(final_text))
                if bad:
                    retry_triggered = True
                    retry_reason = reason
                    suffix = self._retry_suffix(kind, now_schema)
                    retry_prompt = (prompt.rstrip() + "\n\n" + suffix.strip()).strip()

                    for ri in range(self.max_retries):
                        t1 = time.time()
                        rt, gt, gp, slp, mlp, rdiag = self._generate_once(
                            images=images,
                            prompt=retry_prompt,
                            kind=kind,
                            now_schema=now_schema,
                            cand=cand,
                            roi_cache=roi_cache,
                        )
                        dt = float(time.time() - t1)
                        attempts.append({
                            "attempt": int(2 + ri),
                            "latency_sec": dt,
                            "diag": rdiag,
                            "retry_suffix_used": suffix,
                        })
                        final_text = rt
                        gen_tokens, gen_token_probs, sent_logp, mean_logp = gt, gp, slp, mlp

                        bad2, reason2 = _schema_invalid(kind, now_schema, cand, _first_tag_block(final_text))
                        if not bad2:
                            retry_reason = ""
                            break
                        retry_reason = reason2

            latency_total = float(sum([a.get("latency_sec", 0.0) for a in attempts]))

            clean = _first_tag_block(final_text)
            parsed = _parse_xmlish(clean)

            # cand_conf convenience (extend to sh_pred/ms_pred/ms_rtrv)
            if self._active_pred_flavor == "cand_conf" and self._active_task in {"sh_pred", "sh_pred_full", "ms_pred", "ms_rtrv"}:
                m = re.search(r"[A-D]", (clean or "").upper())
                if m:
                    parsed.setdefault("ans", m.group(0).upper())

            rec = {
                "idx": idx,
                "t_eval": _safe_float(s.get("t_eval", 0.0), 0.0),
                "t_eval_rel": _safe_float(s.get("t_eval_rel", t_eval_rel), t_eval_rel),
                "prompt": prompt,
                "response_text": final_text,
                "clean_response": clean,
                "parsed": parsed,
                "gen_tokens": gen_tokens,
                "gen_token_probs": gen_token_probs,
                "sent_logp": sent_logp,
                "mean_logp": mean_logp,
                "latency_sec": latency_total,
                "diagnostics": {
                    "kind": kind,
                    "now_schema": now_schema if kind == "now" else None,
                    "candidate": bool(cand),
                    "pred_flavor_active": self._active_pred_flavor,
                    "retry_triggered": bool(retry_triggered),
                    "retry_reason": retry_reason,
                    "attempts": attempts,
                    # strict-online window (interval-clip coords)
                    "window_start_rel": float(ws_rel),
                    "window_end_rel": float(we_rel),
                    "lookback_used_sec": float(lookback_used),
                    # frame sampling (explicit)
                    "fps_used": 1.0,
                    "frame_indices_1fps": indices,
                    "num_images": int(len(images)),
                    "effective_max_frames": int(self.effective_max_frames),
                    # ROI
                    "roi_cache_used": bool(roi_cache is not None),
                    "roi_cache_hit": int(roi_hit),
                    "roi_cache_total": int(len(indices)),
                    "roi_cache_hit_rate": float(roi_hit / max(1, len(indices))),
                    # model
                    "attn_impl_effective": str(getattr(self, "attn_impl_effective", "")),
                },
                "raw": None,
            }

            out_obj["samples"].append(rec)
            _dump_json(out_path, out_obj)

        _dump_json(out_path, out_obj)
        return out_path


def create_adapter():
    return _ROI_TimeChatOnlineAdapter()
