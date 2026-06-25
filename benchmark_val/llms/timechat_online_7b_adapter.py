#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TimeChat-Online adapter for runner+helper pipeline.

Contract:
  - create_adapter() -> instance with .run(...)
  - run(video_path, queryset_path, out_json_path, ...) writes a unified JSON:
      {
        dataset, task, video_uid, model_name, model_id, source_queryset,
        video_clip, generated_at_unix, params, samples:[...]
      }

Key behaviors:
  - Use helper-produced prompts (sample["prompt"]) directly.
  - Strict-online windowing:
      prefer sample["window_start_sec"/"window_end_sec"] (scheduled time),
      convert to clip-relative using params["time_offset_sec"],
      and clamp window_end_rel <= t_eval_rel.
    fallback: [t_eval_rel-lookback, t_eval_rel].
  - No 1FPS downsample. Use real fps from decord.
  - Compute token probs/logp from generate(output_scores=True).
  - Optional: stop-on-close-tags, candidate bad words, de-anchor, lightweight retry.

[IMPORTANT MODIFICATION in this version]
  - Follow official TimeChat-Online README: feed a "video" item + process_vision_info(...)
    and enable DTD by passing drop_method/drop_threshold/drop_absolute/(optional)dr_save_path
    into model.generate(...).

[SCHEME A FIX]
  - Keep `second_per_grid_ts` on CPU (do NOT move it to CUDA), to avoid
    get_rope_index() mixing CPU torch.arange with CUDA scalar tensors.

Env knobs (all optional):
  MODEL_NAME
  TIMECHAT_REPO_ROOT
  TIMECHAT_HF_MODEL_ID
  TIMECHAT_DEVICE
  TIMECHAT_ATTN_IMPL: flash_attention_2|sdpa|eager (default flash_attention_2, fallback to sdpa)
  TIMECHAT_MAX_FRAMES (default 64)
  TIMECHAT_SDPA_MAX_FRAMES (default 6)
  TIMECHAT_VIDEO_MIN_FRAMES (default 4)
  TIMECHAT_VIDEO_FPS (optional; if empty we auto-pick based on clip_len and max_frames, clamped [1,30])

  # DTD knobs (official)
  TIMECHAT_DROP_METHOD (default feature)
  TIMECHAT_DROP_THRESHOLD (default 0.5)
  TIMECHAT_DROP_ABSOLUTE (default 1)
  TIMECHAT_REQUIRE_DTD (default 1)  # if 1 and model.generate doesn't accept drop_*, raise
  TIMECHAT_SAVE_DR (default 0)      # if 1, write dr_save_path jsonl
  TIMECHAT_DR_SAVE_DIR (default cache/timechat_dr)

  MAX_NEW_TOKENS (default 128)
  TEMPERATURE/TOP_P
  RETURN_LOGPROBS (default 1)
  STOP_ON_CLOSE_TAGS (default 1)
  DEANCHOR_ANS (default 1)
  CAND_BAD_WORDS
  ENABLE_RETRY (default 1)
  MAX_RETRIES (default 1)
  RESUME (default 1)

  # temp clip cache
  TIMECHAT_CLIP_CACHE_DIR (default cache/timechat_clips)
  TIMECHAT_CLIP_CODEC (default libx264)
  TIMECHAT_CLIP_PRESET (default ultrafast)
  TIMECHAT_CLIP_CRF (default 28)

NEW (cand_conf mode):
  - Triggered when env PRED_FLAVOR == "cand_conf" or qs["params"]["pred_flavor"] == "cand_conf".
  - Only active for TASK in {"ms_pred","ms_rtrv","sh_pred","sh_pred_full"}.
  - Rewrites prompt to force output ONLY one letter A/B/C/D (no tags).
  - Computes next-token probability mass over {A,B,C,D} from logits at the first generation step:
      p_raw[A..D], mass_abcd, p_cond[A..D], chosen_by_p_cond, entropy_p_cond
  - Outputs response_text/clean_response as a single letter.
  - Writes probe info into diagnostics.attempts[0].diag["cand_conf_probe"] and also top-level gen diag.
  - Optional env knobs:
      CAND_CONF_MAX_NEW_TOKENS (default 4)
      CAND_CONF_FORCE_FALLBACK (default 1)  # if model doesn't output A-D, fallback to chosen_by_p_cond

NEW (finetune loading, same as ROI adapter):
  - TIMECHAT_FT_DIR: load PEFT LoRA adapter from this dir
  - optional extra_trainable.pt: load extra trainable weights (e.g. projector) if present
  - TIMECHAT_MERGE_LORA=1: merge LoRA into base for faster inference
  - TIMECHAT_FT_VERBOSE=1: print load summary

CLI:
  --vision on|off   default: on
    - on : use visual input normally
    - off: text-only ablation (no clip cutting, no video input)
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

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, StoppingCriteria, StoppingCriteriaList

from decord import VideoReader, cpu  # type: ignore
from PIL import Image

# ---- NEW: PEFT support (same as ROI adapter) ----
try:
    from peft import PeftModel  # type: ignore
except Exception:
    PeftModel = None  # type: ignore

# ---- official util ----
try:
    from qwen_vl_utils import process_vision_info  # type: ignore
except Exception:
    try:
        from qwen_vl_utils.src.qwen_vl_utils import process_vision_info  # type: ignore
    except Exception as e:
        raise ImportError(
            "Cannot import process_vision_info. TimeChat-Online README requires it.\n"
            "Try: pip install qwen-vl-utils  (or ensure qwen_vl_utils is on PYTHONPATH).\n"
            f"Original error: {repr(e)}"
        )


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


def _parse_vision_enabled_from_argv(default: bool = True) -> bool:
    """
    Read --vision on|off from current process argv.

    This adapter is usually imported and called by runner.py within the SAME
    Python process, so parsing sys.argv here allows:
        python runner.py --mode cand --vision off
    """
    try:
        argv = list(sys.argv or [])
    except Exception:
        return bool(default)

    if not argv:
        return bool(default)

    v = None
    for i, a in enumerate(argv):
        if a == "--vision" and i + 1 < len(argv):
            vv = argv[i + 1]
            if vv and not vv.startswith("--"):
                v = vv
                break

    if v is None:
        return bool(default)

    vv = str(v).strip().lower()
    if vv in {"off", "0", "false", "no", "none", "text", "text_only", "text-only"}:
        return False
    if vv in {"on", "1", "true", "yes", "vision", "visual", "image", "images"}:
        return True
    return bool(default)


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
    repo_root = os.environ.get("TIMECHAT_REPO_ROOT", "").strip()
    if not repo_root:
        return
    rr = str(Path(repo_root).expanduser().resolve())
    if rr not in sys.path:
        sys.path.insert(0, rr)


# -------------------------
# Video metadata (keep decord for fps/duration)
# -------------------------
def _get_vr(video_path: Path) -> VideoReader:
    return VideoReader(str(video_path), ctx=cpu(0))


def _get_fps(vr: VideoReader) -> float:
    fps = 0.0
    try:
        if hasattr(vr, "get_avg_fps"):
            fps = float(vr.get_avg_fps())
    except Exception:
        fps = 0.0
    if not (fps > 0):
        fps = 30.0
    return float(fps)


# -------------------------
# Prompt helpers (minimal + safe)
# -------------------------
def _deanchor_ans_template(prompt: str) -> Tuple[str, bool]:
    if not prompt:
        return prompt, False
    p2 = re.sub(r"<\s*ANS\s*>\s*A\s*<\s*/\s*ANS\s*>", "<ANS>choice_here</ANS>", prompt, flags=re.IGNORECASE)
    return p2, (p2 != prompt)


def _memory_env_enabled() -> bool:
    return _env("MEMORY_MODE", "0").strip().lower() in {"1", "true", "yes", "y", "on"}


def _memory_fail_on_missing_caption() -> bool:
    return _env("MEMORY_FAIL_ON_MISSING_CAPTION", "0").strip().lower() in {"1", "true", "yes", "y", "on"}


def _memory_require_keyframes() -> bool:
    return _env("MEMORY_REQUIRE_KEYFRAMES", "0").strip().lower() in {"1", "true", "yes", "y", "on"}


def _memory_sec_label(value: Any) -> str:
    try:
        v = float(value)
        return f"{v:.3f}".rstrip("0").rstrip(".")
    except Exception:
        return str(value)


def _path_from_memory_keyframe_item(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in ("path", "frame_path", "image_path", "file", "filename"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _resolve_existing_memory_path(raw_path: str, base_dirs: List[Path]) -> Optional[str]:
    raw_path = str(raw_path or "").strip()
    if not raw_path:
        return None

    p = Path(raw_path).expanduser()
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        for base_dir in base_dirs:
            candidates.append(base_dir / p)
        candidates.append(p)

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        try:
            if resolved.is_file():
                return str(resolved)
        except Exception:
            continue
    return None


def _resolve_memory_keyframe_items(items: Any, base_dirs: List[Path]) -> List[str]:
    if isinstance(items, (str, dict)):
        iterable = [items]
    elif isinstance(items, list):
        iterable = items
    else:
        iterable = []

    out: List[str] = []
    seen = set()
    for item in iterable:
        raw_path = _path_from_memory_keyframe_item(item)
        resolved = _resolve_existing_memory_path(raw_path, base_dirs)
        if resolved and resolved not in seen:
            out.append(resolved)
            seen.add(resolved)
    return out


def _load_json_any(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raw = path.read_text(encoding="utf-8-sig").strip()
    if not raw:
        raise ValueError(f"Empty file: {path}")
    return json.loads(raw)


def _manifest_keyframe_items(manifest_obj: Any) -> List[Any]:
    if isinstance(manifest_obj, list):
        return manifest_obj
    if isinstance(manifest_obj, dict):
        items: List[Any] = []
        for key in ("keyframe_paths", "keyframes", "frames", "items", "images", "paths"):
            value = manifest_obj.get(key)
            if isinstance(value, list):
                items.extend(value)
        if items:
            return items
        return [manifest_obj]
    return []


def _memory_keyframe_paths_from_caption(caption_record: Dict[str, Any], caption_path: Path) -> List[str]:
    caption_dir = caption_path.parent

    direct_paths = _resolve_memory_keyframe_items(caption_record.get("keyframe_paths"), [caption_dir])
    if direct_paths:
        return direct_paths

    manifest_raw = str(caption_record.get("keyframe_manifest_path", "") or "").strip()
    manifest_resolved = _resolve_existing_memory_path(manifest_raw, [caption_dir])
    if not manifest_resolved:
        return []

    manifest_path = Path(manifest_resolved)
    manifest_obj = _load_json_any(manifest_path)
    return _resolve_memory_keyframe_items(_manifest_keyframe_items(manifest_obj), [manifest_path.parent, caption_dir])


def build_prompt_with_memory(sample: Dict[str, Any], base_prompt: str) -> Tuple[str, Dict[str, Any], List[str]]:
    if not _memory_env_enabled():
        return base_prompt, {}, []

    memory = sample.get("memory", {}) if isinstance(sample, dict) else {}
    if not isinstance(memory, dict) or not memory.get("enabled"):
        return base_prompt, {}, []

    sparse = memory.get("sparse_window", {}) if isinstance(memory.get("sparse_window"), dict) else {}
    dense = memory.get("dense_window", {}) if isinstance(memory.get("dense_window"), dict) else {}
    caption_path = str(sparse.get("caption_cache_path", "") or "").strip()

    diag: Dict[str, Any] = {
        "enabled": True,
        "setting_name": memory.get("setting_name"),
        "caption_used": False,
        "caption_cache_hit": False,
        "caption_missing": True,
        "caption_cache_path": caption_path,
        "sparse_window": dict(sparse),
        "dense_window": dict(dense),
        "no_lookahead": memory.get("no_lookahead"),
        "prompt_injected": False,
        "keyframes_available": False,
        "keyframe_paths": [],
        "keyframe_count": 0,
        "keyframes_used": False,
    }

    caption_text = ""
    caption_error = ""
    memory_keyframe_paths: List[str] = []
    caption_file_exists = bool(caption_path and Path(caption_path).is_file())
    diag["caption_cache_hit"] = bool(caption_file_exists)

    if caption_file_exists:
        try:
            caption_record = _load_json_or_one_jsonl(Path(caption_path))
            caption_text = str(caption_record.get("caption_text", "") or "").strip()
            try:
                memory_keyframe_paths = _memory_keyframe_paths_from_caption(caption_record, Path(caption_path))
            except Exception as e:
                diag["keyframe_error"] = repr(e)
        except Exception as e:
            caption_error = repr(e)

    diag["keyframes_available"] = bool(memory_keyframe_paths)
    diag["keyframe_paths"] = memory_keyframe_paths
    diag["keyframe_count"] = len(memory_keyframe_paths)

    if _memory_require_keyframes() and not memory_keyframe_paths:
        raise RuntimeError(
            "MEMORY_REQUIRE_KEYFRAMES=1 but no valid sparse keyframe paths found for caption cache: "
            f"{caption_path or '<missing path>'}"
        )

    if not caption_text:
        if caption_error:
            diag["caption_error"] = caption_error
        if _memory_fail_on_missing_caption():
            detail = f"caption_cache_path={caption_path or '<missing path>'}"
            if caption_error:
                detail += f"; error={caption_error}"
            raise RuntimeError(f"MEMORY_MODE=1 but memory caption is missing or empty: {detail}")
        return base_prompt, diag, memory_keyframe_paths

    setting_name = str(memory.get("setting_name", "") or "")
    start_sec = _memory_sec_label(sparse.get("start_sec", ""))
    end_sec = _memory_sec_label(sparse.get("end_sec", ""))
    memory_block = (
        "[Sparse memory from earlier observed video]\n"
        "This memory summarizes visual evidence observed before the current dense video window. "
        "Use it only as past context. Do not treat it as future evidence.\n"
        f"Memory setting: {setting_name}\n"
        f"Sparse memory window: {start_sec}s to {end_sec}s, before current time.\n"
        f"Caption: {caption_text}\n"
        "[Current task]\n\n"
    )

    diag["caption_used"] = True
    diag["caption_missing"] = False
    diag["prompt_injected"] = True
    return memory_block + base_prompt, diag, memory_keyframe_paths


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


# -------------------------
# ffmpeg clip helper (strict-online window -> temp clip)
# -------------------------
def _run_ffmpeg_cut(src_video: Path, dst_clip: Path, start_sec: float, end_sec: float) -> None:
    dst_clip.parent.mkdir(parents=True, exist_ok=True)

    start_sec = float(max(0.0, start_sec))
    end_sec = float(max(start_sec, end_sec))
    dur = float(end_sec - start_sec)

    codec = _env("TIMECHAT_CLIP_CODEC", "libx264")
    preset = _env("TIMECHAT_CLIP_PRESET", "ultrafast")
    crf = _env("TIMECHAT_CLIP_CRF", "28")

    tmp = dst_clip.with_name(dst_clip.stem + ".tmp" + dst_clip.suffix)
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.6f}",
        "-i", str(src_video),
        "-t", f"{dur:.6f}",
        "-an",
        "-c:v", codec,
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(tmp),
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    if not tmp.exists() or tmp.stat().st_size < 1024:
        raise RuntimeError(f"ffmpeg produced invalid clip: {tmp}")
    os.replace(str(tmp), str(dst_clip))


def _make_clip_path(video_uid: str, idx: int, ws_rel: float, we_rel: float) -> Path:
    cache_root = Path(_env("TIMECHAT_CLIP_CACHE_DIR", "cache/timechat_clips")).expanduser()
    name = f"{video_uid or 'video'}_idx{idx}_ws{ws_rel:.3f}_we{we_rel:.3f}.mp4"
    return cache_root / name


def _pick_video_fps(clip_len_sec: float, max_frames: int) -> float:
    v = os.environ.get("TIMECHAT_VIDEO_FPS", "").strip()
    if v:
        try:
            fv = float(v)
            if fv > 0:
                return float(fv)
        except Exception:
            pass
    if clip_len_sec <= 1e-6:
        return 1.0
    fps = float(max_frames) / float(clip_len_sec)
    fps = max(1.0, min(30.0, fps))
    return float(fps)


# -------------------------
# Adapter
# -------------------------
class _TimeChatOnlineAdapter:
    def __init__(self) -> None:
        _ensure_repo_on_syspath()

        try:
            from eval.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration  # type: ignore
        except Exception:
            try:
                from eval.qwen2_5_vl.modeling_qwen2_5_vl_DTD import Qwen2_5_VLForConditionalGeneration  # type: ignore
            except Exception as e:
                raise ImportError(
                    "Failed to import TimeChat model code.\n"
                    "Set env TIMECHAT_REPO_ROOT to your TimeChat-Online-main repo, e.g.\n"
                    "  export TIMECHAT_REPO_ROOT=~/egocentric_streaming_vlm/baseline/TimeChat-Online-main\n"
                    f"Original error: {repr(e)}"
                ) from e

        self._ModelCls = Qwen2_5_VLForConditionalGeneration

        self.model_id = _env("TIMECHAT_HF_MODEL_ID", "wyccccc/TimeChatOnline-7B")
        self.model_name = _env("MODEL_NAME", "timechat_online")

        dev = _env("TIMECHAT_DEVICE", "cuda").lower()
        if dev == "cpu" or not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = "cuda"

        self.attn_impl = _env("TIMECHAT_ATTN_IMPL", "flash_attention_2").strip()
        self.max_frames = _env_int("TIMECHAT_MAX_FRAMES", 64)
        self.sdpa_max_frames = _env_int("TIMECHAT_SDPA_MAX_FRAMES", 6)
        self.video_min_frames = _env_int("TIMECHAT_VIDEO_MIN_FRAMES", 4)

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
            "choice_here,verb_here,noun_here_or_none,short_concise_sentence,STATE_HERE,STATE_HERE</STATE>"
        )

        self.drop_method = _env("TIMECHAT_DROP_METHOD", "feature")
        self.drop_threshold = _env_float("TIMECHAT_DROP_THRESHOLD", 0.5)
        self.drop_absolute = _env("TIMECHAT_DROP_ABSOLUTE", "1") != "0"
        self.require_dtd = _env("TIMECHAT_REQUIRE_DTD", "1") != "0"
        self.save_dr = _env("TIMECHAT_SAVE_DR", "0") != "0"
        self.dr_save_dir = Path(_env("TIMECHAT_DR_SAVE_DIR", "cache/timechat_dr")).expanduser()

        self.vision_enabled = _parse_vision_enabled_from_argv(default=True)

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
        # NEW: OPTIONAL finetune loading switch via env var (same as ROI adapter)
        #   - TIMECHAT_FT_DIR: PEFT adapter directory (e.g., .../final/step_0000706)
        #   - optional extra_trainable.pt: projector/other trainables
        #   - optional merge: TIMECHAT_MERGE_LORA=1
        # -------------------------
        self.ft_dir = _env("TIMECHAT_FT_DIR", "").strip()
        self.ft_verbose = _env("TIMECHAT_FT_VERBOSE", "1") != "0"
        self.merge_lora = _env("TIMECHAT_MERGE_LORA", "0") != "0"

        if self.ft_dir:
            ft = Path(self.ft_dir).expanduser()
            if not ft.exists():
                raise FileNotFoundError(f"TIMECHAT_FT_DIR does not exist: {ft}")

            if PeftModel is None:
                raise ImportError(
                    "peft is not available but TIMECHAT_FT_DIR is set.\n"
                    "Install: pip install -U peft\n"
                    f"TIMECHAT_FT_DIR={str(ft)}"
                )

            if self.ft_verbose:
                print(f"[FT] loading PEFT adapter from TIMECHAT_FT_DIR={str(ft)}")

            try:
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
                    print(f"[FT] extra_trainable.pt not found under {str(ft)} (OK if none was saved)")

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

        # -------------------------
        # NEW: cand_conf mode controls
        # -------------------------
        self._active_pred_flavor: str = ""
        self._active_task: str = ""
        self.cand_conf_max_new_tokens = _env_int("CAND_CONF_MAX_NEW_TOKENS", 4)
        self.cand_conf_force_fallback = _env("CAND_CONF_FORCE_FALLBACK", "1") != "0"

    # -------------------------
    # NEW: cand_conf task gate + out_path rewrite (for sh_pred)
    # -------------------------
    def _task_supports_cand_conf(self, task_name: str) -> bool:
        t = str(task_name or "").strip().lower()
        if not t:
            return False
        if t in {"ms_pred", "ms_rtrv", "sh_pred", "sh_pred_full"}:
            return True
        if t.startswith("sh_pred"):
            return True
        return False

    def _rewrite_out_path_for_cand_conf(self, out_path: Path) -> Path:
        s = out_path.as_posix()
        if "/cand_conf/" in s:
            return out_path

        # rewrite folder
        s2 = s
        for task_key in ["ms_pred", "ms_rtrv", "sh_pred", "sh_pred_full"]:
            if f"/{task_key}/cand/" in s2:
                s2 = s2.replace(f"/{task_key}/cand/", f"/{task_key}/cand_conf/")
                break
            if f"/{task_key}/cand_full/" in s2:
                s2 = s2.replace(f"/{task_key}/cand_full/", f"/{task_key}/cand_conf/")
                break

        p2 = Path(s2)

        # rewrite filename (best-effort, keep other naming unchanged)
        name = p2.name
        if "__cand_full.json" in name:
            name = name.replace("__cand_full.json", "__cand_conf.json")
            p2 = p2.with_name(name)
        elif "__cand.json" in name:
            name = name.replace("__cand.json", "__cand_conf.json")
            p2 = p2.with_name(name)
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

        we_rel = min(float(we_rel), float(t_eval_rel))
        ws_rel = max(0.0, min(float(ws_rel), float(we_rel)))

        if clip_dur_sec > 0:
            ws_rel = float(max(0.0, min(ws_rel, clip_dur_sec)))
            we_rel = float(max(0.0, min(we_rel, clip_dur_sec)))

        lookback_used = float(max(0.0, we_rel - ws_rel))
        return t_eval_rel, ws_rel, we_rel, lookback_used

    @torch.inference_mode()
    def _generate_once(
        self,
        *,
        clip_path: Optional[Path],
        clip_len_sec: float,
        prompt: str,
        kind: str,
        now_schema: str,
        cand: bool,
        video_fps_hint: float,
        memory_keyframe_paths: Optional[List[str]] = None,
    ) -> Tuple[str, List[str], List[float], Optional[float], Optional[float], Dict[str, Any]]:
        diag: Dict[str, Any] = {}
        diag["vision_enabled"] = bool(self.vision_enabled)
        memory_keyframe_paths = [str(p) for p in (memory_keyframe_paths or []) if str(p).strip()]
        diag["memory_keyframes"] = {
            "requested": bool(memory_keyframe_paths),
            "used": bool(memory_keyframe_paths and self.vision_enabled),
            "count": len(memory_keyframe_paths),
            "paths": memory_keyframe_paths,
        }

        if self.deanchor_ans and cand:
            prompt, changed = _deanchor_ans_template(prompt)
            diag["deanchor_ans_applied"] = bool(changed)
        else:
            diag["deanchor_ans_applied"] = False

        # ============================================================
        # NEW: cand_conf mode for ms_pred/ms_rtrv/sh_pred
        #   - prompt rewritten to force output ONLY A/B/C/D (no tags)
        #   - confidence measured from next-token logits at first step
        # ============================================================
        cand_conf_enabled = (str(getattr(self, "_active_pred_flavor", "") or "").strip().lower() == "cand_conf")
        if cand_conf_enabled and self._task_supports_cand_conf(str(getattr(self, "_active_task", "") or "")):
            # detect options even if helper didn't mark it as cand
            has_opts = bool(re.search(r"^\s*[A-D]\.\s+", prompt, flags=re.MULTILINE))
            if has_opts and self.tokenizer is not None:
                def _extract_options_raw_map(text: str) -> Dict[str, str]:
                    out: Dict[str, str] = {}
                    for ln in (text or "").splitlines():
                        m = re.match(r"^\s*([A-D])\.\s+(.*)$", ln.strip(), flags=re.IGNORECASE)
                        if not m:
                            continue
                        k = m.group(1).upper()
                        v = m.group(2).strip()
                        if v:
                            out[k] = v
                    return out

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

                def _choice_token_ids(tokenizer, letter: str) -> Tuple[List[int], Dict[str, Any]]:
                    cand_texts = [letter, " " + letter, "\n" + letter, "\n\n" + letter, "\t" + letter]
                    ids: List[int] = []
                    used: List[str] = []
                    for t in cand_texts:
                        try:
                            enc = tokenizer.encode(t, add_special_tokens=False)
                        except Exception:
                            enc = []
                        if isinstance(enc, list) and len(enc) == 1:
                            tid = int(enc[0])
                            if tid not in ids:
                                ids.append(tid)
                                used.append(t)
                    diag_local = {"letter": letter, "variants_used": used, "single_token_ids": ids, "approx": False}
                    if not ids:
                        try:
                            enc0 = tokenizer.encode(letter, add_special_tokens=False)
                        except Exception:
                            enc0 = []
                        if isinstance(enc0, list) and len(enc0) >= 1:
                            ids = [int(enc0[0])]
                            diag_local["single_token_ids"] = ids
                            diag_local["approx"] = True
                    return ids, diag_local

                def _probe_choice_distribution(inputs: Dict[str, Any]) -> Dict[str, Any]:
                    out = self.model(**inputs)
                    logits = getattr(out, "logits", None)
                    if logits is None or not isinstance(logits, torch.Tensor) or logits.ndim != 3:
                        return {"ok": False, "error": "missing_logits"}
                    last = logits[0, -1, :].float()
                    probs = torch.softmax(last, dim=-1)

                    tok_diag: Dict[str, Any] = {}
                    p_raw: Dict[str, float] = {}
                    token_ids_map: Dict[str, List[int]] = {}

                    for L in ["A", "B", "C", "D"]:
                        ids, d0 = _choice_token_ids(self.tokenizer, L)
                        tok_diag[L] = d0
                        token_ids_map[L] = ids
                        if ids:
                            idx = torch.tensor(ids, device=probs.device, dtype=torch.long)
                            pv = float(probs.index_select(0, idx).sum().item())
                        else:
                            pv = 0.0
                        p_raw[L] = float(pv)

                    mass = float(p_raw["A"] + p_raw["B"] + p_raw["C"] + p_raw["D"])
                    p_cond: Dict[str, Optional[float]] = {}
                    if mass > 0.0:
                        for L in ["A", "B", "C", "D"]:
                            p_cond[L] = float(p_raw[L] / mass)
                    else:
                        for L in ["A", "B", "C", "D"]:
                            p_cond[L] = None

                    chosen = None
                    if mass > 0.0:
                        chosen = max(["A", "B", "C", "D"], key=lambda x: float(p_cond[x] or 0.0))

                    ent = None
                    if mass > 0.0:
                        ee = 0.0
                        for L in ["A", "B", "C", "D"]:
                            v = float(p_cond[L] or 0.0)
                            if v > 0:
                                ee += -v * math.log(v + 1e-12)
                        ent = float(ee)

                    return {
                        "ok": True,
                        "p_raw": p_raw,
                        "mass_abcd": float(mass),
                        "p_cond": p_cond,
                        "chosen_by_p_cond": chosen,
                        "entropy_p_cond": ent,
                        "token_ids_diag": tok_diag,
                        "token_ids_map": token_ids_map,
                    }

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

                # Build short prompt
                opts_raw = _extract_options_raw_map(prompt)
                prompt_conf = _build_ms_cand_conf_prompt(prompt, opts_raw)
                diag["cand_conf_prompt_preview"] = prompt_conf[:1200] if len(prompt_conf) > 1200 else prompt_conf

                if self.vision_enabled:
                    assert clip_path is not None
                    video_uri = clip_path.resolve().as_uri()
                    max_frames = int(self.effective_max_frames)
                    min_frames = int(max(1, min(self.video_min_frames, max_frames)))

                    content: List[Dict[str, Any]] = []
                    for keyframe_path in memory_keyframe_paths:
                        content.append({"type": "image", "image": Path(keyframe_path).expanduser().resolve().as_uri()})
                    content.append({"type": "video", "video": video_uri, "max_frames": max_frames, "min_frames": min_frames, "fps": float(video_fps_hint)})
                    content.append({"type": "text", "text": prompt_conf})

                    messages = [{
                        "role": "user",
                        "content": content,
                    }]

                    text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )

                    # Move tensors to model device, BUT keep second_per_grid_ts on CPU (Scheme A)
                    if hasattr(inputs, "to"):
                        inputs = inputs.to(self.model.device)
                    else:
                        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

                    try:
                        if "second_per_grid_ts" in inputs and hasattr(inputs["second_per_grid_ts"], "to"):
                            inputs["second_per_grid_ts"] = inputs["second_per_grid_ts"].to("cpu")
                    except Exception:
                        pass
                else:
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_conf},
                        ],
                    }]
                    text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    inputs = self.processor(
                        text=[text],
                        padding=True,
                        return_tensors="pt",
                    )
                    if hasattr(inputs, "to"):
                        inputs = inputs.to(self.model.device)
                    else:
                        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

                prompt_len = int(inputs["input_ids"].shape[1])

                # Probe next-token distribution
                probe = _probe_choice_distribution(inputs)
                diag["cand_conf_probe"] = probe

                # Small controlled generation (optional; mainly for sanity)
                gen_kwargs: Dict[str, Any] = dict(
                    max_new_tokens=int(max(1, min(int(self.cand_conf_max_new_tokens), 8))),
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

                stopping = StoppingCriteriaList([_StopOnFirstABCD(self.tokenizer, prompt_len=prompt_len, max_decode_tokens=16)])
                gen_kwargs["stopping_criteria"] = stopping

                if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token_id", None) is not None:
                    gen_kwargs["pad_token_id"] = int(self.tokenizer.eos_token_id)

                if self.vision_enabled:
                    dtd_kwargs: Dict[str, Any] = dict(
                        drop_method=str(self.drop_method),
                        drop_threshold=float(self.drop_threshold),
                        drop_absolute=bool(self.drop_absolute),
                    )
                    dr_save_path = None
                    if self.save_dr:
                        self.dr_save_dir.mkdir(parents=True, exist_ok=True)
                        dr_save_path = str(self.dr_save_dir / f"drop_{int(time.time())}_pid{os.getpid()}.jsonl")
                        dtd_kwargs["dr_save_path"] = dr_save_path
                    diag["dtd"] = {
                        "drop_method": str(self.drop_method),
                        "drop_threshold": float(self.drop_threshold),
                        "drop_absolute": bool(self.drop_absolute),
                        "dr_save_path": dr_save_path,
                    }
                else:
                    dtd_kwargs = {}
                    diag["dtd"] = {
                        "disabled_because_vision_off": True,
                    }

                t0 = time.time()
                try:
                    if self.vision_enabled:
                        try:
                            out = self.model.generate(**inputs, **gen_kwargs, **dtd_kwargs)
                        except TypeError as e:
                            if self.require_dtd:
                                raise TypeError(
                                    "model.generate() does not accept DTD kwargs (drop_method/drop_threshold/drop_absolute).\n"
                                    "This means you are NOT running the TimeChat-Online DTD-enabled model class.\n"
                                    "Double-check TIMECHAT_REPO_ROOT and that you import Qwen2_5_VLForConditionalGeneration from eval.qwen2_5_vl.\n"
                                    f"Original error: {repr(e)}"
                                ) from e
                            out = self.model.generate(**inputs, **gen_kwargs)
                    else:
                        out = self.model.generate(**inputs, **gen_kwargs)
                except Exception:
                    raise
                diag["latency_generate_sec"] = float(time.time() - t0)

                seq = out.sequences[0]
                gen_ids = seq[prompt_len:]
                tok = self.tokenizer
                resp_text_raw = tok.decode(gen_ids.tolist(), skip_special_tokens=True).strip() if tok is not None else ""

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

                return (letter if letter else resp_text_raw.strip()), gen_tokens, gen_token_probs, sent_logp, mean_logp, diag
        # =================== end cand_conf branch ===================

        if self.vision_enabled:
            assert clip_path is not None
            video_uri = clip_path.resolve().as_uri()
            max_frames = int(self.effective_max_frames)
            min_frames = int(max(1, min(self.video_min_frames, max_frames)))

            content: List[Dict[str, Any]] = []
            for keyframe_path in memory_keyframe_paths:
                content.append({"type": "image", "image": Path(keyframe_path).expanduser().resolve().as_uri()})
            content.append({"type": "video", "video": video_uri, "max_frames": max_frames, "min_frames": min_frames, "fps": float(video_fps_hint)})
            content.append({"type": "text", "text": prompt})

            messages = [{
                "role": "user",
                "content": content,
            }]

            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            # ====== Move tensors to model device, BUT keep second_per_grid_ts on CPU (Scheme A) ======
            if hasattr(inputs, "to"):
                inputs = inputs.to(self.model.device)
            else:
                inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            # Critical: get_rope_index() uses CPU torch.arange; keeping this tensor on CPU avoids cpu/cuda mix.
            try:
                if "second_per_grid_ts" in inputs and hasattr(inputs["second_per_grid_ts"], "to"):
                    inputs["second_per_grid_ts"] = inputs["second_per_grid_ts"].to("cpu")
            except Exception:
                pass
            # ======================================================================================

            dtd_kwargs: Dict[str, Any] = dict(
                drop_method=str(self.drop_method),
                drop_threshold=float(self.drop_threshold),
                drop_absolute=bool(self.drop_absolute),
            )
            dr_save_path = None
            if self.save_dr:
                self.dr_save_dir.mkdir(parents=True, exist_ok=True)
                dr_save_path = str(self.dr_save_dir / f"drop_{int(time.time())}_pid{os.getpid()}.jsonl")
                dtd_kwargs["dr_save_path"] = dr_save_path
            diag["dtd"] = {
                "drop_method": str(self.drop_method),
                "drop_threshold": float(self.drop_threshold),
                "drop_absolute": bool(self.drop_absolute),
                "dr_save_path": dr_save_path,
            }
        else:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }]

            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )

            if hasattr(inputs, "to"):
                inputs = inputs.to(self.model.device)
            else:
                inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            dtd_kwargs = {}
            diag["dtd"] = {
                "disabled_because_vision_off": True,
            }

        prompt_len = int(inputs["input_ids"].shape[1])

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

        if self.stop_on_close_tags and self.tokenizer is not None and kind in {"now", "past", "pred"}:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([
                _StopOnCloseTags(self.tokenizer, prompt_len=prompt_len, stop_strs_upper=self.close_tag_strs)
            ])

        if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token_id", None) is not None:
            gen_kwargs["pad_token_id"] = int(self.tokenizer.eos_token_id)

        t0 = time.time()
        if self.vision_enabled:
            try:
                out = self.model.generate(**inputs, **gen_kwargs, **dtd_kwargs)
            except TypeError as e:
                if self.require_dtd:
                    raise TypeError(
                        "model.generate() does not accept DTD kwargs (drop_method/drop_threshold/drop_absolute).\n"
                        "This means you are NOT running the TimeChat-Online DTD-enabled model class.\n"
                        "Double-check TIMECHAT_REPO_ROOT and that you import Qwen2_5_VLForConditionalGeneration from eval.qwen2_5_vl.\n"
                        f"Original error: {repr(e)}"
                    ) from e
                out = self.model.generate(**inputs, **gen_kwargs)
        else:
            out = self.model.generate(**inputs, **gen_kwargs)
        diag["latency_generate_sec"] = float(time.time() - t0)

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

        return response_text, gen_tokens, gen_token_probs, sent_logp, mean_logp, diag

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

        # NEW: decide active pred flavor (cand / cand_conf / etc.)
        params = qs.get("params", {}) if isinstance(qs.get("params"), dict) else {}
        env_flavor = os.environ.get("PRED_FLAVOR", "").strip().lower()
        qs_flavor = str(params.get("pred_flavor", "") or "").strip().lower()
        self._active_pred_flavor = env_flavor if env_flavor else qs_flavor
        self._active_task = str(task or "").strip().lower()

        # NEW: reroute output file for cand_conf to avoid overwriting /cand/
        if self._active_pred_flavor == "cand_conf" and self._task_supports_cand_conf(self._active_task):
            out_path = self._rewrite_out_path_for_cand_conf(out_path)

        vm = qs.get("video_metadata", {}) if isinstance(qs.get("video_metadata"), dict) else {}
        video_uid = str(vm.get("video_uid", qs.get("video_uid", ""))).strip()

        samples = qs.get("samples", [])
        if not isinstance(samples, list):
            raise ValueError(f"Effective queryset 'samples' must be list: {queryset_path}")

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

        if self.vision_enabled:
            vr = _get_vr(video_path)
            fps = _get_fps(vr)
            n_total = len(vr)
            clip_dur_sec = float(n_total / max(fps, 1e-6))
        else:
            fps = 0.0
            clip_dur_sec = 0.0

        out_obj: Dict[str, Any] = {
            "dataset": qs.get("dataset", "Ego4D"),
            "task": task,
            "video_uid": video_uid,
            "model_name": self.model_name,
            "model_id": self.model_id,
            "vision_enabled": bool(self.vision_enabled),
            "source_queryset": str(queryset_path),
            "video_clip": str(video_path),
            "generated_at_unix": float(time.time()),
            "params": qs.get("params", {}),
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
            prompt, memory_diag, memory_keyframe_paths = build_prompt_with_memory(s, prompt)

            # Caption+dense memory is the default TimeChat memory mode.
            # Raw sparse keyframe images are disabled by default because TimeChat-Online DTD
            # can crash on mixed image+video patch grids. Enable only explicitly.
            use_raw_memory_keyframes = _env("TIMECHAT_USE_RAW_MEMORY_KEYFRAMES", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
            if not use_raw_memory_keyframes:
                if memory_diag:
                    if memory_keyframe_paths:
                        memory_diag["keyframes_available"] = True
                        memory_diag["keyframes_disabled_by_env"] = True
                    memory_diag["keyframes_used"] = False
                    memory_diag["keyframe_count_used"] = 0
                memory_keyframe_paths = []

            kind, now_schema, cand = _infer_kind_and_schema(prompt)

            t_eval_rel, ws_rel, we_rel, lookback_used = self._infer_window_rel(qs, s, clip_dur_sec=clip_dur_sec)
            clip_len = float(max(0.0, we_rel - ws_rel))

            if self.vision_enabled:
                # ====== enforce a minimum clip duration to avoid empty MP4 from ffmpeg ======
                min_clip_sec = float(_env_float("TIMECHAT_MIN_CLIP_SEC", 0.25))
                eps = 1e-3
                if clip_len < min_clip_sec:
                    if we_rel > 0.0:
                        ws2 = float(max(0.0, we_rel - min_clip_sec))
                        we2 = float(we_rel)
                    else:
                        ws2 = float(ws_rel)
                        if clip_dur_sec > 0.0:
                            we2 = float(min(ws_rel + min_clip_sec, clip_dur_sec))
                        else:
                            we2 = float(ws_rel + min_clip_sec)

                    if (we2 - ws2) < eps:
                        we2 = float(ws2 + max(min_clip_sec, eps))

                    ws_rel, we_rel = float(ws2), float(we2)
                    clip_len = float(max(0.0, we_rel - ws_rel))
                    lookback_used = float(max(0.0, we_rel - ws_rel))
                # =================================================================================

                clip_out: Optional[Path] = _make_clip_path(video_uid=video_uid, idx=idx, ws_rel=ws_rel, we_rel=we_rel)
                if not clip_out.exists():
                    _run_ffmpeg_cut(video_path, clip_out, start_sec=ws_rel, end_sec=we_rel)

                video_fps_hint = _pick_video_fps(clip_len_sec=clip_len, max_frames=int(self.effective_max_frames))
            else:
                clip_out = None
                video_fps_hint = 0.0

            t0 = time.time()
            resp_text, gen_tokens, gen_token_probs, sent_logp, mean_logp, gen_diag = self._generate_once(
                clip_path=clip_out,
                clip_len_sec=clip_len,
                prompt=prompt,
                kind=kind,
                now_schema=now_schema,
                cand=cand,
                video_fps_hint=video_fps_hint,
                memory_keyframe_paths=memory_keyframe_paths,
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
                            clip_path=clip_out,
                            clip_len_sec=clip_len,
                            prompt=retry_prompt,
                            kind=kind,
                            now_schema=now_schema,
                            cand=cand,
                            video_fps_hint=video_fps_hint,
                            memory_keyframe_paths=memory_keyframe_paths,
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

            if self._active_pred_flavor == "cand_conf" and self._task_supports_cand_conf(self._active_task):
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
                    "vision_enabled": bool(self.vision_enabled),
                    "kind": kind,
                    "now_schema": now_schema if kind == "now" else None,
                    "candidate": bool(cand),
                    "pred_flavor_active": self._active_pred_flavor,
                    "retry_triggered": bool(retry_triggered),
                    "retry_reason": retry_reason,
                    "attempts": attempts,
                    "latency_first_pass_sec": float(attempts[0].get("latency_sec", latency_total)) if attempts else latency_total,
                    "latency_total_sec": latency_total,
                    "fps": float(fps),
                    "clip_dur_sec": float(clip_dur_sec),
                    "window_start_rel": float(ws_rel),
                    "window_end_rel": float(we_rel),
                    "lookback_used_sec": float(lookback_used),
                    "temp_clip_path": str(clip_out) if clip_out is not None else None,
                    "temp_clip_len_sec": float(clip_len) if clip_out is not None else 0.0,
                    "video_fps_hint": float(video_fps_hint),
                    "attn_impl_effective": str(getattr(self, "attn_impl_effective", "")),
                    "effective_max_frames": int(self.effective_max_frames),
                },
                "raw": None,
            }
            if memory_diag:
                used = any(
                    (attempt.get("diag", {}).get("memory_keyframes", {}) or {}).get("used")
                    for attempt in attempts
                )
                memory_diag["keyframes_used"] = bool(used)
                memory_diag["keyframe_count_used"] = len(memory_keyframe_paths or []) if used else 0
                rec["diagnostics"]["memory"] = memory_diag
                if isinstance(s.get("memory"), dict):
                    rec["memory"] = s.get("memory")

            out_obj["samples"].append(rec)
            _dump_json(out_path, out_obj)

        _dump_json(out_path, out_obj)
        return out_path


def create_adapter():
    return _TimeChatOnlineAdapter()
