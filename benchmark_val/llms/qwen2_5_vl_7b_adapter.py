#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-VL-7B-Instruct task adapter (for your model-agnostic runner/helper).

继承你在 video_llava_7b adapter 上已经做好的：
  - strict-online window 口径
  - prompt compaction
  - now schema-aware (state-only / action-mcq) 校验与 retry
  - stop-on-close-tags、防乱说
  - cand 选项解析与 VN-copy 校验
  - ms_* cand 的 postproc（仅 past/pred）

仅做“跑 Qwen2.5-VL”所需的点对点改造：
  - 使用 Qwen2_5_VLForConditionalGeneration + AutoProcessor
  - encode 输入改为 Qwen 的 messages/chat_template + qwen_vl_utils.process_vision_info
  - 继续沿用你的 decord 抽帧与 window 截取：frames -> 临时 jpg 列表 -> {"type":"video","video":[...]} 输入给 Qwen
"""

from __future__ import annotations

import os
import re
import json
import time
import math
import subprocess
import tempfile
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
)

try:
    # transformers 需要较新版本才有这个类
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception as e:
    raise ImportError(
        "Cannot import Qwen2_5_VLForConditionalGeneration. "
        "Please upgrade transformers (recommend latest) and retry.\n"
        "Example: pip install -U transformers accelerate\n"
        f"Original error: {repr(e)}"
    )

try:
    from qwen_vl_utils import process_vision_info  # type: ignore
except Exception as e:
    raise ImportError(
        "Cannot import qwen_vl_utils.process_vision_info. "
        "Please install qwen-vl-utils.\n"
        "Example: pip install -U qwen-vl-utils\n"
        f"Original error: {repr(e)}"
    )

try:
    from caption_cache import caption_record_path, load_caption_record
except Exception:
    repo_root_for_cache = Path(__file__).resolve().parents[1]
    if str(repo_root_for_cache) not in sys.path:
        sys.path.insert(0, str(repo_root_for_cache))
    from caption_cache import caption_record_path, load_caption_record

FALLBACK_CHAT_TEMPLATE = r"""{% for message in messages %}
{% if message['role'] == 'user' %}USER: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}ASSISTANT: {{ message['content'] }}
{% elif message['role'] == 'system' %}SYSTEM: {{ message['content'] }}
{% endif %}{% endfor %}
{% if add_generation_prompt %}ASSISTANT:{% endif %}"""


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
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except Exception:
        return default


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


def _truncate_text(s: str, max_chars: int) -> str:
    if s is None:
        return ""
    s = str(s)
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    head = max(0, max_chars // 2)
    tail = max(0, max_chars - head)
    return s[:head] + "\n...[TRUNCATED_CHARS]...\n" + s[-tail:]


# -------------------------
# Video decoding / sampling
# -------------------------

def _load_frames_decord(video_path: Path, start_sec: float, end_sec: float, num_frames: int) -> Tuple[np.ndarray, float]:
    try:
        from decord import VideoReader, cpu  # type: ignore
    except Exception as e:
        raise RuntimeError("decord not available; please `pip install decord`") from e

    vr = VideoReader(str(video_path), ctx=cpu(0))
    fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 30.0
    n_total = len(vr)
    clip_dur = float(n_total / max(fps, 1e-6))

    start_sec = float(max(0.0, start_sec))
    end_sec = float(min(max(start_sec, end_sec), clip_dur))

    start_idx = int(round(start_sec * fps))
    end_idx = int(round(end_sec * fps))
    start_idx = max(0, min(start_idx, n_total - 1))
    end_idx = max(start_idx + 1, min(end_idx, n_total))  # end exclusive

    if num_frames <= 1:
        idxs = np.array([start_idx], dtype=np.int64)
    else:
        idxs = np.linspace(start_idx, end_idx - 1, num_frames).astype(np.int64)
        idxs = np.clip(idxs, 0, n_total - 1)

    frames = vr.get_batch(idxs).asnumpy()  # [T,H,W,3] RGB
    return frames, clip_dur


def _load_frames_decord_with_timestamps(video_path: Path, start_sec: float, end_sec: float, num_frames: int) -> Tuple[np.ndarray, List[float], float]:
    try:
        from decord import VideoReader, cpu  # type: ignore
    except Exception as e:
        raise RuntimeError("decord not available; please `pip install decord`") from e

    vr = VideoReader(str(video_path), ctx=cpu(0))
    fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 30.0
    n_total = len(vr)
    clip_dur = float(n_total / max(fps, 1e-6))
    if n_total <= 0:
        raise RuntimeError(f"Video has no frames: {video_path}")

    start_sec = float(max(0.0, start_sec))
    end_sec = float(min(max(start_sec, end_sec), clip_dur))
    start_idx = int(math.floor(start_sec * fps + 1e-9))
    end_idx = int(math.floor(end_sec * fps + 1e-9))
    start_idx = max(0, min(start_idx, n_total - 1))
    end_idx = max(start_idx, min(end_idx, n_total - 1))

    num_frames = max(1, int(num_frames))
    if num_frames <= 1 or start_idx == end_idx:
        idxs = np.array([end_idx], dtype=np.int64)
    else:
        idxs = np.linspace(start_idx, end_idx, num_frames).round().astype(np.int64)
        idxs = np.clip(idxs, start_idx, end_idx)

    frames = vr.get_batch(idxs).asnumpy()
    timestamps = [float(int(i) / max(fps, 1e-6)) for i in idxs.tolist()]
    timestamps = [float(min(ts, end_sec)) for ts in timestamps]
    return frames, timestamps, clip_dur


def _load_keyframe_images(paths: List[str]) -> np.ndarray:
    frames: List[np.ndarray] = []
    for path in paths:
        im = Image.open(path).convert("RGB")
        frames.append(np.asarray(im, dtype=np.uint8))
    if not frames:
        return np.zeros((0, 1, 1, 3), dtype=np.uint8)
    return np.stack(frames, axis=0)


def _ensure_min_window(start_sec: float, end_sec: float, min_len: float) -> Tuple[float, float]:
    """
    strict-online: 只向过去扩展窗口，保持 end_sec 不变（不引入 look-ahead）。
    """
    if min_len <= 0:
        return start_sec, end_sec
    if end_sec - start_sec >= min_len:
        return start_sec, end_sec
    end2 = float(end_sec)
    start2 = float(max(0.0, end2 - float(min_len)))
    return start2, end2


# -------------------------
# Prompt compaction (保留你原来的，尽量不动)
# -------------------------

def _important_lines_for_all_tasks(lines: List[str]) -> List[str]:
    keep: List[str] = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if i == 0 and s:
            keep.append(ln)
            continue
        if not s:
            continue
        if re.search(r"output\s+exactly|do\s+not\s+output|rules?:|options?:", s, flags=re.IGNORECASE):
            keep.append(ln)
            continue
        if re.search(r"<\s*(now|past|future)\s*>|</\s*(now|past|future)\s*>", s, flags=re.IGNORECASE):
            keep.append(ln)
            continue
        if re.search(r"<\s*(state|verb|noun|desc|conf|ans)\s*>|</\s*(state|verb|noun|desc|conf|ans)\s*>", s, flags=re.IGNORECASE):
            keep.append(ln)
            continue
        if re.match(r"^[A-D]\.\s+", s):
            keep.append(ln)
            continue
        if s.startswith("Candidates:"):
            keep.append(ln)
            continue
    return keep


def _truncate_by_tokens(tokenizer, text: str, max_tokens: int, head_tokens: int, tail_tokens: int) -> str:
    if tokenizer is None or max_tokens <= 0:
        return text
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text

    head = max(0, int(head_tokens))
    tail = max(0, int(tail_tokens))
    if head + tail > max_tokens:
        head = min(head, max_tokens // 3)
        tail = max_tokens - head

    if tail > 0:
        out = tokenizer.decode(ids[:head], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip() \
              + "\n\n[TRUNCATED]\n\n" \
              + tokenizer.decode(ids[-tail:], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    else:
        out = tokenizer.decode(ids[:max_tokens], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return out.strip()


def _compact_prompt_task_agnostic(tokenizer, prompt_text: str, max_tokens: int, head_tokens: int, tail_tokens: int) -> str:
    s = (prompt_text or "").strip()
    if not s or tokenizer is None or max_tokens <= 0:
        return s
    ids = tokenizer.encode(s, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return s

    lines = s.splitlines()
    kept_lines = _important_lines_for_all_tasks(lines)
    compact = "\n".join(kept_lines).strip() if kept_lines else s
    ids2 = tokenizer.encode(compact, add_special_tokens=False)
    if len(ids2) <= max_tokens:
        return compact
    return _truncate_by_tokens(tokenizer, compact, max_tokens=max_tokens, head_tokens=head_tokens, tail_tokens=tail_tokens)


# -------------------------
# Encoding with auto frame downscale to fit context
# -------------------------

def _infer_model_max_len(tokenizer, model) -> int:
    # tokenizer.model_max_length 有时是超大哨兵值；以 config 为准
    maxlen = None
    if tokenizer is not None:
        try:
            ml = int(getattr(tokenizer, "model_max_length", 0))
            if 128 <= ml <= 32768:
                maxlen = ml
        except Exception:
            pass

    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("max_position_embeddings", "max_sequence_length", "seq_length"):
            v = getattr(cfg, attr, None)
            if isinstance(v, int) and 128 <= v <= 32768:
                maxlen = v if maxlen is None else min(maxlen, v)
        tc = getattr(cfg, "text_config", None)
        if tc is not None:
            v = getattr(tc, "max_position_embeddings", None)
            if isinstance(v, int) and 128 <= v <= 32768:
                maxlen = v if maxlen is None else min(maxlen, v)

    return int(maxlen or 4096)


def _subsample_frames_uniform(frames: np.ndarray, target_t: int) -> np.ndarray:
    if not isinstance(frames, np.ndarray) or frames.ndim != 4:
        return frames
    T = int(frames.shape[0])
    if target_t >= T:
        return frames
    if target_t <= 1:
        return frames[[T // 2]]
    idxs = np.linspace(0, T - 1, target_t).round().astype(np.int64)
    idxs = np.clip(idxs, 0, T - 1)
    return frames[idxs]


def _frames_to_temp_jpgs(frames_rgb: np.ndarray, tmp_dir: Path) -> List[str]:
    """
    Qwen 的 video(list) 输入用本地帧路径最稳：["/tmp/.../0000.jpg", ...]
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_paths: List[str] = []
    T = int(frames_rgb.shape[0])
    for i in range(T):
        p = tmp_dir / f"{i:04d}.jpg"
        im = Image.fromarray(frames_rgb[i].astype(np.uint8), mode="RGB")
        im.save(str(p), format="JPEG", quality=90)
        out_paths.append(str(p))
    return out_paths


def _encode_inputs_with_autofit(
    processor,
    tokenizer,
    prompt_text: str,
    frames: np.ndarray,
    device: torch.device,
    *,
    max_input_tokens: int,
) -> Tuple[Dict[str, torch.Tensor], int, str, Dict[str, Any]]:
    """
    Qwen2.5-VL encode：
      messages -> processor.apply_chat_template -> process_vision_info -> processor(...)
    并保持你原来的“自动降帧”逻辑：如果 input_ids 超过 max_input_tokens，则减少帧数重试。

    返回：moved_inputs, frames_used, prompt_final, encode_diag
    """
    prompt_text = (prompt_text or "").strip()

    T = int(frames.shape[0]) if isinstance(frames, np.ndarray) else 0
    candidates: List[int] = []
    for c in [T, 12, 8, 6, 4, 2, 1]:
        if c >= 1 and (not candidates or candidates[-1] != c):
            if c <= max(1, T):
                candidates.append(c)

    tries: List[Dict[str, Any]] = []
    last_inputs = None
    last_t = None
    last_in_len = None
    last_prompt = None
    last_chat_error = None

    for c in candidates:
        frames_c = _subsample_frames_uniform(frames, int(c)) if isinstance(frames, np.ndarray) else frames

        tmp_root = Path(_env("QWEN_TMP_FRAMES_DIR", "/tmp/qwen_vl_frames"))
        tmp_dir = Path(tempfile.mkdtemp(prefix="qwen_vl_", dir=str(tmp_root)))
        try:
            frame_paths = _frames_to_temp_jpgs(frames_c, tmp_dir)

            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": frame_paths},
                    {"type": "text", "text": prompt_text},
                ]
            }]

            prompt = None
            chat_mode = "unknown"
            chat_error = None

            # Qwen 推荐 processor.apply_chat_template
            if hasattr(processor, "apply_chat_template"):
                try:
                    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    chat_mode = "processor.apply_chat_template"
                except Exception as e:
                    prompt = None
                    chat_error = repr(e)
                    chat_mode = "processor.apply_chat_template_failed"

            # fallback（尽量不用，但保证可跑）
            if not prompt:
                user_text = f"<video>\n{prompt_text}"
                prompt = f"USER: {user_text}\nASSISTANT:"
                chat_mode = "manual_USER_ASSISTANT" if chat_mode == "unknown" else chat_mode

            # process_vision_info：不同版本可能不支持 return_video_kwargs，这里做兼容
            image_inputs = None
            video_inputs = None
            video_kwargs: Dict[str, Any] = {}

            try:
                # 尝试新签名
                image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)  # type: ignore
                if video_kwargs is None:
                    video_kwargs = {}
            except TypeError:
                # 老签名
                image_inputs, video_inputs = process_vision_info(messages)  # type: ignore
                video_kwargs = {}
            except Exception as e:
                raise RuntimeError(f"process_vision_info failed: {repr(e)}")

            # processor(...)：不同版本 processor 可能不接受某些 kwargs，这里先尝试带上 video_kwargs，不行再降级
            try:
                inputs = processor(
                    text=[prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    **(video_kwargs or {}),
                )
            except TypeError:
                inputs = processor(
                    text=[prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )

            if "input_ids" in inputs and isinstance(inputs["input_ids"], torch.Tensor):
                in_len = int(inputs["input_ids"].shape[-1])
            else:
                in_len = 10**9

            tries.append({"frames": int(c), "input_len": int(in_len)})

            last_inputs, last_t, last_in_len, last_prompt = inputs, int(c), int(in_len), str(prompt)
            last_chat_error = chat_error

            if in_len <= int(max_input_tokens):
                break

        finally:
            # inputs 已经是 tensor 化的，不再需要帧文件
            try:
                shutil.rmtree(str(tmp_dir), ignore_errors=True)
            except Exception:
                pass

    assert last_inputs is not None and last_t is not None and last_in_len is not None and last_prompt is not None

    prompt_tok_len = None
    if tokenizer is not None:
        try:
            prompt_tok_len = int(len(tokenizer.encode(last_prompt, add_special_tokens=False)))
        except Exception:
            prompt_tok_len = None

    encode_diag: Dict[str, Any] = {
        "chat_mode": "processor.apply_chat_template",
        "chat_error": last_chat_error,
        "prompt_tok_len_text_only": prompt_tok_len,
        "max_input_tokens_budget": int(max_input_tokens),
        "tries": tries,
        "chosen_frames": int(last_t),
        "chosen_input_len": int(last_in_len),
    }

    moved = {k: v.to(device) for k, v in last_inputs.items() if isinstance(v, torch.Tensor)}
    return moved, int(last_t), str(last_prompt), encode_diag


def _uniform_indices(n: int, target: int) -> List[int]:
    if n <= 0 or target <= 0:
        return []
    if target >= n:
        return list(range(n))
    if target == 1:
        return [n - 1]
    idxs = np.linspace(0, n - 1, target).round().astype(np.int64)
    out: List[int] = []
    seen = set()
    for idx in idxs.tolist():
        idx = max(0, min(n - 1, int(idx)))
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def _select_memory_frame_indices(roles: List[str], target_total: int) -> List[int]:
    n = len(roles)
    if target_total >= n:
        return list(range(n))
    if target_total <= 0:
        return []
    dense = [i for i, role in enumerate(roles) if role == "dense_frame"]
    sparse = [i for i, role in enumerate(roles) if role == "sparse_keyframe"]
    other = [i for i, role in enumerate(roles) if role not in {"dense_frame", "sparse_keyframe"}]

    if len(dense) >= target_total:
        return [dense[i] for i in _uniform_indices(len(dense), target_total)]

    selected = list(dense)
    remain = target_total - len(selected)
    if remain > 0 and sparse:
        selected.extend([sparse[i] for i in _uniform_indices(len(sparse), min(remain, len(sparse)))])
    remain = target_total - len(selected)
    if remain > 0 and other:
        selected.extend([other[i] for i in _uniform_indices(len(other), min(remain, len(other)))])
    return sorted(selected)


def _encode_inputs_with_autofit_memory(
    processor,
    tokenizer,
    prompt_text: str,
    frames: np.ndarray,
    visual_roles: List[str],
    device: torch.device,
    *,
    max_input_tokens: int,
) -> Tuple[Dict[str, torch.Tensor], int, str, Dict[str, Any]]:
    prompt_text = (prompt_text or "").strip()
    T = int(frames.shape[0]) if isinstance(frames, np.ndarray) else 0
    roles = list(visual_roles or [])
    if len(roles) != T:
        roles = ["frame"] * T
    dense_count = sum(1 for role in roles if role == "dense_frame")
    sparse_count = sum(1 for role in roles if role == "sparse_keyframe")

    raw_candidates = [
        T,
        dense_count + min(sparse_count, 12),
        dense_count + min(sparse_count, 8),
        dense_count + min(sparse_count, 4),
        dense_count,
        12,
        8,
        6,
        4,
        2,
        1,
    ]
    candidates: List[int] = []
    for c in raw_candidates:
        c = int(c)
        if c >= 1 and c <= max(1, T) and c not in candidates:
            candidates.append(c)

    tries: List[Dict[str, Any]] = []
    last_inputs = None
    last_t = None
    last_in_len = None
    last_prompt = None
    last_chat_error = None
    last_selected_indices: List[int] = []
    last_selected_roles: List[str] = []

    for c in candidates:
        selected_indices = _select_memory_frame_indices(roles, int(c))
        if not selected_indices:
            selected_indices = [max(0, T - 1)]
        frames_c = frames[np.array(selected_indices, dtype=np.int64)]
        roles_c = [roles[i] for i in selected_indices]

        tmp_root = Path(_env("QWEN_TMP_FRAMES_DIR", "/tmp/qwen_vl_frames"))
        tmp_dir = Path(tempfile.mkdtemp(prefix="qwen_vl_", dir=str(tmp_root)))
        try:
            frame_paths = _frames_to_temp_jpgs(frames_c, tmp_dir)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": frame_paths},
                    {"type": "text", "text": prompt_text},
                ]
            }]

            prompt = None
            chat_mode = "unknown"
            chat_error = None
            if hasattr(processor, "apply_chat_template"):
                try:
                    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    chat_mode = "processor.apply_chat_template"
                except Exception as e:
                    prompt = None
                    chat_error = repr(e)
                    chat_mode = "processor.apply_chat_template_failed"
            if not prompt:
                user_text = f"<video>\n{prompt_text}"
                prompt = f"USER: {user_text}\nASSISTANT:"
                chat_mode = "manual_USER_ASSISTANT" if chat_mode == "unknown" else chat_mode

            image_inputs = None
            video_inputs = None
            video_kwargs: Dict[str, Any] = {}
            try:
                image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)  # type: ignore
                if video_kwargs is None:
                    video_kwargs = {}
            except TypeError:
                image_inputs, video_inputs = process_vision_info(messages)  # type: ignore
                video_kwargs = {}
            except Exception as e:
                raise RuntimeError(f"process_vision_info failed: {repr(e)}")

            try:
                inputs = processor(
                    text=[prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    **(video_kwargs or {}),
                )
            except TypeError:
                inputs = processor(
                    text=[prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )

            if "input_ids" in inputs and isinstance(inputs["input_ids"], torch.Tensor):
                in_len = int(inputs["input_ids"].shape[-1])
            else:
                in_len = 10**9

            tries.append({
                "frames": int(len(selected_indices)),
                "input_len": int(in_len),
                "selected_indices": selected_indices,
                "selected_roles": roles_c,
            })
            last_inputs, last_t, last_in_len, last_prompt = inputs, int(len(selected_indices)), int(in_len), str(prompt)
            last_chat_error = chat_error
            last_selected_indices = selected_indices
            last_selected_roles = roles_c
            if in_len <= int(max_input_tokens):
                break
        finally:
            try:
                shutil.rmtree(str(tmp_dir), ignore_errors=True)
            except Exception:
                pass

    assert last_inputs is not None and last_t is not None and last_in_len is not None and last_prompt is not None

    prompt_tok_len = None
    if tokenizer is not None:
        try:
            prompt_tok_len = int(len(tokenizer.encode(last_prompt, add_special_tokens=False)))
        except Exception:
            prompt_tok_len = None

    encode_diag: Dict[str, Any] = {
        "chat_mode": "processor.apply_chat_template",
        "chat_error": last_chat_error,
        "prompt_tok_len_text_only": prompt_tok_len,
        "max_input_tokens_budget": int(max_input_tokens),
        "tries": tries,
        "chosen_frames": int(last_t),
        "chosen_input_len": int(last_in_len),
        "memory_visual_input": True,
        "initial_roles": roles,
        "chosen_indices": last_selected_indices,
        "chosen_roles": last_selected_roles,
        "dropped_sparse_keyframes": int(max(0, sparse_count - sum(1 for r in last_selected_roles if r == "sparse_keyframe"))),
        "dense_frames_preserved": int(sum(1 for r in last_selected_roles if r == "dense_frame")),
    }

    moved = {k: v.to(device) for k, v in last_inputs.items() if isinstance(v, torch.Tensor)}
    return moved, int(last_t), str(last_prompt), encode_diag


# -------------------------
# Self-heal corrupted interval clips (moov atom not found)
# -------------------------

def _run_ffmpeg(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (rc={p.returncode}).\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDERR(last 4000 chars):\n{p.stderr[-4000:]}"
        )


def _resolve_source_video(video_root: Path, video_uid: str) -> Path:
    p = video_root / f"{video_uid}.mp4"
    if p.exists():
        return p
    hits = list(video_root.rglob(f"{video_uid}.mp4"))
    if hits:
        hits.sort(key=lambda x: x.as_posix())
        return hits[0]
    raise FileNotFoundError(f"Cannot find source video for uid={video_uid} under {video_root}")


def _recut_interval_clip_from_source(*, src_video: Path, dst_clip: Path, start_sec: float, end_sec: float) -> None:
    dst_clip.parent.mkdir(parents=True, exist_ok=True)

    start_sec = float(max(0.0, start_sec))
    end_sec = float(max(start_sec, end_sec))
    dur = end_sec - start_sec

    tmp = dst_clip.with_suffix(dst_clip.suffix + ".tmp")
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
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(tmp),
    ]
    _run_ffmpeg(cmd)

    if not tmp.exists() or tmp.stat().st_size < 1024:
        raise RuntimeError(f"ffmpeg produced an invalid clip: {tmp} (size<1KB)")

    os.replace(str(tmp), str(dst_clip))


# -------------------------
# Adapter implementation
# -------------------------

class _Qwen25VLAdapter:
    def __init__(self) -> None:
        self.model_id = _env("MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
        trust_remote_code = _env("TRUST_REMOTE_CODE", "1") != "0"
        device_map = _env("DEVICE_MAP", "auto")

        td = _env("TORCH_DTYPE", "bfloat16").lower()
        if td == "bfloat16":
            torch_dtype = torch.bfloat16
        elif td == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float16

        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=trust_remote_code)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

        self.device = getattr(self.model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = getattr(self.processor, "tokenizer", None)

        # Generation config
        self.max_new_tokens = _env_int("MAX_NEW_TOKENS", 128)
        self.temperature = _env_float("TEMPERATURE", 0.0)
        self.top_p = _env_float("TOP_P", 1.0)

        # Video sampling config
        self.num_frames = _env_int("NUM_FRAMES", 16)
        self.min_window = _env_float("MIN_WINDOW_SEC", 0.5)

        # Memory Proxy config. Only used when MEMORY_MODE=1.
        self.memory_mode = _env("MEMORY_MODE", "0").lower() in {"1", "true", "yes", "y", "on"}
        self.memory_dense_num_frames = max(1, _env_int("MEMORY_DENSE_NUM_FRAMES", 15))
        self.memory_caption_cache_root = _env("MEMORY_CAPTION_CACHE_ROOT", "memory_cache/captions")
        self.memory_fail_on_missing_caption = _env("MEMORY_FAIL_ON_MISSING_CAPTION", "1") != "0"
        self.memory_require_keyframes = _env("MEMORY_REQUIRE_KEYFRAMES", "0") == "1"

        # Window fallback
        lb_override = os.environ.get("LOOKBACK_OVERRIDE_SEC", "").strip()
        self.lookback_override_sec: Optional[float] = None
        if lb_override:
            try:
                v = float(lb_override)
                if v > 0:
                    self.lookback_override_sec = v
            except Exception:
                self.lookback_override_sec = None
        self.lookback_default_sec = _env_float("LOOKBACK_DEFAULT_SEC", 20.0)

        # Prompt length control
        env_mpt = os.environ.get("MAX_PROMPT_TOKENS", "").strip()
        if env_mpt:
            self.max_prompt_tokens = _safe_int(env_mpt, 0)
        else:
            self.max_prompt_tokens = 2048

        self.prompt_head_tokens = _env_int("PROMPT_HEAD_TOKENS", 96)
        self.prompt_tail_tokens = _env_int("PROMPT_TAIL_TOKENS", 512)

        self.resume = _env("RESUME", "1") != "0"

        # Diagnostics control
        self.diag_prompt_max_chars = _env_int("DIAG_PROMPT_MAX_CHARS", 4000)
        self.diag_save_full_prompt = _env("DIAG_SAVE_FULL_PROMPT", "0") != "0"

        # PRED fix controls
        self.pred_min_new_tokens = _env_int("PRED_MIN_NEW_TOKENS", 32)
        self.pred_retry = _env("PRED_RETRY", "1") != "0"
        self.pred_retry_suffix = _env(
            "PRED_RETRY_SUFFIX",
            "Finish the required one-line output and close the </FUTURE> tag. Output only the tags line."
        )

        # NOW/PAST fix controls
        self.now_min_new_tokens = _env_int("NOW_MIN_NEW_TOKENS", 48)
        self.past_min_new_tokens = _env_int("PAST_MIN_NEW_TOKENS", 48)
        self.now_retry = _env("NOW_RETRY", "1") != "0"
        self.past_retry = _env("PAST_RETRY", "1") != "0"
        self.now_retry_suffix = _env(
            "NOW_RETRY_SUFFIX",
            "Output EXACTLY one line in the required <NOW>...</NOW> format WITH <STATE><VERB><NOUN><DESC><CONF>. "
            "STATE must be exactly INTERACTION or NO_INTERACTION (uppercase). "
            "Do NOT output any bracketed numbers like [0.1, 0.2]. Output only the tags line."
        )

        # NOW state-only / action-mcq retry suffixes
        self.now_state_only_retry_suffix = _env(
            "NOW_STATE_ONLY_RETRY_SUFFIX",
            "Output EXACTLY one line in the required <NOW>...</NOW> format WITH ONLY <STATE><CONF>. "
            "STATE must be exactly INTERACTION or NO_INTERACTION (uppercase). "
            "Do NOT output VERB/NOUN/DESC or any extra explanation. "
            "Do NOT output anything before <NOW>. Output only the tags line.\n"
            "<NOW><STATE>INTERACTION</STATE><CONF>0.00</CONF></NOW>"
        )
        self.now_action_mcq_retry_suffix = _env(
            "NOW_ACTION_MCQ_RETRY_SUFFIX",
            "Output EXACTLY one line in the required <NOW>...</NOW> candidate format WITH <ANS><VERB><NOUN><CONF>. "
            "IMPORTANT: <ANS> must be EXACTLY ONE uppercase letter from A/B/C/D. "
            "Set <VERB> and <NOUN> by COPYING EXACTLY from PARSED_OPTIONS for the chosen <ANS>. "
            "Do NOT output <STATE> or <DESC>. "
            "Do NOT output anything before <NOW>. "
            "Do NOT output placeholders like choice_here, verb_here, noun_here_or_none, short_concise_sentence. "
            "Output only the tags line."
        )

        self.past_retry_suffix = _env(
            "PAST_RETRY_SUFFIX",
            "Output EXACTLY one line in the required <PAST>...</PAST> format WITH <VERB><NOUN><DESC><CONF> "
            "(and <ANS> if required). Do NOT output any bracketed numbers like [0.1, 0.2]. Output only the tags line."
        )

        # --- ms_rtrv open anti-echo decode + strong template retry ---
        self.ms_rtrv_open_rep_penalty = _env_float("MS_RTRV_OPEN_REP_PENALTY", 1.12)
        self.ms_rtrv_open_no_repeat_ngram = _env_int("MS_RTRV_OPEN_NO_REPEAT_NGRAM", 3)
        self.ms_rtrv_open_retry_suffix = _env(
            "MS_RTRV_OPEN_RETRY_SUFFIX",
            "Do NOT repeat or quote any instruction sentence from the question. "
            "Do NOT output any bracketed numbers like [0.1, 0.2]. "
            "Output EXACTLY ONE line by filling the required tags:\n"
            "<PAST><VERB>none</VERB><NOUN>none</NOUN><DESC>YOU do nothing</DESC><CONF>0.00</CONF></PAST>"
        )
        self.ms_rtrv_open_max_retries = _env_int("MS_RTRV_OPEN_MAX_RETRIES", 3)
        self.ms_rtrv_open_bad_words = _env(
            "MS_RTRV_OPEN_BAD_WORDS",
            "[,], [0,[1,[2,[3,[4,[5,[6,[7,[8,[9,[0.,[1.,[2.,[3.,[4.,[5.,[6.,[7.,[8.,[9."
        )

        # --- candidate-mode fixes ---
        self.cand_retry = _env("CAND_RETRY", "1") != "0"
        self.cand_max_retries = _env_int("CAND_MAX_RETRIES", 2)
        self.cand_bad_words = _env(
            "CAND_BAD_WORDS",
            "choice_here,verb_here,noun_here_or_none,short_concise_sentence"
        )
        self.cand_retry_suffix = _env(
            "CAND_RETRY_SUFFIX",
            "Output EXACTLY one line in the required tag format. "
            "IMPORTANT: <ANS> must be EXACTLY ONE uppercase letter from A/B/C/D (not 'choice_here'). "
            "You MUST choose exactly one of A/B/C/D in <ANS>. "
            "Set <VERB> and <NOUN> by COPYING EXACTLY from PARSED_OPTIONS for the chosen <ANS>. "
            "Set <DESC> to EXACTLY: 'YOU <VERB> <NOUN>' (or 'YOU do nothing' if VERB/NOUN are none). "
            "Do NOT output placeholders like choice_here, verb_here, noun_here_or_none, short_concise_sentence. "
            "Output only the tags line."
        )

        # --- ms_pred cand specific ---
        self.ms_pred_cand_retry_suffix = _env(
            "MS_PRED_CAND_RETRY_SUFFIX",
            "DESC must be plain text only. Do NOT include any '<' or '>' or any tags inside <DESC>. "
            + self.cand_retry_suffix
        )

        # --- ms cand post-processing (only ms_*_cand) ---
        self.ms_cand_postproc = _env("MS_CAND_POSTPROC", "1") != "0"

        # Model max length & input budget
        self.model_max_len = _infer_model_max_len(self.tokenizer, self.model)
        safety = _env_int("CTX_SAFETY_TOKENS", 16)
        self.max_input_tokens = int(max(256, self.model_max_len - self.max_new_tokens - safety))

        # Schema-specific generation budgets
        self.now_state_only_max_new_tokens = _env_int("NOW_STATE_ONLY_MAX_NEW_TOKENS", 32)
        self.now_action_mcq_max_new_tokens = _env_int("NOW_ACTION_MCQ_MAX_NEW_TOKENS", 48)

        # stop-on-close-tags
        self.stop_on_close_tags = _env("STOP_ON_CLOSE_TAGS", "1") != "0"
        self._close_tag_strs_upper = ["</NOW>", "</PAST>", "</FUTURE>"]

        # -------------------------
        # NEW: cand_conf mode controls
        # -------------------------
        # Active flavor is decided in run() from qs.params.pred_flavor and/or env PRED_FLAVOR.
        self._active_pred_flavor: str = ""
        self.cand_conf_max_new_tokens = _env_int("CAND_CONF_MAX_NEW_TOKENS", 4)
        self.cand_conf_force_fallback = _env("CAND_CONF_FORCE_FALLBACK", "1") != "0"

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

    def _try_self_heal_clip(self, video_path: Path, qs: Dict[str, Any], err: Exception) -> bool:
        msg = str(err).lower()
        keywords = ["moov atom not found", "invalid data found", "error opening", "cannot open", "failed to open"]
        if not any(k in msg for k in keywords):
            return False

        video_root_env = os.environ.get("VIDEO_ROOT", "").strip()
        if not video_root_env:
            return False
        video_root = Path(video_root_env).expanduser()

        vm = qs.get("video_metadata", {}) if isinstance(qs.get("video_metadata"), dict) else {}
        video_uid = str(vm.get("video_uid", qs.get("video_uid", ""))).strip()
        if not video_uid:
            return False

        interval_start = vm.get("interval_start_sec", None)
        interval_end = vm.get("interval_end_sec", None)
        if interval_start is None or interval_end is None:
            return False

        try:
            src = _resolve_source_video(video_root, video_uid)
            _recut_interval_clip_from_source(
                src_video=src,
                dst_clip=video_path,
                start_sec=float(interval_start),
                end_sec=float(interval_end),
            )
            return True
        except Exception:
            try:
                if video_path.exists():
                    video_path.unlink()
            except Exception:
                pass
            return False

    def _probe_clip_duration(self, video_path: Path, qs: Dict[str, Any]) -> float:
        try:
            _frames0, clip_dur = _load_frames_decord(video_path, 0.0, 0.01, 1)
            return float(clip_dur)
        except Exception as e:
            healed = self._try_self_heal_clip(video_path, qs, e)
            if healed:
                _frames0, clip_dur = _load_frames_decord(video_path, 0.0, 0.01, 1)
                return float(clip_dur)
            raise RuntimeError(f"Error reading {video_path}... Original error: {e}") from e

    def _caption_path_for_memory(self, memory_meta: Dict[str, Any]) -> str:
        sparse = memory_meta.get("sparse_window", {}) if isinstance(memory_meta.get("sparse_window"), dict) else {}
        path = sparse.get("caption_cache_path") or memory_meta.get("caption_cache_path")
        if path:
            return str(path)
        return caption_record_path(self.memory_caption_cache_root, memory_meta)

    def _build_memory_prompt(self, original_prompt: str, memory_meta: Dict[str, Any], caption_text: str, keyframe_timestamps: List[float]) -> str:
        dense = memory_meta.get("dense_window", {}) if isinstance(memory_meta.get("dense_window"), dict) else {}
        sparse = memory_meta.get("sparse_window", {}) if isinstance(memory_meta.get("sparse_window"), dict) else {}
        ctx = memory_meta.get("cache_context", {}) if isinstance(memory_meta.get("cache_context"), dict) else {}
        t_eval_rel = _safe_float(ctx.get("t_eval_rel"), _safe_float(dense.get("end_sec"), 0.0))
        sparse_start = _safe_float(sparse.get("start_sec"), 0.0)
        sparse_end = _safe_float(sparse.get("end_sec"), 0.0)
        rel_start = sparse_start - t_eval_rel
        rel_end = sparse_end - t_eval_rel
        rel_times = [f"{(float(ts) - t_eval_rel):.1f}s" for ts in keyframe_timestamps]
        caption = (caption_text or "").strip()
        if not caption:
            caption = "(caption unavailable)"
        prefix = (
            "[Sparse memory from earlier observed video]\n"
            "The following memory summary and key frames are generated only from video observed before the current query time. "
            "They may be incomplete. Use them as background context, but answer using the required task format.\n\n"
            f"Memory time range: from t{rel_start:.1f}s to t{rel_end:.1f}s.\n"
            f"Memory caption: {caption}\n\n"
            f"Sparse keyframe times before query: {', '.join(rel_times) if rel_times else '(none)'}\n\n"
            "[Recent dense visual buffer]\n"
            "The dense visual frames cover only the most recent 10 seconds before the query time.\n\n"
            "[Current query]\n"
        )
        return (prefix + (original_prompt or "").strip()).strip()

    def _prepare_memory_inputs(
        self,
        *,
        video_path: Path,
        sample: Dict[str, Any],
        original_prompt: str,
        qs: Dict[str, Any],
        clip_dur: float,
    ) -> Tuple[str, np.ndarray, List[str], Dict[str, Any]]:
        del qs
        memory_meta = sample.get("memory", {})
        if not isinstance(memory_meta, dict) or not memory_meta.get("enabled"):
            raise RuntimeError("MEMORY_MODE=1 but sample['memory'] is missing or disabled")

        dense = memory_meta.get("dense_window", {}) if isinstance(memory_meta.get("dense_window"), dict) else {}
        sparse = memory_meta.get("sparse_window", {}) if isinstance(memory_meta.get("sparse_window"), dict) else {}
        dense_start = max(0.0, _safe_float(dense.get("start_sec"), 0.0))
        dense_end = min(_safe_float(dense.get("end_sec"), dense_start), float(clip_dur))
        dense_num = max(1, _safe_int(dense.get("num_frames", self.memory_dense_num_frames), self.memory_dense_num_frames))
        t_eval_rel = _safe_float(sample.get("t_eval_rel"), dense_end)

        caption_path = self._caption_path_for_memory(memory_meta)
        caption_record = load_caption_record(caption_path)
        caption_cache_hit = bool(caption_record is not None and caption_record.get("api_status", "ok") == "ok")
        caption_missing = not caption_cache_hit
        if caption_missing and self.memory_fail_on_missing_caption:
            raise RuntimeError(f"missing caption cache: {caption_path}")

        caption_text = ""
        keyframe_paths: List[str] = []
        keyframe_timestamps = [float(x) for x in sparse.get("expected_keyframe_timestamps_sec", [])]
        if caption_record is not None:
            caption_text = str(caption_record.get("caption_text", "") or "").strip()
            record_paths = caption_record.get("keyframe_paths", [])
            if isinstance(record_paths, list):
                keyframe_paths = [str(p) for p in record_paths if str(p).strip()]
            record_ts = caption_record.get("keyframe_timestamps_sec", [])
            if isinstance(record_ts, list) and record_ts:
                keyframe_timestamps = [float(x) for x in record_ts]

        existing_keyframe_paths = [p for p in keyframe_paths if Path(p).is_file()]
        keyframes_missing = bool(keyframe_paths and len(existing_keyframe_paths) < len(keyframe_paths)) or (not existing_keyframe_paths)
        if keyframes_missing and self.memory_require_keyframes:
            raise RuntimeError(f"missing sparse keyframe files for caption cache: {caption_path}")

        sparse_frames = _load_keyframe_images(existing_keyframe_paths) if existing_keyframe_paths else np.zeros((0, 1, 1, 3), dtype=np.uint8)
        dense_frames, dense_timestamps, _ = _load_frames_decord_with_timestamps(video_path, dense_start, dense_end, dense_num)

        if sparse_frames.shape[0] > 0:
            if sparse_frames.shape[1:3] != dense_frames.shape[1:3]:
                resized: List[np.ndarray] = []
                target_hw = (int(dense_frames.shape[2]), int(dense_frames.shape[1]))
                for frame in sparse_frames:
                    resized.append(np.asarray(Image.fromarray(frame.astype(np.uint8)).resize(target_hw).convert("RGB"), dtype=np.uint8))
                sparse_frames = np.stack(resized, axis=0)
            frames = np.concatenate([sparse_frames, dense_frames], axis=0)
        else:
            frames = dense_frames

        roles = ["sparse_keyframe"] * int(sparse_frames.shape[0]) + ["dense_frame"] * int(dense_frames.shape[0])
        prompt = self._build_memory_prompt(original_prompt, memory_meta, caption_text, keyframe_timestamps)
        sparse_end = _safe_float(sparse.get("end_sec"), 0.0)
        no_lookahead = bool(
            dense_end <= t_eval_rel + 1e-6
            and sparse_end <= dense_start + 1e-6
            and all(float(ts) <= sparse_end + 1e-6 for ts in keyframe_timestamps)
            and all(float(ts) <= dense_end + 1e-6 for ts in dense_timestamps)
        )

        diag = {
            "enabled": True,
            "setting_name": memory_meta.get("setting_name"),
            "schema_version": memory_meta.get("schema_version"),
            "dense_window": dict(dense),
            "sparse_window": dict(sparse),
            "caption_cache_key": memory_meta.get("caption_cache_key") or sparse.get("caption_cache_key"),
            "caption_cache_path": caption_path,
            "caption_cache_hit": caption_cache_hit,
            "caption_missing": caption_missing,
            "caption_used": bool(caption_text),
            "keyframes_used": bool(existing_keyframe_paths),
            "keyframe_paths": existing_keyframe_paths,
            "keyframe_timestamps_sec": keyframe_timestamps,
            "selected_dense_timestamps_sec": dense_timestamps,
            "visual_input_roles": roles,
            "visual_input_strategy": "merged_single_video_frame_list",
            "sparse_keyframes_initial": int(len(existing_keyframe_paths)),
            "dense_frames_initial": int(dense_frames.shape[0]),
            "caption_api_status": caption_record.get("api_status") if isinstance(caption_record, dict) else None,
            "no_lookahead_checked": True,
            "no_lookahead": no_lookahead,
            "coverage_note": memory_meta.get("coverage_note"),
        }
        return prompt, frames, roles, diag

    def _infer_lookback(self, qs: Dict[str, Any], sample: Dict[str, Any]) -> float:
        if self.lookback_override_sec is not None:
            return float(self.lookback_override_sec)

        for k in ("lookback_sec", "lookback", "context_sec", "context"):
            if k in sample:
                try:
                    v = float(sample[k])
                    if v > 0:
                        return v
                except Exception:
                    pass

        params = qs.get("params")
        if isinstance(params, dict):
            for k in ("lookback_sec", "lookback", "context_sec", "context"):
                if k in params:
                    try:
                        v = float(params.get(k))
                        if v > 0:
                            return v
                    except Exception:
                        pass

        return float(self.lookback_default_sec)

    def _infer_window_rel(self, qs: Dict[str, Any], sample: Dict[str, Any], *, time_offset_sec: float, clip_dur: float) -> Tuple[float, float, float]:
        """
        老版口径（OpenRouter adapter）对齐：
          - t_eval 使用“加载视频时间”（即 t_eval_rel 或 (t_eval - offset)）
          - window_end ALWAYS clamp 到 t_eval（strict-online），不 clamp 到 clip_dur
          - window_start clamp 到 [0, window_end]
          - min_window 仅向过去扩展（保持 end 不变）
        """
        t_rel = sample.get("t_eval_rel", None)
        if t_rel is None:
            t_rel = _safe_float(sample.get("t_eval", 0.0), 0.0) - float(time_offset_sec)
        t_rel = float(max(0.0, float(t_rel)))

        ws_raw = sample.get("window_start_sec", None)
        we_raw = sample.get("window_end_sec", None)
        has_ws = (ws_raw is not None)
        has_we = (we_raw is not None)

        lookback = self._infer_lookback(qs, sample)

        if has_ws or has_we:
            ws_rel = float(_safe_float(ws_raw, 0.0) - float(time_offset_sec)) if has_ws else (t_rel - lookback)
            we_rel = float(_safe_float(we_raw, t_rel) - float(time_offset_sec)) if has_we else t_rel

            we_rel = min(we_rel, t_rel)
            ws_rel = max(0.0, min(ws_rel, we_rel))

            ws_rel, we_rel = _ensure_min_window(ws_rel, we_rel, self.min_window)
            lookback_used = float(max(0.0, we_rel - ws_rel))
            return float(ws_rel), float(we_rel), float(lookback_used)

        ws_rel = max(0.0, t_rel - lookback)
        we_rel = t_rel

        ws_rel, we_rel = _ensure_min_window(ws_rel, we_rel, self.min_window)
        lookback_used = float(max(0.0, we_rel - ws_rel))
        return float(ws_rel), float(we_rel), float(lookback_used)

    @torch.inference_mode()
    def _generate_one(
        self, prompt_text: str, frames: np.ndarray, memory_visual_roles: Optional[List[str]] = None
    ) -> Tuple[str, List[str], List[float], Optional[float], Optional[float], int, Dict[str, Any]]:
        diag: Dict[str, Any] = {}

        # -------------------------
        # Subkind classification - narrow targeting
        # -------------------------
        def _classify_prompt_subkind(txt: str) -> Tuple[str, str, bool]:
            s = (txt or "").lower()
            cand = ("| candidate" in s) or ("<ans>" in s) or ("options:" in s and re.search(r"^[a-d]\.\s+", s, flags=re.MULTILINE) is not None)

            if "now state switch" in s or "now_state_switch" in s or "now-state-switch" in s or "now state" in s:
                return "now", "now_state_switch", cand
            if "now narration" in s or "now_narration" in s or "now-narration" in s or "<now>" in s or "</now>" in s:
                return "now", "now_narration", cand

            if "multistep past retrieval" in s or "ms_rtrv" in s:
                return "past", ("ms_rtrv_cand" if cand else "ms_rtrv_open"), cand
            if "past retrieval" in s or "sh_rtrv" in s or "<past>" in s or "</past>" in s:
                return "past", ("sh_rtrv_cand" if cand else "sh_rtrv_open"), cand

            if "multistep prediction" in s or "ms_pred" in s:
                return "pred", ("ms_pred_cand" if cand else "ms_pred_open"), cand
            if "short-horizon prediction" in s or "sh_pred" in s:
                return "pred", ("sh_pred_cand" if cand else "sh_pred_open"), cand
            if "<future>" in s or "</future>" in s:
                return "pred", ("ms_pred_cand" if cand else "ms_pred_open"), cand

            return "other", "other", cand

        # -------------------------
        # infer NOW schema from prompt
        # -------------------------
        def _infer_now_schema_from_prompt(txt: str) -> str:
            """
            Return one of:
              - now_full        : <STATE><VERB><NOUN><DESC><CONF>
              - now_state_only  : <STATE><CONF>
              - now_action_mcq  : <ANS><VERB><NOUN><CONF>
            """
            s = (txt or "")
            up = s.upper()

            if "<NOW>" not in up and "</NOW>" not in up:
                return "now_full"

            has_ans = ("<ANS>" in up and "</ANS>" in up) or bool(re.search(r"<\s*ANS\s*>", up))
            has_state = bool(re.search(r"<\s*STATE\s*>", up))
            has_verb = bool(re.search(r"<\s*VERB\s*>", up))
            has_noun = bool(re.search(r"<\s*NOUN\s*>", up))
            has_desc = bool(re.search(r"<\s*DESC\s*>", up))
            has_conf = bool(re.search(r"<\s*CONF\s*>", up))

            if has_ans and (has_verb or has_noun) and has_conf and (not has_state):
                return "now_action_mcq"

            if has_state and has_conf and (not has_verb) and (not has_noun) and (not has_desc):
                return "now_state_only"

            return "now_full"

        # -------------------------
        # Candidate ANS de-anchor
        # -------------------------
        def _deanchor_ans_template_for_cand(p: str) -> Tuple[str, bool]:
            if not p:
                return p, False
            p2 = re.sub(r"<\s*ANS\s*>\s*A\s*<\s*/\s*ANS\s*>", "<ANS>choice_here</ANS>", p, flags=re.IGNORECASE)
            return p2, (p2 != p)

        # -------------------------
        # Candidate options parsing + parsed options block
        # -------------------------
        def _extract_options_map(text: str) -> Dict[str, Dict[str, str]]:
            opts: Dict[str, Dict[str, str]] = {}
            if not text:
                return opts
            lines = text.splitlines()
            for ln in lines:
                m = re.match(r"^\s*([A-D])\.\s+(.*)$", ln.strip(), flags=re.IGNORECASE)
                if not m:
                    continue
                key = m.group(1).upper()
                body = m.group(2).strip()
                body2 = body
                if body2.lower().startswith("you "):
                    body2 = body2[4:].strip()
                toks = body2.split()
                if not toks:
                    verb = "none"
                    noun = "none"
                else:
                    verb = toks[0].strip()
                    rest = " ".join(toks[1:]).strip()
                    if verb.lower() == "none":
                        verb = "none"
                        noun = "none"
                    else:
                        noun = rest if rest else "none"
                opts[key] = {"raw": body, "verb": verb, "noun": noun}
            return opts

        def _append_parsed_options_block(base_prompt: str, opts: Dict[str, Dict[str, str]], *, require_desc: bool = True) -> str:
            if not base_prompt or not opts:
                return base_prompt
            if "PARSED_OPTIONS" in base_prompt:
                return base_prompt
            lines = []
            lines.append("")
            lines.append("[PARSED_OPTIONS] (for copying VERB/NOUN exactly in candidate mode)")
            for k in ["A", "B", "C", "D"]:
                if k in opts:
                    lines.append(f"{k}: VERB={opts[k]['verb']} | NOUN={opts[k]['noun']}")
            lines.append("Rule: For candidate mode output, set <VERB> and <NOUN> by COPYING EXACTLY from PARSED_OPTIONS for your chosen <ANS>.")
            lines.append("Do NOT output placeholders like choice_here, verb_here, noun_here_or_none, short_concise_sentence.")
            if require_desc:
                lines.append("Also set <DESC> to EXACTLY: 'YOU <VERB> <NOUN>' (or 'YOU do nothing' if VERB/NOUN are none).")
            else:
                lines.append("Do NOT output <DESC> for this task; output only the required tags.")
            return (base_prompt.rstrip() + "\n" + "\n".join(lines)).strip()

        # -------------------------
        # Validators
        # -------------------------
        def _extract_tag_value(text: str, tag: str) -> str:
            if not text:
                return ""
            m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
            return m.group(1).strip() if m else ""

        def _has_all_tags(text: str, tags: List[str]) -> bool:
            up = (text or "").upper()
            return all(f"<{t}>" in up and f"</{t}>" in up for t in tags)

        def _schema_incomplete(kind_local: str, cand_local: bool, text: str, *, now_schema: str = "now_full") -> Tuple[bool, str]:
            s = (text or "").strip()
            if not s:
                return True, "empty"
            up = s.upper()

            if re.search(r"<(NOW|PAST|FUTURE)>\s*\[[^\]]+\]\s*</\1>", s, flags=re.IGNORECASE):
                return True, "bracket_numbers_inside_root"

            if kind_local == "pred":
                if "</FUTURE>" not in up:
                    return True, "missing_close_future"
                need = ["VERB", "NOUN", "DESC", "CONF"]
                if cand_local:
                    need = ["ANS"] + need
                if not _has_all_tags(s, need):
                    return True, "missing_subtags_future"
                return False, ""

            if kind_local == "now":
                if "</NOW>" not in up:
                    return True, "missing_close_now"

                if now_schema == "now_state_only":
                    need = ["STATE", "CONF"]
                elif now_schema == "now_action_mcq":
                    need = ["ANS", "VERB", "NOUN", "CONF"]
                else:
                    need = ["STATE", "VERB", "NOUN", "DESC", "CONF"]

                if not _has_all_tags(s, need):
                    return True, "missing_subtags_now"
                return False, ""

            if kind_local == "past":
                if "</PAST>" not in up:
                    return True, "missing_close_past"
                need = ["VERB", "NOUN", "DESC", "CONF"]
                if cand_local:
                    need = ["ANS"] + need
                if not _has_all_tags(s, need):
                    return True, "missing_subtags_past"
                return False, ""

            return False, ""

        def _contains_placeholders(text: str) -> bool:
            if not text:
                return True
            low = text.lower()
            bads = [
                "verb_here", "noun_here_or_none", "short_concise_sentence",
                "state_here", "desc_here", "conf_here", "ans_here",
                "choice_here",
            ]
            return any(b in low for b in bads)

        def _norm_ws(s: str) -> str:
            return re.sub(r"\s+", " ", (s or "").strip())

        def _cand_expected_desc(letter: str, opts: Dict[str, Dict[str, str]]) -> str:
            o = opts.get(letter, {})
            v = (o.get("verb") or "none").strip()
            n = (o.get("noun") or "none").strip()
            if v.lower() == "none" or n.lower() == "none":
                return "YOU do nothing"
            return f"YOU {v} {n}"

        def _cand_desc_mismatch(text: str, opts: Dict[str, Dict[str, str]]) -> Tuple[bool, str]:
            if not text or not opts:
                return False, ""
            ans = _extract_tag_value(text, "ANS").strip().upper()
            if ans == "CHOICE_HERE":
                return True, "ans_is_choice_here_placeholder"
            letter = ans[:1] if ans else ""
            if letter not in {"A", "B", "C", "D"}:
                return True, "bad_ans_letter"
            exp = _cand_expected_desc(letter, opts)
            got = _extract_tag_value(text, "DESC")
            if not got:
                return True, "missing_desc"

            exp_n = _norm_ws(exp).rstrip(".").lower()
            got_n = _norm_ws(got).rstrip(".").lower()
            if got_n != exp_n:
                return True, "desc_not_matching_option"
            return False, ""

        def _cand_ans_vn_mismatch(text: str, opts: Dict[str, Dict[str, str]]) -> Tuple[bool, str]:
            if not text or not opts:
                return False, ""
            ans = _extract_tag_value(text, "ANS")
            if not ans:
                return True, "missing_ans"
            ans_clean = ans.strip().upper()
            if ans_clean == "CHOICE_HERE":
                return True, "ans_is_choice_here_placeholder"
            letter = ans_clean[:1]
            if letter not in {"A", "B", "C", "D"}:
                return True, "bad_ans_letter"
            if letter not in opts:
                return True, "ans_not_in_options"
            ev = opts[letter]["verb"]
            en = opts[letter]["noun"]
            ov = _extract_tag_value(text, "VERB")
            on = _extract_tag_value(text, "NOUN")
            if ov.strip() != ev.strip() or on.strip() != en.strip():
                return True, "verb_noun_not_copied_from_parsed_options"
            return False, ""

        def _ms_pred_cand_desc_has_tags(text: str) -> bool:
            desc = _extract_tag_value(text, "DESC")
            if not desc:
                return False
            return ("<" in desc) or (">" in desc)

        def _now_state_enum_invalid(text: str) -> Tuple[bool, str]:
            st = _extract_tag_value(text, "STATE")
            if not st:
                return True, "missing_state_value"
            up = st.strip().upper()
            if up not in {"INTERACTION", "NO_INTERACTION"}:
                return True, "state_not_in_enum"
            return False, ""

        def _ms_rtrv_open_has_brackets_anywhere(text: str) -> bool:
            if not text:
                return False
            return ("[" in text) or ("]" in text)

        def _output_bad(
            kind_local: str,
            subkind_local: str,
            cand_local: bool,
            text: str,
            opts: Dict[str, Dict[str, str]],
            *,
            now_schema: str = "now_full",
        ) -> Tuple[bool, str]:
            bad, reason = _schema_incomplete(kind_local, cand_local, text, now_schema=now_schema)
            if bad:
                return True, reason

            if kind_local == "now":
                if now_schema in {"now_full", "now_state_only"}:
                    bad2, r2 = _now_state_enum_invalid(text)
                    if bad2:
                        return True, r2

            if subkind_local == "ms_rtrv_open":
                if _ms_rtrv_open_has_brackets_anywhere(text):
                    return True, "ms_rtrv_open_contains_brackets"

            if cand_local:
                if _contains_placeholders(text):
                    return True, "contains_placeholders"
                b3, r3 = _cand_ans_vn_mismatch(text, opts)
                if b3:
                    return True, r3

                if not (kind_local == "now" and now_schema == "now_action_mcq"):
                    b4, r4 = _cand_desc_mismatch(text, opts)
                    if b4:
                        return True, r4

            if subkind_local == "ms_pred_cand":
                if _ms_pred_cand_desc_has_tags(text):
                    return True, "desc_contains_tags"

            return False, ""

        def _build_bad_words_ids_from_csv(csv: str) -> Optional[List[List[int]]]:
            if self.tokenizer is None:
                return None
            if not csv:
                return None
            parts = [p.strip() for p in csv.split(",") if p.strip()]
            if not parts:
                return None
            bad: List[List[int]] = []
            for s in parts:
                try:
                    ids = self.tokenizer.encode(s, add_special_tokens=False)
                except Exception:
                    ids = []
                if ids:
                    bad.append([int(x) for x in ids])
            return bad if bad else None

        # -------------------------
        # ms_*_cand post-processing (past/pred only)
        # -------------------------
        def _parse_conf_any(text: str) -> float:
            if not text:
                return 0.0
            m = re.search(r"<\s*CONF\s*>\s*([0-9]*\.?[0-9]+)\s*<\s*/\s*CONF\s*>", text, flags=re.IGNORECASE)
            if not m:
                m = re.search(r"\bCONF\s*[:=]\s*([0-9]*\.?[0-9]+)\b", text, flags=re.IGNORECASE)
            if not m:
                m = re.search(r"\[([0-9]*\.?[0-9]+)\]\s*$", text.strip())
            if not m:
                return 0.0
            try:
                v = float(m.group(1))
            except Exception:
                v = 0.0
            if v < 0.0:
                v = 0.0
            if v > 1.0:
                v = 1.0
            return float(v)

        def _parse_vn_any(text: str) -> Tuple[str, str]:
            if not text:
                return "", ""
            v = _extract_tag_value(text, "VERB")
            n = _extract_tag_value(text, "NOUN")
            if v or n:
                return v.strip(), n.strip()

            m = re.search(r"\bVERB\s*[:=]\s*([A-Za-z]+)\b", text, flags=re.IGNORECASE)
            v2 = m.group(1).strip() if m else ""
            m = re.search(r"\bNOUN\s*[:=]\s*([^\n\r<]+)", text, flags=re.IGNORECASE)
            n2 = m.group(1).strip() if m else ""
            if v2 or n2:
                return v2, n2

            m = re.search(r"\bYOU\s+([A-Za-z]+)\s+([^\n\r<\[]+)", text, flags=re.IGNORECASE)
            if m:
                v3 = m.group(1).strip()
                n3 = m.group(2).strip()
                n3 = re.sub(r"\s*(?:\[.*?\])\s*$", "", n3).strip()
                n3 = n3.rstrip(".").strip()
                return v3, n3
            return "", ""

        def _parse_ans_any(text: str) -> str:
            if not text:
                return ""
            ans = _extract_tag_value(text, "ANS")
            if ans:
                ans_u = ans.strip().upper()
                if ans_u != "CHOICE_HERE":
                    return ans_u[:1]
            m = re.search(r"\bANS\s*[:=]\s*([A-D])\b", text, flags=re.IGNORECASE)
            if m:
                return m.group(1).upper()
            m = re.search(r"^\s*([A-D])\.", text.strip(), flags=re.IGNORECASE)
            if m:
                return m.group(1).upper()
            m = re.search(r"\b([A-D])\.\s+YOU\b", text, flags=re.IGNORECASE)
            if m:
                return m.group(1).upper()
            return ""

        def _match_letter_by_vn(v: str, n: str, opts: Dict[str, Dict[str, str]]) -> str:
            if not v or not n:
                return ""
            for k in ["A", "B", "C", "D"]:
                if k in opts and opts[k].get("verb", "").strip() == v.strip() and opts[k].get("noun", "").strip() == n.strip():
                    return k
            return ""

        def _fallback_letter(text: str, opts: Dict[str, Dict[str, str]]) -> str:
            low = (text or "").lower()
            if ("do nothing" in low) or (re.search(r"\bnone\b", low) is not None):
                for k in ["A", "B", "C", "D"]:
                    if k in opts and opts[k].get("verb", "").strip().lower() == "none" and opts[k].get("noun", "").strip().lower() == "none":
                        return k
            for k in ["A", "B", "C", "D"]:
                if k in opts:
                    return k
            return "A"

        def _coerce_ms_cand_to_tags(kind_local: str, subkind_local: str, text: str, opts: Dict[str, Dict[str, str]]) -> Tuple[str, bool, str]:
            if kind_local not in {"past", "pred"}:
                return text, False, "not_past_or_pred"
            if subkind_local not in {"ms_rtrv_cand", "ms_pred_cand"}:
                return text, False, "not_ms_cand"
            if not opts:
                return text, False, "no_options_parsed"

            bad, _reason = _output_bad(kind_local, subkind_local, True, text, opts, now_schema="now_full")
            if not bad:
                return text, False, "already_valid"

            root = "PAST" if kind_local == "past" else "FUTURE"

            ans = _parse_ans_any(text)
            v, n = _parse_vn_any(text)
            if ans and ans in opts:
                letter = ans
            else:
                letter = _match_letter_by_vn(v, n, opts)
                if not letter:
                    letter = _fallback_letter(text, opts)

            verb = opts.get(letter, {}).get("verb", "none").strip() or "none"
            noun = opts.get(letter, {}).get("noun", "none").strip() or "none"
            if verb.lower() == "none" or noun.lower() == "none":
                desc = "YOU do nothing"
                verb = "none"
                noun = "none"
            else:
                desc = f"YOU {verb} {noun}"

            conf = _parse_conf_any(text)
            conf_s = f"{conf:.2f}"

            out = f"<{root}><ANS>{letter}</ANS><VERB>{verb}</VERB><NOUN>{noun}</NOUN><DESC>{desc}</DESC><CONF>{conf_s}</CONF></{root}>"
            return out, True, "coerced_from_malformed_ms_cand"

        # -------------------------
        # Stop generation after close tags
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

        # ---------- prompt diagnostics ----------
        prompt_raw = (prompt_text or "").strip()
        diag["prompt_raw_preview"] = _truncate_text(prompt_raw, self.diag_prompt_max_chars)
        if self.diag_save_full_prompt:
            diag["prompt_raw_full"] = prompt_raw

        raw_tok_len = None
        if self.tokenizer is not None:
            try:
                raw_tok_len = int(len(self.tokenizer.encode(prompt_raw, add_special_tokens=False)))
            except Exception:
                raw_tok_len = None
        diag["prompt_raw_tok_len"] = raw_tok_len

        prompt_after = prompt_raw
        compacted = False
        if self.tokenizer is not None and self.max_prompt_tokens > 0:
            prompt_after = _compact_prompt_task_agnostic(
                self.tokenizer,
                prompt_raw,
                max_tokens=int(self.max_prompt_tokens),
                head_tokens=int(self.prompt_head_tokens),
                tail_tokens=int(self.prompt_tail_tokens),
            )
            compacted = (prompt_after != prompt_raw)

        diag["prompt_compacted"] = bool(compacted)

        kind, subkind, cand = _classify_prompt_subkind(prompt_after)

        if cand:
            prompt_after2, changed = _deanchor_ans_template_for_cand(prompt_after)
            diag["cand_deanchor_applied"] = bool(changed)
            if changed:
                prompt_after = prompt_after2
        else:
            diag["cand_deanchor_applied"] = False

        now_schema = "now_full"
        if kind == "now":
            now_schema = _infer_now_schema_from_prompt(prompt_after)
        diag["now_schema"] = now_schema if kind == "now" else None

        diag["prompt_after_preview"] = _truncate_text(prompt_after, self.diag_prompt_max_chars)
        if self.diag_save_full_prompt:
            diag["prompt_after_full"] = prompt_after
        diag["prompt_contains_TRUNCATED_marker"] = ("[TRUNCATED]" in prompt_after)

        after_tok_len = None
        if self.tokenizer is not None:
            try:
                after_tok_len = int(len(self.tokenizer.encode(prompt_after, add_special_tokens=False)))
            except Exception:
                after_tok_len = None
        diag["prompt_after_tok_len"] = after_tok_len
        diag["max_prompt_tokens_budget"] = int(self.max_prompt_tokens)

        diag["task_like_kind"] = kind
        diag["task_like_subkind"] = subkind
        diag["task_like_candidate"] = bool(cand)

        opts_map: Dict[str, Dict[str, str]] = {}
        prompt_for_run = prompt_after
        if cand:
            opts_map = _extract_options_map(prompt_after)
            if opts_map:
                require_desc = not (kind == "now" and now_schema == "now_action_mcq")
                prompt_for_run = _append_parsed_options_block(prompt_after, opts_map, require_desc=require_desc)
        diag["parsed_options_count"] = int(len(opts_map))

        bad_words_ids_msrtrv = None
        if subkind == "ms_rtrv_open":
            bad_words_ids_msrtrv = _build_bad_words_ids_from_csv(self.ms_rtrv_open_bad_words)
            diag["ms_rtrv_open_bad_words_enabled"] = bool(bad_words_ids_msrtrv)
        else:
            diag["ms_rtrv_open_bad_words_enabled"] = False

        bad_words_ids_cand = None
        if cand:
            bad_words_ids_cand = _build_bad_words_ids_from_csv(self.cand_bad_words)
            diag["cand_bad_words_enabled"] = bool(bad_words_ids_cand)
            diag["cand_max_retries"] = int(self.cand_max_retries)
        else:
            diag["cand_bad_words_enabled"] = False

        return_logprobs = os.environ.get("RETURN_LOGPROBS", "1").strip() != "0"

        # ============================================================
        # NEW: cand_conf mode for ms_pred/ms_rtrv/sh_pred (candidate) only
        #   - Prompt is rewritten to force output ONLY A/B/C/D (no <ANS>).
        #   - Confidence is measured from next-token logits at the first generation step:
        #       p_raw(A..D) from full softmax
        #       m_ABCD = sum p_raw
        #       p_cond = p_raw / m_ABCD
        # ============================================================
        cand_conf_enabled = (str(getattr(self, "_active_pred_flavor", "") or "").strip().lower() == "cand_conf")
        if cand_conf_enabled and cand and subkind in {"ms_pred_cand", "ms_rtrv_cand", "sh_pred_cand"} and (self.tokenizer is not None):
            def _build_ms_cand_conf_prompt(orig: str, opts: Dict[str, Dict[str, str]], subkind_local: str) -> str:
                first = ""
                try:
                    first = (orig.splitlines()[0] or "").strip()
                except Exception:
                    first = ""
                title = first if first else ("[multistep prediction]" if subkind_local == "ms_pred_cand" else "[multistep past retrieval]")

                lines: List[str] = []
                lines.append(title)
                lines.append("")
                lines.append("Choose exactly ONE option (A/B/C/D) based on the egocentric video up to now.")
                lines.append("Output ONLY the single uppercase letter A, B, C, or D. No other text.")
                lines.append("")
                lines.append("Options:")
                for k in ["A", "B", "C", "D"]:
                    if k in opts and isinstance(opts[k], dict) and str(opts[k].get("raw", "")).strip():
                        lines.append(f"{k}. {str(opts[k].get('raw')).strip()}")
                return "\n".join(lines).strip()

            def _choice_token_ids(tokenizer, letter: str) -> Tuple[List[int], Dict[str, Any]]:
                # Collect single-token variants so we don't guess whether a leading space/newline is used.
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
                    # fallback: use the first token of encoding(letter)
                    try:
                        enc0 = tokenizer.encode(letter, add_special_tokens=False)
                    except Exception:
                        enc0 = []
                    if isinstance(enc0, list) and len(enc0) >= 1:
                        ids = [int(enc0[0])]
                        diag_local["single_token_ids"] = ids
                        diag_local["approx"] = True
                return ids, diag_local

            def _probe_choice_distribution(inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
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
                else:
                    chosen = None

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

            # Build short prompt (ONLY A/B/C/D)
            prompt_conf = _build_ms_cand_conf_prompt(prompt_after, opts_map, subkind)
            diag["cand_conf_prompt_preview"] = _truncate_text(prompt_conf, self.diag_prompt_max_chars)

            # Encode
            if memory_visual_roles is not None:
                inputs, frames_used, prompt_final, encode_diag = _encode_inputs_with_autofit_memory(
                    self.processor,
                    self.tokenizer,
                    prompt_conf,
                    frames,
                    memory_visual_roles,
                    self.device,
                    max_input_tokens=int(self.max_input_tokens),
                )
            else:
                inputs, frames_used, prompt_final, encode_diag = _encode_inputs_with_autofit(
                    self.processor,
                    self.tokenizer,
                    prompt_conf,
                    frames,
                    self.device,
                    max_input_tokens=int(self.max_input_tokens),
                )
            input_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
            encode_diag = dict(encode_diag) if isinstance(encode_diag, dict) else {}
            encode_diag["input_len_from_inputs"] = int(input_len)

            # Probe next-token choice distribution
            probe = _probe_choice_distribution(inputs)
            diag["cand_conf_probe"] = probe
            diag["cand_conf_enabled"] = True
            diag["cand_conf_subkind"] = subkind

            # Small controlled generation (still generate, but stop once we see A-D)
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": int(max(1, min(self.cand_conf_max_new_tokens, 8))),
                "do_sample": False,
                "repetition_penalty": 1.0,
                "no_repeat_ngram_size": 0,
            }
            if self.temperature and self.temperature > 0:
                gen_kwargs.update({"do_sample": True, "temperature": float(self.temperature), "top_p": float(self.top_p)})
            else:
                gen_kwargs.update({"do_sample": False})

            stopping = None
            if self.tokenizer is not None:
                stopping = StoppingCriteriaList([_StopOnFirstABCD(self.tokenizer, prompt_len=input_len, max_decode_tokens=16)])

            attempt_diag: Dict[str, Any] = {
                "encode": encode_diag,
                "final_prompt_preview": _truncate_text(prompt_final, self.diag_prompt_max_chars),
                "gen_kwargs": {
                    "max_new_tokens": int(gen_kwargs.get("max_new_tokens", 0)),
                    "do_sample": bool(gen_kwargs.get("do_sample", False)),
                    "temperature": float(self.temperature),
                    "top_p": float(self.top_p),
                    "repetition_penalty": float(gen_kwargs.get("repetition_penalty", 1.0)),
                    "no_repeat_ngram_size": int(gen_kwargs.get("no_repeat_ngram_size", 0)),
                    "stop_on_first_abcd": True,
                },
            }
            if self.diag_save_full_prompt:
                attempt_diag["final_prompt_full"] = prompt_final

            def _do_generate(with_kwargs: Dict[str, Any]):
                if stopping is not None:
                    with_kwargs = dict(with_kwargs)
                    with_kwargs["stopping_criteria"] = stopping
                if return_logprobs:
                    return self.model.generate(**inputs, **with_kwargs, return_dict_in_generate=True, output_scores=True)
                return self.model.generate(**inputs, **with_kwargs)

            t0 = time.time()
            gen_out = _do_generate(gen_kwargs)
            attempt_diag["latency_sec_generate"] = float(time.time() - t0)

            # Decode
            if return_logprobs:
                gen = gen_out
                seq = gen.sequences[0]
                gen_ids = seq[input_len:] if input_len > 0 else seq

                resp_text_raw = ""
                if self.tokenizer is not None:
                    resp_text_raw = self.tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()

                # Build token probs for generated tokens (as before)
                gen_tokens: List[str] = []
                gen_token_probs: List[float] = []
                sent_logp: Optional[float] = None
                mean_logp: Optional[float] = None

                if self.tokenizer is not None and getattr(gen, "scores", None):
                    logps: List[float] = []
                    for i, score_t in enumerate(gen.scores):
                        pos = input_len + i
                        if pos >= seq.shape[-1]:
                            break
                        tok_id = int(seq[pos].item())
                        lp = float(torch.log_softmax(score_t[0].float(), dim=-1)[tok_id].item())
                        logps.append(lp)
                        gen_token_probs.append(float(math.exp(lp)))
                        gen_tokens.append(self.tokenizer.decode([tok_id], skip_special_tokens=True, clean_up_tokenization_spaces=False))
                    if logps:
                        sent_logp = float(sum(logps))
                        mean_logp = float(sent_logp / max(1, len(logps)))

                # Extract final letter; fallback to argmax(p_cond) if needed
                letter = ""
                m = re.search(r"[A-D]", (resp_text_raw or "").upper())
                if m:
                    letter = m.group(0).upper()

                if (not letter) and probe.get("ok") and self.cand_conf_force_fallback:
                    ch = probe.get("chosen_by_p_cond", None)
                    if isinstance(ch, str) and ch in {"A", "B", "C", "D"}:
                        letter = ch

                resp_text = letter if letter else (resp_text_raw.strip() if resp_text_raw else "")
                diag["attempts"] = [attempt_diag]
                diag["cand_conf_generated_raw"] = _truncate_text(resp_text_raw, self.diag_prompt_max_chars)
                diag["cand_conf_final_letter"] = letter if letter else None
                return resp_text, gen_tokens, gen_token_probs, sent_logp, mean_logp, int(frames_used), diag

            # no logprobs path
            out = gen_out
            seq = out[0]
            gen_ids = seq[input_len:] if input_len > 0 else seq
            resp_text_raw = ""
            if self.tokenizer is not None:
                resp_text_raw = self.tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()

            letter = ""
            m = re.search(r"[A-D]", (resp_text_raw or "").upper())
            if m:
                letter = m.group(0).upper()
            if (not letter) and probe.get("ok") and self.cand_conf_force_fallback:
                ch = probe.get("chosen_by_p_cond", None)
                if isinstance(ch, str) and ch in {"A", "B", "C", "D"}:
                    letter = ch

            resp_text = letter if letter else (resp_text_raw.strip() if resp_text_raw else "")
            diag["attempts"] = [attempt_diag]
            diag["cand_conf_generated_raw"] = _truncate_text(resp_text_raw, self.diag_prompt_max_chars)
            diag["cand_conf_final_letter"] = letter if letter else None
            return resp_text, [], [], None, None, int(frames_used), diag

        # -------------------------
        # Original path (unchanged)
        # -------------------------
        def _run_once(ptext: str, kind_local: str, subkind_local: str, cand_local: bool, *, now_schema_local: str) -> Tuple[str, List[str], List[float], Optional[float], Optional[float], int, Dict[str, Any]]:
            if memory_visual_roles is not None:
                inputs, frames_used, prompt_final, encode_diag = _encode_inputs_with_autofit_memory(
                    self.processor,
                    self.tokenizer,
                    ptext,
                    frames,
                    memory_visual_roles,
                    self.device,
                    max_input_tokens=int(self.max_input_tokens),
                )
            else:
                inputs, frames_used, prompt_final, encode_diag = _encode_inputs_with_autofit(
                    self.processor,
                    self.tokenizer,
                    ptext,
                    frames,
                    self.device,
                    max_input_tokens=int(self.max_input_tokens),
                )
            input_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
            encode_diag = dict(encode_diag) if isinstance(encode_diag, dict) else {}
            encode_diag["input_len_from_inputs"] = int(input_len)

            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": int(self.max_new_tokens),
                "do_sample": False,
                "repetition_penalty": 1.15,
                "no_repeat_ngram_size": 3,
            }

            if kind_local == "now":
                if now_schema_local == "now_state_only":
                    gen_kwargs["max_new_tokens"] = int(min(gen_kwargs["max_new_tokens"], max(8, self.now_state_only_max_new_tokens)))
                elif now_schema_local == "now_action_mcq":
                    gen_kwargs["max_new_tokens"] = int(min(gen_kwargs["max_new_tokens"], max(12, self.now_action_mcq_max_new_tokens)))

            if kind_local in {"pred", "now", "past"}:
                gen_kwargs["repetition_penalty"] = 1.0
                gen_kwargs["no_repeat_ngram_size"] = 0

                if kind_local == "pred":
                    mn = int(self.pred_min_new_tokens)
                elif kind_local == "now":
                    if now_schema_local in {"now_state_only", "now_action_mcq"}:
                        mn = 0
                    else:
                        mn = int(self.now_min_new_tokens)
                else:
                    mn = int(self.past_min_new_tokens)

                if mn < 0:
                    mn = 0
                if mn > int(gen_kwargs["max_new_tokens"]):
                    mn = int(gen_kwargs["max_new_tokens"])
                if mn > 0:
                    gen_kwargs["min_new_tokens"] = mn
                else:
                    gen_kwargs.pop("min_new_tokens", None)

            if subkind_local == "ms_rtrv_open":
                gen_kwargs["repetition_penalty"] = float(self.ms_rtrv_open_rep_penalty)
                gen_kwargs["no_repeat_ngram_size"] = int(self.ms_rtrv_open_no_repeat_ngram)
                gen_kwargs.pop("min_new_tokens", None)
                if bad_words_ids_msrtrv:
                    gen_kwargs["bad_words_ids"] = bad_words_ids_msrtrv

            if cand_local and bad_words_ids_cand:
                if "bad_words_ids" in gen_kwargs and isinstance(gen_kwargs["bad_words_ids"], list):
                    gen_kwargs["bad_words_ids"] = list(gen_kwargs["bad_words_ids"]) + list(bad_words_ids_cand)
                else:
                    gen_kwargs["bad_words_ids"] = bad_words_ids_cand

            if self.temperature and self.temperature > 0:
                gen_kwargs.update({"do_sample": True, "temperature": float(self.temperature), "top_p": float(self.top_p)})
            else:
                gen_kwargs.update({"do_sample": False})

            stopping = None
            if self.stop_on_close_tags and self.tokenizer is not None and kind_local in {"now", "past", "pred"}:
                stopping = StoppingCriteriaList([
                    _StopOnCloseTags(self.tokenizer, prompt_len=input_len, stop_strs_upper=self._close_tag_strs_upper)
                ])

            attempt_diag: Dict[str, Any] = {
                "encode": encode_diag,
                "final_prompt_preview": _truncate_text(prompt_final, self.diag_prompt_max_chars),
                "gen_kwargs": {
                    "max_new_tokens": int(gen_kwargs.get("max_new_tokens", 0)),
                    "min_new_tokens": int(gen_kwargs.get("min_new_tokens", 0)) if kind_local in {"pred", "now", "past"} else 0,
                    "do_sample": bool(gen_kwargs.get("do_sample", False)),
                    "temperature": float(self.temperature),
                    "top_p": float(self.top_p),
                    "repetition_penalty": float(gen_kwargs.get("repetition_penalty", 1.0)),
                    "no_repeat_ngram_size": int(gen_kwargs.get("no_repeat_ngram_size", 0)),
                    "bad_words_ids": bool("bad_words_ids" in gen_kwargs),
                    "stop_on_close_tags": bool(stopping is not None),
                },
            }
            if self.diag_save_full_prompt:
                attempt_diag["final_prompt_full"] = prompt_final

            def _do_generate(with_kwargs: Dict[str, Any]):
                if stopping is not None:
                    with_kwargs = dict(with_kwargs)
                    with_kwargs["stopping_criteria"] = stopping

                if return_logprobs:
                    return self.model.generate(**inputs, **with_kwargs, return_dict_in_generate=True, output_scores=True)
                return self.model.generate(**inputs, **with_kwargs)

            try:
                gen_out = _do_generate(gen_kwargs)
            except TypeError as e:
                if "min_new_tokens" in gen_kwargs:
                    attempt_diag["warn"] = f"min_new_tokens unsupported, fallback: {repr(e)}"
                    gen_kwargs.pop("min_new_tokens", None)
                    gen_out = _do_generate(gen_kwargs)
                else:
                    raise

            if return_logprobs:
                gen = gen_out
                seq = gen.sequences[0]
                gen_ids = seq[input_len:] if input_len > 0 else seq

                if self.tokenizer is not None:
                    resp_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
                else:
                    resp_text = ""

                gen_tokens: List[str] = []
                gen_token_probs: List[float] = []
                sent_logp: Optional[float] = None
                mean_logp: Optional[float] = None

                if self.tokenizer is not None and getattr(gen, "scores", None):
                    logps: List[float] = []
                    for i, score_t in enumerate(gen.scores):
                        pos = input_len + i
                        if pos >= seq.shape[-1]:
                            break
                        tok_id = int(seq[pos].item())
                        lp = float(torch.log_softmax(score_t[0].float(), dim=-1)[tok_id].item())
                        logps.append(lp)
                        gen_token_probs.append(float(math.exp(lp)))
                        gen_tokens.append(self.tokenizer.decode([tok_id], skip_special_tokens=True, clean_up_tokenization_spaces=False))
                    if logps:
                        sent_logp = float(sum(logps))
                        mean_logp = float(sent_logp / max(1, len(logps)))

                return resp_text, gen_tokens, gen_token_probs, sent_logp, mean_logp, frames_used, attempt_diag

            out = gen_out
            seq = out[0]
            gen_ids = seq[input_len:] if input_len > 0 else seq

            if self.tokenizer is not None:
                resp_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
            else:
                resp_text = ""

            return resp_text, [], [], None, None, frames_used, attempt_diag

        # ----- attempt 1 -----
        t0 = time.time()
        resp_text, gen_tokens, gen_token_probs, sent_logp, mean_logp, frames_used, attempt1_diag = _run_once(
            prompt_for_run, kind, subkind, cand, now_schema_local=now_schema
        )
        attempt1_diag["latency_sec_generate"] = float(time.time() - t0)

        attempts = [attempt1_diag]

        def _choose_retry_suffix(kind_local: str, subkind_local: str, cand_local: bool, *, now_schema_local: str) -> Tuple[bool, str]:
            if subkind_local == "ms_rtrv_open":
                return self.past_retry, self.ms_rtrv_open_retry_suffix

            if kind_local == "now":
                if now_schema_local == "now_state_only":
                    return self.now_retry, self.now_state_only_retry_suffix
                if now_schema_local == "now_action_mcq":
                    return self.cand_retry, self.now_action_mcq_retry_suffix
                return self.now_retry, self.now_retry_suffix

            if cand_local:
                if subkind_local == "ms_pred_cand":
                    return self.cand_retry, self.ms_pred_cand_retry_suffix
                return self.cand_retry, self.cand_retry_suffix
            if kind_local == "pred":
                return self.pred_retry, self.pred_retry_suffix
            if kind_local == "past":
                return self.past_retry, self.past_retry_suffix
            return False, ""

        enabled, retry_suffix = _choose_retry_suffix(kind, subkind, cand, now_schema_local=now_schema)

        if enabled and retry_suffix:
            if subkind == "ms_rtrv_open":
                diag["retry_triggered"] = False
                diag["retry_kind"] = kind
                diag["retry_subkind"] = subkind
                max_r = int(max(1, self.ms_rtrv_open_max_retries))
                for ri in range(max_r):
                    bad, reason = _output_bad(kind, subkind, cand, resp_text, opts_map, now_schema=now_schema)
                    if not bad:
                        break
                    diag["retry_triggered"] = True
                    diag["retry_reason"] = reason
                    retry_prompt = (prompt_for_run.rstrip() + "\n\n" + retry_suffix.strip()).strip()

                    t1 = time.time()
                    resp2, tok2, prob2, slp2, mlp2, frames_used2, attempt_diag = _run_once(
                        retry_prompt, kind, subkind, cand, now_schema_local=now_schema
                    )
                    attempt_diag["latency_sec_generate"] = float(time.time() - t1)
                    attempt_diag["retry_suffix_used"] = retry_suffix
                    attempt_diag["retry_index"] = int(ri + 1)

                    attempts.append(attempt_diag)
                    resp_text, gen_tokens, gen_token_probs, sent_logp, mean_logp, frames_used = resp2, tok2, prob2, slp2, mlp2, frames_used2

            elif cand:
                diag["retry_triggered"] = False
                diag["retry_kind"] = kind
                diag["retry_subkind"] = subkind
                max_r = int(max(1, self.cand_max_retries))
                for ri in range(max_r):
                    bad, reason = _output_bad(kind, subkind, cand, resp_text, opts_map, now_schema=now_schema)
                    if not bad:
                        break
                    diag["retry_triggered"] = True
                    diag["retry_reason"] = reason

                    retry_prompt = (prompt_for_run.rstrip() + "\n\n" + retry_suffix.strip()).strip()
                    t1 = time.time()
                    resp2, tok2, prob2, slp2, mlp2, frames_used2, attempt_diag = _run_once(
                        retry_prompt, kind, subkind, cand, now_schema_local=now_schema
                    )
                    attempt_diag["latency_sec_generate"] = float(time.time() - t1)
                    attempt_diag["retry_suffix_used"] = retry_suffix
                    attempt_diag["retry_index"] = int(ri + 1)

                    attempts.append(attempt_diag)
                    resp_text, gen_tokens, gen_token_probs, sent_logp, mean_logp, frames_used = resp2, tok2, prob2, slp2, mlp2, frames_used2

            else:
                bad, reason = _output_bad(kind, subkind, cand, resp_text, opts_map, now_schema=now_schema)
                if bad:
                    diag["retry_triggered"] = True
                    diag["retry_kind"] = kind
                    diag["retry_subkind"] = subkind
                    diag["retry_reason"] = reason

                    retry_prompt = (prompt_for_run.rstrip() + "\n\n" + retry_suffix.strip()).strip()

                    t1 = time.time()
                    resp2, tok2, prob2, slp2, mlp2, frames_used2, attempt2_diag = _run_once(
                        retry_prompt, kind, subkind, cand, now_schema_local=now_schema
                    )
                    attempt2_diag["latency_sec_generate"] = float(time.time() - t1)
                    attempt2_diag["retry_suffix_used"] = retry_suffix

                    attempts.append(attempt2_diag)

                    resp_text, gen_tokens, gen_token_probs, sent_logp, mean_logp, frames_used = resp2, tok2, prob2, slp2, mlp2, frames_used2
                else:
                    diag["retry_triggered"] = False
        else:
            diag["retry_triggered"] = False

        diag["attempts"] = attempts

        final_attempt = attempts[-1] if attempts else {}
        if isinstance(final_attempt, dict):
            diag["encode"] = final_attempt.get("encode", None)
            diag["final_prompt_preview"] = final_attempt.get("final_prompt_preview", None)
            if self.diag_save_full_prompt and "final_prompt_full" in final_attempt:
                diag["final_prompt_full"] = final_attempt.get("final_prompt_full", None)
            diag["gen_kwargs"] = final_attempt.get("gen_kwargs", None)

        # ----- ms cand post-processing AFTER retries -----
        if self.ms_cand_postproc and cand and subkind in {"ms_rtrv_cand", "ms_pred_cand"}:
            coerced, applied, reason = _coerce_ms_cand_to_tags(kind, subkind, resp_text, opts_map)
            diag["ms_cand_postproc_applied"] = bool(applied)
            if applied:
                diag["ms_cand_postproc_reason"] = str(reason)
                resp_text = coerced
            else:
                diag["ms_cand_postproc_reason"] = str(reason)

        return resp_text, gen_tokens, gen_token_probs, sent_logp, mean_logp, frames_used, diag

    def _rewrite_out_path_for_cand_conf(self, out_path: Path) -> Path:
        """
        NEW: ensure cand_conf outputs do NOT overwrite existing /cand/ results.
        If out_path contains .../ms_pred/cand/... or .../ms_rtrv/cand/...or.../sh_pred/cand/...,
        rewrite to .../cand_conf/... and add suffix "__pred__cand_conf.json" when possible.
        """
        s = out_path.as_posix()
        if "/ms_pred/cand/" in s:
            s2 = s.replace("/ms_pred/cand/", "/ms_pred/cand_conf/")
        elif "/ms_rtrv/cand/" in s:
            s2 = s.replace("/ms_rtrv/cand/", "/ms_rtrv/cand_conf/")
        elif "/sh_pred/cand/" in s:
            s2 = s.replace("/sh_pred/cand/", "/sh_pred/cand_conf/")
        elif "/sh_pred_full/cand/" in s:
            s2 = s.replace("/sh_pred_full/cand/", "/sh_pred_full/cand_conf/")
        else:
            # fallback: just create a sibling folder cand_conf under the same task folder if possible
            s2 = s

        p2 = Path(s2)

        name = p2.name
        if "__pred__cand.json" in name:
            name2 = name.replace("__pred__cand.json", "__pred__cand_conf.json")
            p2 = p2.with_name(name2)
        elif name.endswith("__pred.json"):
            p2 = p2.with_name(name[:-9] + "__pred__cand_conf.json")
        elif name.endswith(".json") and ("cand_conf" not in name):
            p2 = p2.with_name(name[:-5] + "__cand_conf.json")

        return p2

    def run(self, *args, **kwargs) -> Path:
        video_path, queryset_path, out_path = self._resolve_call_signature(*args, **kwargs)

        qs = _load_json_or_one_jsonl(queryset_path)
        task = _guess_task(qs)
        params = qs.get("params", {}) if isinstance(qs.get("params"), dict) else {}
        time_offset_sec = float(_safe_float(params.get("time_offset_sec", 0.0), 0.0))

        # NEW: decide active pred flavor (cand / cand_conf / etc.)
        env_flavor = os.environ.get("PRED_FLAVOR", "").strip().lower()
        qs_flavor = str(params.get("pred_flavor", "") or "").strip().lower()
        self._active_pred_flavor = env_flavor if env_flavor else qs_flavor

        # NEW: reroute output file for cand_conf to avoid overwriting /cand/
        if self._active_pred_flavor == "cand_conf" and task in {"ms_pred", "ms_rtrv", "sh_pred", "sh_pred_full"}:
            out_path = self._rewrite_out_path_for_cand_conf(out_path)

        vm = qs.get("video_metadata", {}) if isinstance(qs.get("video_metadata"), dict) else {}
        video_uid = vm.get("video_uid", qs.get("video_uid", ""))

        model_name = os.environ.get("MODEL_NAME", "").strip() or "unnamed_model"

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

        clip_dur = self._probe_clip_duration(video_path, qs)

        out_obj: Dict[str, Any] = {
            "dataset": qs.get("dataset", "Ego-Exo4D"),
            "task": task,
            "video_uid": str(video_uid),
            "model_name": model_name,
            "model_id": self.model_id,
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

            prompt_text = str(s.get("prompt", "")).strip()
            if not prompt_text:
                out_obj["samples"].append({"idx": idx, "error": "missing prompt"})
                _dump_json(out_path, out_obj)
                continue

            memory_diag: Dict[str, Any] = {}
            memory_visual_roles: Optional[List[str]] = None

            if self.memory_mode:
                try:
                    prompt_text, frames, memory_visual_roles, memory_diag = self._prepare_memory_inputs(
                        video_path=video_path,
                        sample=s,
                        original_prompt=prompt_text,
                        qs=qs,
                        clip_dur=clip_dur,
                    )
                except Exception as e:
                    healed = self._try_self_heal_clip(video_path, qs, e)
                    if healed:
                        prompt_text, frames, memory_visual_roles, memory_diag = self._prepare_memory_inputs(
                            video_path=video_path,
                            sample=s,
                            original_prompt=prompt_text,
                            qs=qs,
                            clip_dur=clip_dur,
                        )
                    else:
                        raise
                dense_diag = memory_diag.get("dense_window", {}) if isinstance(memory_diag.get("dense_window"), dict) else {}
                ws_rel = _safe_float(dense_diag.get("start_sec"), 0.0)
                we_rel = _safe_float(dense_diag.get("end_sec"), ws_rel)
                lookback_used = float(max(0.0, we_rel - ws_rel))
            else:
                ws_rel, we_rel, lookback_used = self._infer_window_rel(qs, s, time_offset_sec=time_offset_sec, clip_dur=clip_dur)

                try:
                    frames, _ = _load_frames_decord(video_path, ws_rel, we_rel, int(self.num_frames))
                except Exception as e:
                    healed = self._try_self_heal_clip(video_path, qs, e)
                    if healed:
                        frames, _ = _load_frames_decord(video_path, ws_rel, we_rel, int(self.num_frames))
                    else:
                        raise

            t0 = time.time()
            resp_text, gen_tokens, gen_token_probs, sent_logp, mean_logp, frames_used, gen_diag = self._generate_one(
                prompt_text,
                frames,
                memory_visual_roles=memory_visual_roles,
            )
            dt = time.time() - t0

            clean = _first_tag_block(resp_text)
            parsed = _parse_xmlish(clean)

            # NEW: for cand_conf (letter-only), store parsed.ans for convenience (does not affect normal modes)
            if self._active_pred_flavor == "cand_conf":
                m = re.search(r"[A-D]", (clean or "").upper())
                if m:
                    parsed.setdefault("ans", m.group(0).upper())

            rec = {
                "idx": idx,
                "t_eval": _safe_float(s.get("t_eval", 0.0), 0.0),
                "t_eval_rel": _safe_float(s.get("t_eval_rel", _safe_float(s.get("t_eval", 0.0), 0.0) - time_offset_sec), 0.0),
                "prompt": prompt_text,
                "response_text": resp_text,
                "clean_response": clean,
                "parsed": parsed,
                "gen_tokens": gen_tokens,
                "gen_token_probs": gen_token_probs,
                "sent_logp": sent_logp,
                "mean_logp": mean_logp,
                "latency_sec": float(dt),
                "diagnostics": {
                    "window_start_sec": float(ws_rel + time_offset_sec),
                    "window_end_sec": float(we_rel + time_offset_sec),
                    "window_start_rel": float(ws_rel),
                    "window_end_rel": float(we_rel),
                    "lookback_used_sec": float(lookback_used),
                    "frames_extracted": int(frames.shape[0]) if isinstance(frames, np.ndarray) else int(self.num_frames),
                    "frames_used": int(frames_used),
                    "model_max_len": int(self.model_max_len),
                    "max_input_tokens": int(self.max_input_tokens),
                    "clip_dur_sec": float(clip_dur),
                    "prompt_and_encoding_debug": gen_diag,
                    "memory": memory_diag if self.memory_mode else {"enabled": False},
                },
                "raw": None,
            }
            out_obj["samples"].append(rec)
            _dump_json(out_path, out_obj)

        _dump_json(out_path, out_obj)
        return out_path


def create_adapter():
    return _Qwen25VLAdapter()
