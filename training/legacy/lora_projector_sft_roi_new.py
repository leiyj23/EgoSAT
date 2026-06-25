#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal training script (single file) + DDP multi-GPU support.

Run:
  CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 train_min_ddp.py

Notes:
- DDP speeds up (data parallel). Per-GPU memory ~ same as 1 GPU.
- Effective batch = world_size * BATCH_SIZE * GRAD_ACCUM.
  If you want same effective batch as single-GPU, halve GRAD_ACCUM when using 2 GPUs.
"""

from __future__ import annotations

import os
import sys
import re
import json
import math
import time
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from PIL import Image
from decord import VideoReader, cpu  # type: ignore

from transformers import AutoProcessor

# PEFT (LoRA)
from peft import LoraConfig, get_peft_model, PeftModel


# -------------------------
# Public defaults. This resume-capable reference is not the default public
# path for mixed7_stateheavy; ../train_full_sft.py runs mixed7 independently
# unless --resume-from is explicitly provided.
# -------------------------
MANIFEST_PATH = Path(
    os.environ.get("MANIFEST_PATH", "/path/to/egosat_sft/train_manifest_mixed7_stateheavy.sanitized.jsonl")
).expanduser()
OUT_DIR = Path(os.environ.get("OUT_DIR", "outputs/sft/roi_timechat_mixed7_stateheavy_resume")).expanduser()

# Previous training output dir (used to auto-find latest checkpoint to continue training)
PREV_TRAIN_OUT_DIR = Path(os.environ.get("PREV_TRAIN_OUT_DIR", "/path/to/previous/roi_timechat_adapter")).expanduser()

# Optional manual override:
#   RESUME_DIR=/path/to/.../step_0000123
# or
#   RESUME_DIR=/path/to/.../final
# or
#   RESUME_DIR=/path/to/.../checkpoints
RESUME_DIR_ENV = os.environ.get("RESUME_DIR", "").strip()

# Where full Ego4D videos live (used only when interval clip is missing)
VIDEO_ROOT = Path(os.environ.get("EGO4D_ROOT", os.environ.get("VIDEO_ROOT", "/path/to/ego4d/full_scale"))).expanduser()

# fps=1 cache dir (training-specific)
FPS1_CACHE_DIR = Path(os.environ.get("FPS1_CACHE_DIR", "outputs/sft/_fps1_cache")).expanduser()

# TimeChat repo root (needed for importing your ROI-DTD model class)
TIMECHAT_REPO_ROOT = Path(os.environ.get("TIMECHAT_REPO_ROOT", "/path/to/TimeChat-Online")).expanduser()

# HF model id (same as your adapter)
MODEL_ID = os.environ.get("TIMECHAT_HF_MODEL_ID", "wyccccc/TimeChatOnline-7B").strip()


# -------------------------
# Training knobs (keep simple)
# -------------------------
SEED = int(float(os.environ.get("SEED", "1234")))
EPOCHS = int(float(os.environ.get("EPOCHS", "1")))

BATCH_SIZE = int(float(os.environ.get("BATCH_SIZE", "1")))  # per-GPU
GRAD_ACCUM = int(float(os.environ.get("GRAD_ACCUM", "8")))
LR = float(os.environ.get("LR", "2e-4"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.0"))
MAX_STEPS = int(float(os.environ.get("MAX_STEPS", "0")))  # 0 => no cap; if resume, means extra steps to run

MAX_FRAMES = int(float(os.environ.get("MAX_FRAMES", "16")))  # training default smaller than inference

SAVE_EVERY_STEPS = int(float(os.environ.get("SAVE_EVERY_STEPS", "500")))
LOG_EVERY = int(float(os.environ.get("LOG_EVERY", "20")))

# DTD/ROI (optional in training)
USE_DTD_IN_TRAIN = os.environ.get("USE_DTD_IN_TRAIN", "1").strip() != "0"
DROP_METHOD = os.environ.get("TIMECHAT_DROP_METHOD", "roi_feature").strip()
DROP_THRESHOLD = float(os.environ.get("TIMECHAT_DROP_THRESHOLD", "0.85"))
DROP_ABSOLUTE = os.environ.get("TIMECHAT_DROP_ABSOLUTE", "1").strip() != "0"

# NEW: attention impl (match adapter behavior)
ATTN_IMPL = os.environ.get("TIMECHAT_ATTN_IMPL", "flash_attention_2").strip()
SDPA_MAX_FRAMES = int(float(os.environ.get("TIMECHAT_SDPA_MAX_FRAMES", "6")))
EFFECTIVE_MAX_FRAMES = min(MAX_FRAMES, SDPA_MAX_FRAMES) if ATTN_IMPL == "sdpa" else MAX_FRAMES

# LoRA target modules (Qwen-like default)
LORA_R = int(float(os.environ.get("LORA_R", "16")))
LORA_ALPHA = int(float(os.environ.get("LORA_ALPHA", "32")))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))
LORA_TARGETS = os.environ.get(
    "LORA_TARGETS",
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
).strip().split(",")

# Which extra modules to unfreeze (projector/merger)
UNFREEZE_PATTERNS = os.environ.get(
    "UNFREEZE_PATTERNS",
    "mm_projector,projector,merger,multi_modal_projector,visual.merger"
).strip().split(",")

# NEW: debug vision tokens
DEBUG_VTOK = os.environ.get("DEBUG_VTOK", "0").strip() != "0"

# NEW: resize controls (NO crop, NO pad)
TRAIN_LONG_SIDE = int(float(os.environ.get("TRAIN_LONG_SIDE", "448")))          # longest side after resize
TRAIN_PATCH_MULT = int(float(os.environ.get("TRAIN_PATCH_MULT", "14")))        # round H/W to multiple of 14


# -------------------------
# DDP helpers
# -------------------------
def _is_dist() -> bool:
    return ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ) and int(os.environ.get("WORLD_SIZE", "1")) > 1

def _rank() -> int:
    return int(os.environ.get("RANK", "0"))

def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))

def _is_main() -> bool:
    return (_rank() == 0)

def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def _ddp_setup() -> torch.device:
    if _is_dist():
        dist.init_process_group(backend="nccl")
        lrk = _local_rank()
        torch.cuda.set_device(lrk)
        return torch.device("cuda", lrk)
    # single GPU / CPU
    return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

def _ddp_cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# -------------------------
# Utilities
# -------------------------
def _seed_everything(seed: int, rank_offset: int = 0) -> None:
    s = int(seed) + int(rank_offset)
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def _load_jsonl_lines(p: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            items.append(json.loads(ln))
    return items

def _ensure_repo_on_syspath(repo_root: Path) -> None:
    rr = str(repo_root.expanduser().resolve())
    if rr not in sys.path:
        sys.path.insert(0, rr)

def _run_ffmpeg(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (rc={p.returncode}).\nCMD: {' '.join(cmd)}\nSTDERR:\n{p.stderr[-4000:]}"
        )

def _resolve_source_video(video_uid: str) -> Path:
    p = VIDEO_ROOT / f"{video_uid}.mp4"
    if p.exists():
        return p
    hits = list(VIDEO_ROOT.rglob(f"{video_uid}.mp4"))
    if hits:
        hits.sort(key=lambda x: x.as_posix())
        return hits[0]
    raise FileNotFoundError(f"Cannot find source video for uid={video_uid} under {VIDEO_ROOT}")

def _cut_interval_clip(src_video: Path, dst_clip: Path, start_sec: float, end_sec: float) -> None:
    dst_clip.parent.mkdir(parents=True, exist_ok=True)
    start_sec = float(max(0.0, start_sec))
    end_sec = float(max(start_sec, end_sec))
    dur = end_sec - start_sec

    tmp = dst_clip.with_name(dst_clip.stem + f".tmp_rank{_rank()}" + dst_clip.suffix)

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
        str(tmp),
    ]
    _run_ffmpeg(cmd)
    if not tmp.exists() or tmp.stat().st_size < 1024:
        raise RuntimeError(f"ffmpeg cut produced invalid file: {tmp}")
    os.replace(str(tmp), str(dst_clip))

def _ffmpeg_fps1(src_video: Path, dst_video: Path) -> None:
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_video.with_name(dst_video.stem + f".tmp_rank{_rank()}" + dst_video.suffix)
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
    _run_ffmpeg(cmd)
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

def _load_roi_cache_jsonl_to_map(path: Path) -> Dict[int, Dict[str, Any]]:
    by_idx: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return by_idx
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            if "frame_idx" in obj:
                try:
                    by_idx[int(obj["frame_idx"])] = obj
                except Exception:
                    pass
    return by_idx

def _build_roi_cache_for_indices(
    roi_by_frameidx: Dict[int, Dict[str, Any]],
    indices: List[int],
    frame_w: Optional[int],
    frame_h: Optional[int],
    coord: str = "norm",
    gaze_r_default: float = 0.12,
) -> Dict[str, Any]:
    frames: List[Dict[str, Any]] = []
    for idx in indices:
        fr = dict(roi_by_frameidx.get(int(idx), {}) or {})
        fr.setdefault("coord", coord)
        if "gaze" in fr and isinstance(fr["gaze"], dict):
            fr["gaze"].setdefault("r", gaze_r_default)
        if coord == "pixel":
            if frame_w is not None:
                fr.setdefault("frame_w", frame_w)
            if frame_h is not None:
                fr.setdefault("frame_h", frame_h)
        frames.append(fr)
    return {"frames": frames, "frame_w": frame_w, "frame_h": frame_h}

def _infer_window_rel(params: Dict[str, Any], sample: Dict[str, Any], clip_dur_sec: float) -> Tuple[float, float, float]:
    time_offset = float(params.get("time_offset_sec", 0.0) or 0.0)

    t_eval_rel = sample.get("t_eval_rel", None)
    if t_eval_rel is None:
        t_eval = float(sample.get("t_eval", 0.0) or 0.0)
        t_eval_rel = t_eval - time_offset
    t_eval_rel = float(max(0.0, float(t_eval_rel)))

    ws_sched = sample.get("window_start_sec", None)
    we_sched = sample.get("window_end_sec", None)

    if (ws_sched is not None) and (we_sched is not None):
        ws_rel = float(ws_sched) - time_offset
        we_rel = float(we_sched) - time_offset
    else:
        lookback = sample.get("lookback_sec", None)
        if lookback is None:
            lookback = params.get("lookback_sec", 20.0)
        lookback = float(lookback or 20.0)
        ws_rel = float(max(0.0, t_eval_rel - lookback))
        we_rel = float(t_eval_rel)

    we_rel = min(float(we_rel), float(t_eval_rel))
    ws_rel = max(0.0, min(float(ws_rel), float(we_rel)))

    if clip_dur_sec > 0:
        ws_rel = float(max(0.0, min(ws_rel, clip_dur_sec)))
        we_rel = float(max(0.0, min(we_rel, clip_dur_sec)))

    return t_eval_rel, ws_rel, we_rel


# NEW: round helpers + resize (NO crop, NO pad)
def _round_to_mult(x: int, mult: int) -> int:
    if mult <= 1:
        return max(1, int(x))
    return max(mult, int(round(x / float(mult))) * mult)

def _resize_keep_aspect_no_crop(
    images: List[Image.Image],
    long_side: int,
    mult: int,
) -> List[Image.Image]:
    """
    Resize each image with the same aspect ratio, no crop, no pad.
    - scale so that max(W,H) == long_side
    - round W/H to multiples of `mult` (e.g., 14) to stabilize visual token grids
    """
    if not images:
        return images
    if long_side <= 0:
        return images

    w0, h0 = images[0].size
    max0 = max(w0, h0)
    if max0 <= 0:
        return images

    scale = float(long_side) / float(max0)
    if abs(scale - 1.0) < 1e-6:
        new_w = _round_to_mult(w0, mult)
        new_h = _round_to_mult(h0, mult)
    else:
        new_w = _round_to_mult(int(round(w0 * scale)), mult)
        new_h = _round_to_mult(int(round(h0 * scale)), mult)

    new_w = max(mult, new_w)
    new_h = max(mult, new_h)

    if new_w == w0 and new_h == h0:
        return images

    out: List[Image.Image] = []
    for im in images:
        out.append(im.resize((new_w, new_h), resample=Image.BICUBIC))
    return out

# NEW: sanity check ROI cache is normalized 0~1 when coord=="norm"
def _roi_all_in_01(vals: List[float], eps: float = 1e-6) -> bool:
    return all((v >= -eps) and (v <= 1.0 + eps) for v in vals)

def _assert_roi_cache_norm(roi_cache: Dict[str, Any]) -> None:
    frames = roi_cache.get("frames", [])
    if not isinstance(frames, list) or not frames:
        return
    for fr in frames:
        if not isinstance(fr, dict):
            continue
        coord = fr.get("coord", "norm")
        if coord != "norm":
            raise ValueError(f"roi_cache coord is not 'norm' (got {coord!r}); refusing to proceed.")
        for key in ["hand_boxes", "left_box", "right_box"]:
            v = fr.get(key, None)
            if v is None:
                continue
            if key == "hand_boxes":
                if isinstance(v, list):
                    for box in v:
                        if isinstance(box, (list, tuple)) and len(box) == 4:
                            if not _roi_all_in_01([float(x) for x in box]):
                                raise ValueError(f"roi_cache {key} has non-normalized values: {box}")
            else:
                if isinstance(v, (list, tuple)) and len(v) == 4:
                    if not _roi_all_in_01([float(x) for x in v]):
                        raise ValueError(f"roi_cache {key} has non-normalized values: {v}")
        gaze = fr.get("gaze", None)
        if isinstance(gaze, dict):
            if "x" in gaze and "y" in gaze:
                xy = [float(gaze["x"]), float(gaze["y"])]
                if not _roi_all_in_01(xy):
                    raise ValueError(f"roi_cache gaze has non-normalized values: {xy}")


# NEW: debug printer
def _debug_print_vision_tokens(
    *,
    images: List[Image.Image],
    indices: List[int],
    inputs: Dict[str, Any],
) -> None:
    if not DEBUG_VTOK:
        return
    if not _is_main():
        return

    print("\n[DEBUG_VTOK] ----- vision input summary -----")
    print(f"[DEBUG_VTOK] num_images={len(images)} indices={indices}")
    if images:
        for i, im in enumerate(images):
            w, h = im.size
            print(f"[DEBUG_VTOK] frame[{i}] idx={indices[i] if i < len(indices) else 'NA'} PIL_size={w}x{h}")

    pv = inputs.get("pixel_values", None)
    if pv is not None and hasattr(pv, "shape"):
        try:
            print(f"[DEBUG_VTOK] pixel_values.shape={tuple(pv.shape)} dtype={getattr(pv, 'dtype', None)} device={getattr(pv, 'device', None)}")
        except Exception:
            pass

    igt = inputs.get("image_grid_thw", None)
    if igt is not None and hasattr(igt, "shape"):
        try:
            thw_list = igt.detach().cpu().tolist()
            print(f"[DEBUG_VTOK] image_grid_thw.shape={tuple(igt.shape)} values(first 32)={thw_list[:32]}")
            toks = []
            for j, thw in enumerate(thw_list):
                if isinstance(thw, (list, tuple)) and len(thw) == 3:
                    t, h, w = int(thw[0]), int(thw[1]), int(thw[2])
                    n = t * h * w
                    toks.append(n)
                    print(f"[DEBUG_VTOK] frame[{j}] grid_thw=({t},{h},{w}) -> tokens={n}")
            if toks:
                print(f"[DEBUG_VTOK] total_visual_tokens(sum t*h*w)={sum(toks)}")
        except Exception as e:
            print(f"[DEBUG_VTOK] failed to parse image_grid_thw: {repr(e)}")
    else:
        print("[DEBUG_VTOK] image_grid_thw not found in processor outputs (cannot compute exact visual token count).")

    print("[DEBUG_VTOK] -----------------------------------\n")


def _extract_step_from_name(p: Path) -> int:
    m = re.search(r"step_(\d+)$", p.name)
    return int(m.group(1)) if m else -1

def _find_latest_step_dir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None

    candidates: List[Path] = []

    if root.is_dir() and _extract_step_from_name(root) >= 0:
        candidates.append(root)

    for ch in root.glob("step_*"):
        if ch.is_dir() and _extract_step_from_name(ch) >= 0:
            candidates.append(ch)

    for sub in ["final", "checkpoints"]:
        d = root / sub
        if d.exists():
            for ch in d.glob("step_*"):
                if ch.is_dir() and _extract_step_from_name(ch) >= 0:
                    candidates.append(ch)

    if not candidates:
        return None

    candidates.sort(key=lambda x: (_extract_step_from_name(x), x.as_posix()))
    return candidates[-1]

def resolve_resume_dir() -> Path:
    if RESUME_DIR_ENV:
        root = Path(RESUME_DIR_ENV).expanduser().resolve()
        found = _find_latest_step_dir(root)
        if found is None:
            raise FileNotFoundError(f"RESUME_DIR provided but no valid step_* checkpoint found under: {root}")
        return found

    found = _find_latest_step_dir(PREV_TRAIN_OUT_DIR)
    if found is None:
        raise FileNotFoundError(
            f"Could not auto-find previous checkpoint under {PREV_TRAIN_OUT_DIR}. "
            f"Please set RESUME_DIR=/path/to/step_xxxxxxx explicitly."
        )
    return found


# -------------------------
# Dataset
# -------------------------
class ManifestDataset(Dataset):
    def __init__(self, manifest_path: Path) -> None:
        assert manifest_path.exists(), f"manifest not found: {manifest_path}"
        items = _load_jsonl_lines(manifest_path)

        kept: List[Dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            tgt = it.get("target_text", None)
            samp = it.get("sample", None)
            if not tgt or not isinstance(samp, dict):
                continue
            if not str(samp.get("prompt", "")).strip():
                continue
            kept.append(it)

        if not kept:
            raise RuntimeError("No usable samples in manifest (need target_text + sample.prompt).")
        self.items = kept

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


def collate_one(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    assert len(batch) == 1
    return batch[0]


# -------------------------
# Model loading / freezing
# -------------------------
def load_timechat_roi_model_and_processor(device: torch.device) -> Tuple[torch.nn.Module, Any]:
    _ensure_repo_on_syspath(TIMECHAT_REPO_ROOT)

    from eval.qwen2_5_vl.modeling_qwen2_5_vl_DTD_ROI import Qwen2_5_VLForConditionalGeneration  # type: ignore

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    load_err = None
    model = None
    for impl in [ATTN_IMPL, "sdpa"]:
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                trust_remote_code=True,
                attn_implementation=impl,
            )
            used_impl = impl
            load_err = None
            break
        except Exception as e:
            load_err = e
            model = None
            used_impl = "sdpa"

    if model is None:
        raise RuntimeError(f"Failed to load model with attn_impl={ATTN_IMPL!r} fallback sdpa.\nError={repr(load_err)}")

    model.to(device)
    model.train()
    if hasattr(model, "config"):
        model.config.use_cache = False

    setattr(model, "_attn_impl_effective", used_impl)

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model, processor

def apply_lora_and_freeze(model: torch.nn.Module) -> torch.nn.Module:
    for p in model.parameters():
        p.requires_grad = False

    cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[t.strip() for t in LORA_TARGETS if t.strip()],
    )
    model = get_peft_model(model, cfg)

    patterns = [p.strip() for p in UNFREEZE_PATTERNS if p.strip()]
    unfrozen = []
    for n, p in model.named_parameters():
        ln = n.lower()
        if any(pt.lower() in ln for pt in patterns):
            p.requires_grad = True
            unfrozen.append(n)

    if _is_main():
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[MODEL] total_params={total:,} trainable_params={trainable:,} ({100.0*trainable/total:.4f}%)")
        if unfrozen:
            print("[MODEL] extra_unfrozen(example up to 20):")
            for x in unfrozen[:20]:
                print("  ", x)
        else:
            print("[WARN] No extra params unfrozen by UNFREEZE_PATTERNS.")

    return model


def _collect_expected_extra_trainable_keys(model: torch.nn.Module) -> List[str]:
    keys: List[str] = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in n.lower():
            continue
        keys.append(n)
    keys.sort()
    return keys


def _strict_load_extra_trainable(model: torch.nn.Module, extra_path: Path) -> List[str]:
    extra_state = torch.load(str(extra_path), map_location="cpu", weights_only=True)
    if not isinstance(extra_state, dict):
        raise RuntimeError(f"extra_trainable.pt is not a dict: {extra_path}")

    expected_keys = _collect_expected_extra_trainable_keys(model)
    ckpt_keys = sorted(list(extra_state.keys()))

    missing_in_ckpt = [k for k in expected_keys if k not in extra_state]
    unexpected_in_ckpt = [k for k in ckpt_keys if k not in expected_keys]

    if missing_in_ckpt or unexpected_in_ckpt:
        parts: List[str] = []
        if missing_in_ckpt:
            parts.append(
                f"[RESUME] extra_trainable missing keys in checkpoint: {len(missing_in_ckpt)}\n"
                + "\n".join(missing_in_ckpt[:50])
            )
        if unexpected_in_ckpt:
            parts.append(
                f"[RESUME] extra_trainable unexpected keys in checkpoint: {len(unexpected_in_ckpt)}\n"
                + "\n".join(unexpected_in_ckpt[:50])
            )
        raise RuntimeError("\n\n".join(parts))

    param_map = {n: p for n, p in model.named_parameters()}
    shape_mismatch: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []

    for k in expected_keys:
        if k not in param_map:
            raise RuntimeError(f"[RESUME] expected extra key not found in current model params: {k}")
        src = extra_state[k]
        dst = param_map[k]
        if tuple(src.shape) != tuple(dst.shape):
            shape_mismatch.append((k, tuple(src.shape), tuple(dst.shape)))

    if shape_mismatch:
        msg = "\n".join([f"{k}: ckpt{src_shape} != model{dst_shape}" for k, src_shape, dst_shape in shape_mismatch[:20]])
        raise RuntimeError(
            f"[RESUME] extra_trainable shape mismatch: {len(shape_mismatch)} keys.\n{msg}"
        )

    with torch.no_grad():
        for k in expected_keys:
            src = extra_state[k]
            dst = param_map[k]
            dst.copy_(src.to(device=dst.device, dtype=dst.dtype))

    return expected_keys


def resume_lora_and_extra(model: torch.nn.Module, resume_dir: Path) -> Tuple[torch.nn.Module, int]:
    if not resume_dir.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {resume_dir}")

    for p in model.parameters():
        p.requires_grad = False

    model = PeftModel.from_pretrained(model, str(resume_dir), is_trainable=True)

    patterns = [p.strip() for p in UNFREEZE_PATTERNS if p.strip()]
    unfrozen = []
    for n, p in model.named_parameters():
        ln = n.lower()
        if any(pt.lower() in ln for pt in patterns):
            p.requires_grad = True
            unfrozen.append(n)

    extra_path = resume_dir / "extra_trainable.pt"
    if extra_path.exists():
        loaded_extra_keys = _strict_load_extra_trainable(model, extra_path)
        if _is_main():
            print(f"[RESUME] loaded extra trainable weights: {extra_path}")
            print(f"[RESUME] matched extra keys = {len(loaded_extra_keys)}")
            if loaded_extra_keys:
                print("[RESUME] sample loaded extra keys:")
                for k in loaded_extra_keys[:20]:
                    print("  ", k)
    else:
        if _is_main():
            print(f"[WARN] extra_trainable.pt not found under {resume_dir}")

    start_step = 0
    meta_path = resume_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            start_step = int(meta.get("step", 0))
        except Exception:
            start_step = 0

    if _is_main():
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[RESUME] resume_dir={resume_dir}")
        print(f"[MODEL] total_params={total:,} trainable_params={trainable:,} ({100.0*trainable/total:.4f}%)")
        if unfrozen:
            print("[MODEL] resumed extra_unfrozen(example up to 20):")
            for x in unfrozen[:20]:
                print("  ", x)

    model.train()
    return model, start_step


# -------------------------
# Build one training sample
# -------------------------
def build_inputs_for_one(
    model: torch.nn.Module,
    processor: Any,
    rec: Dict[str, Any],
) -> Tuple[Dict[str, Any], torch.Tensor, Optional[Dict[str, Any]]]:
    sample = rec["sample"]
    prompt = str(sample["prompt"]).strip()
    target_text = str(rec["target_text"]).strip()

    interval_clip_path = Path(rec["interval_clip_path"]).expanduser()
    vm = rec.get("video_metadata", {}) if isinstance(rec.get("video_metadata"), dict) else {}
    video_uid = str(rec.get("video_uid", vm.get("video_uid", ""))).strip()

    interval_start = float(vm.get("interval_start_sec", 0.0) or 0.0)
    interval_end = float(vm.get("interval_end_sec", interval_start) or interval_start)

    if not interval_clip_path.exists():
        if _is_dist():
            if _is_main():
                src_video = _resolve_source_video(video_uid)
                _cut_interval_clip(src_video, interval_clip_path, interval_start, interval_end)
            _barrier()
        else:
            src_video = _resolve_source_video(video_uid)
            _cut_interval_clip(src_video, interval_clip_path, interval_start, interval_end)

    FPS1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fps1_path = FPS1_CACHE_DIR / f"{interval_clip_path.stem}__fps1.mp4"
    if not fps1_path.exists():
        if _is_dist():
            if _is_main():
                _ffmpeg_fps1(interval_clip_path, fps1_path)
            _barrier()
        else:
            _ffmpeg_fps1(interval_clip_path, fps1_path)

    vr = _get_vr(fps1_path)
    n_total = len(vr)
    if n_total <= 0:
        raise RuntimeError(f"fps1 video has no frames: {fps1_path}")
    clip_dur_sec = float(n_total - 1)

    params = rec.get("params", {}) if isinstance(rec.get("params"), dict) else {}
    _, ws_rel, we_rel = _infer_window_rel(params, sample, clip_dur_sec=clip_dur_sec)

    indices = _select_indices_1fps(ws_rel, we_rel, n_total=n_total, max_frames=EFFECTIVE_MAX_FRAMES)
    images = _frames_to_images(vr, indices)
    images = _resize_keep_aspect_no_crop(images, long_side=TRAIN_LONG_SIDE, mult=TRAIN_PATCH_MULT)

    roi_cache = None
    roi_path = rec.get("roi_cache_path", None)
    if USE_DTD_IN_TRAIN and roi_path:
        rp = Path(str(roi_path)).expanduser()
        if rp.exists() and DROP_METHOD.lower().startswith("roi"):
            roi_map = _load_roi_cache_jsonl_to_map(rp)
            w, h = images[0].size
            roi_cache = _build_roi_cache_for_indices(
                roi_by_frameidx=roi_map,
                indices=indices,
                frame_w=w,
                frame_h=h,
                coord="norm",
                gaze_r_default=0.12,
            )
            _assert_roi_cache_norm(roi_cache)

    messages = [{
        "role": "user",
        "content": ([{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]),
    }]
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    tok = getattr(processor, "tokenizer", None)
    eos = tok.eos_token if (tok is not None and getattr(tok, "eos_token", None)) else ""
    text_full = (text_prompt + target_text + eos)

    inputs = processor(text=[text_full], images=images, padding=True, return_tensors="pt")

    try:
        _debug_print_vision_tokens(images=images, indices=indices, inputs=inputs)
    except Exception as e:
        if DEBUG_VTOK and _is_main():
            print(f"[DEBUG_VTOK] exception in debug print: {repr(e)}")

    dev = next(model.parameters()).device
    if hasattr(inputs, "to"):
        inputs = inputs.to(dev)
    else:
        inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}

    try:
        if "second_per_grid_ts" in inputs and hasattr(inputs["second_per_grid_ts"], "to"):
            inputs["second_per_grid_ts"] = inputs["second_per_grid_ts"].to("cpu")
    except Exception:
        pass

    if tok is None:
        raise RuntimeError("processor.tokenizer is None; cannot build labels safely.")

    prompt_ids = tok(text_prompt, add_special_tokens=False).input_ids
    prompt_len = int(len(prompt_ids))

    input_ids = inputs["input_ids"]
    labels = input_ids.clone()
    if prompt_len > labels.shape[1]:
        labels[:] = -100
    else:
        labels[:, :prompt_len] = -100

    attn = inputs.get("attention_mask", None)
    if attn is not None:
        labels = labels.masked_fill(attn == 0, -100)

    if DEBUG_VTOK and _is_main():
        try:
            print(f"[DEBUG_VTOK] text_prompt_len_tokens={prompt_len} total_input_ids_len={int(input_ids.shape[1])}")
        except Exception:
            pass

    return inputs, labels, roi_cache


# -------------------------
# Save
# -------------------------
def _unwrap_model(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if isinstance(m, DDP) else m

def save_checkpoint(model: torch.nn.Module, out_dir: Path, step: int) -> None:
    if not _is_main():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    ck = out_dir / f"step_{step:07d}"
    ck.mkdir(parents=True, exist_ok=True)

    m = _unwrap_model(model)

    try:
        m.save_pretrained(str(ck))
    except Exception as e:
        print(f"[WARN] model.save_pretrained failed: {repr(e)}")

    extra = {}
    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in n.lower():
            continue
        extra[n] = p.detach().cpu()
    if extra:
        torch.save(extra, str(ck / "extra_trainable.pt"))

    with open(ck / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"step": step, "time": time.time()}, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] {ck}")


# -------------------------
# Main train
# -------------------------
def main() -> None:
    device = _ddp_setup()
    _seed_everything(SEED, rank_offset=_rank())

    resume_dir = resolve_resume_dir()

    if _is_main():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    FPS1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _barrier()

    if _is_main():
        print(f"[INFO] dist={_is_dist()} world_size={_world_size()} rank={_rank()} local_rank={_local_rank()}")
        print(f"[INFO] device={device}")
        print(f"[INFO] MANIFEST={MANIFEST_PATH}")
        print(f"[INFO] OUT_DIR={OUT_DIR}")
        print(f"[INFO] RESUME_DIR={resume_dir}")
        print(f"[INFO] ATTN_IMPL(requested)={ATTN_IMPL} SDPA_MAX_FRAMES={SDPA_MAX_FRAMES} EFFECTIVE_MAX_FRAMES={EFFECTIVE_MAX_FRAMES}")
        print(f"[INFO] MAX_FRAMES={MAX_FRAMES} BATCH_SIZE(perGPU)={BATCH_SIZE} GRAD_ACCUM={GRAD_ACCUM}")
        print(f"[INFO] USE_DTD_IN_TRAIN={USE_DTD_IN_TRAIN} DROP_METHOD={DROP_METHOD}")
        print(f"[INFO] DEBUG_VTOK={DEBUG_VTOK}")
        print(f"[INFO] TRAIN_LONG_SIDE={TRAIN_LONG_SIDE} TRAIN_PATCH_MULT={TRAIN_PATCH_MULT}")

    dataset = ManifestDataset(MANIFEST_PATH)
    if _is_main():
        print(f"[DATA] dataset_size={len(dataset)}")

    sampler = None
    if _is_dist():
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=False)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_one,
        pin_memory=(device.type == "cuda"),
    )

    model, processor = load_timechat_roi_model_and_processor(device)
    if _is_main():
        eff = getattr(model, "_attn_impl_effective", "unknown")
        print(f"[INFO] ATTN_IMPL(effective)={eff}")

    model, start_step = resume_lora_and_extra(model, resume_dir)

    if _is_dist():
        model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

    #global_step = int(start_step)
    global_step=0
    target_global_step = (int(start_step) + MAX_STEPS) if MAX_STEPS > 0 else 0

    if _is_main():
        if MAX_STEPS > 0:
            print(f"[INFO] start_step={start_step} target_global_step={target_global_step} (run {MAX_STEPS} extra steps)")
        else:
            print(f"[INFO] start_step={start_step} target_global_step=unbounded")

    model.train()

    for ep in range(EPOCHS):
        if sampler is not None:
            sampler.set_epoch(ep)

        if _is_main():
            print(f"\n[EPOCH] {ep+1}/{EPOCHS}")

        optim.zero_grad(set_to_none=True)

        for it, rec in enumerate(loader):
            inputs, labels, roi_cache = build_inputs_for_one(_unwrap_model(model), processor, rec)

            try:
                if USE_DTD_IN_TRAIN and roi_cache is not None:
                    out = model(
                        **inputs,
                        labels=labels,
                        drop_method=str(DROP_METHOD),
                        drop_threshold=float(DROP_THRESHOLD),
                        drop_absolute=bool(DROP_ABSOLUTE),
                        roi_cache=roi_cache,
                    )
                else:
                    out = model(**inputs, labels=labels)
            except TypeError:
                out = model(**inputs, labels=labels)

            loss = out.loss
            if loss is None:
                raise RuntimeError("Model returned loss=None; cannot train.")

            (loss / float(GRAD_ACCUM)).backward()

            if (it + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

                if _is_main() and (global_step % LOG_EVERY == 0):
                    print(f"[STEP {global_step}] loss={float(loss.item()):.4f}")

                if SAVE_EVERY_STEPS > 0 and (global_step % SAVE_EVERY_STEPS == 0):
                    save_checkpoint(model, OUT_DIR / "checkpoints", global_step)

                if target_global_step > 0 and global_step >= target_global_step:
                    break

        if target_global_step > 0 and global_step >= target_global_step:
            break

    save_checkpoint(model, OUT_DIR / "final", global_step)
    if _is_main():
        print("\n[DONE] training finished.")
    _ddp_cleanup()


if __name__ == "__main__":
    main()
