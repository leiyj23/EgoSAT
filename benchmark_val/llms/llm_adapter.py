#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
openrouter_task_adapter.py

Task-agnostic OpenRouter adapter for Ego-Exo4D *short-horizon* (strict-online) video QA.

It works for ANY task as long as the (helper-produced) effective queryset provides:
  - samples[*].t_eval   (query time in loaded video-time seconds)
  - samples[*].prompt   (task-specific prompt already built by helper)
Optional:
  - params.lookback_sec or samples[*].lookback_sec
  - samples[*].window_start_sec / window_end_sec (explicit override)
  - samples[*].candidates (list[str]) for candidate-style tasks

Runner contract:
  - create_adapter() -> adapter instance
  - adapter.run(video_path, queryset_path, pred_flavor, out_json_path) -> Path

Env:
  OPENROUTER_API_KEY: required
  OPENROUTER_MODEL: default "openai/gpt-4o"
  OPENROUTER_URL:   default "https://openrouter.ai/api/v1/chat/completions"
  OPENROUTER_REFERER / OPENROUTER_TITLE: optional headers

  FRAME_FPS: default 2
  MAX_FRAMES_PER_QUERY: default 72
  FRAME_SHORT_SIDE: default 384
  KEEP_TMP_FRAMES: default 0

  LOOKBACK_OVERRIDE_SEC: if set (>0), overrides everything
  LOOKBACK_DEFAULT_SEC: default 20

  MAX_TOKENS: default 128
  TEMPERATURE: default 0.0
  REQUEST_TIMEOUT_SEC: default 120
  RETRY: default 2

  ADAPTER_PROGRESS: default 1 (0 disables tqdm)
  ADAPTER_PROGRESS_DESC: optional prefix
  SAVE_RAW_RESPONSE: default 0 (1 saves raw OpenRouter JSON per sample)

CLI:
  --vision on|off   default: on
    - on : use visual frames normally
    - off: text-only ablation (no frames, no image inputs)
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_slug(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "NA"
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in s)


def _run_ffmpeg(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (rc={p.returncode}).\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDERR(last 4000 chars):\n{p.stderr[-4000:]}"
        )


def _uniform_downsample(items: List[Path], k: int) -> List[Path]:
    if k <= 0 or len(items) <= k:
        return items
    n = len(items)
    if k == 1:
        return [items[-1]]
    idxs = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    out: List[Path] = []
    seen = set()
    for idx in idxs:
        idx = int(max(0, min(n - 1, idx)))
        if idx not in seen:
            out.append(items[idx])
            seen.add(idx)
    return out


def _b64_data_url_jpg(img_path: Path) -> str:
    data = img_path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _align_to_grid(t: float, fps: float) -> float:
    if fps <= 0:
        return t
    return math.floor(t * fps) / fps


def _parse_vision_enabled_from_argv(default: bool = True) -> bool:
    """
    Read --vision on|off from current process argv.

    This adapter is usually imported and called by runner.py within the SAME
    Python process, so parsing sys.argv here allows:
        python runner.py --mode cand --vision off
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--vision", type=str, default=None)
    try:
        args, _ = parser.parse_known_args(sys.argv[1:])
        v = args.vision
    except Exception:
        v = None

    if v is None:
        return bool(default)

    vv = str(v).strip().lower()
    if vv in {"off", "0", "false", "no", "none", "text", "text_only", "text-only"}:
        return False
    if vv in {"on", "1", "true", "yes", "vision", "visual", "image", "images"}:
        return True
    return bool(default)


def extract_frames_window(
    *,
    video_path: Path,
    start_sec: float,
    end_sec: float,
    fps: float,
    short_side: int,
    max_frames: int,
    work_dir: Path,
    keep_tmp: bool = False,
) -> Tuple[List[str], Dict[str, Any]]:
    _ensure_dir(work_dir)
    diag: Dict[str, Any] = {
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
        "fps": float(fps),
        "short_side": int(short_side),
        "max_frames": int(max_frames),
        "aligned_start_sec": None,
        "extracted_frames": 0,
        "used_frames": 0,
    }

    start_sec = max(0.0, float(start_sec))
    end_sec = max(0.0, float(end_sec))
    if end_sec < start_sec:
        end_sec = start_sec

    aligned_start = _align_to_grid(start_sec, fps)
    diag["aligned_start_sec"] = float(aligned_start)

    eps = 1e-3
    duration = max(0.0, end_sec - aligned_start)
    out_pattern = str(work_dir / "frame_%06d.jpg")

    if duration < 1e-2:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{end_sec:.6f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-vf", f"scale={short_side}:-1",
            "-q:v", "3",
            "-an",
            out_pattern,
        ]
        _run_ffmpeg(cmd)
    else:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{aligned_start:.6f}",
            "-t", f"{(duration ):.6f}",
            "-i", str(video_path),
            "-vf", f"fps={fps},scale={short_side}:-1",
            "-q:v", "3",
            "-an",
            out_pattern,
        ]
        _run_ffmpeg(cmd)

    frames = sorted(work_dir.glob("frame_*.jpg"))
    diag["extracted_frames"] = len(frames)

    frames = _uniform_downsample(frames, max_frames)
    diag["used_frames"] = len(frames)

    data_urls = [_b64_data_url_jpg(p) for p in frames]

    if not keep_tmp:
        shutil.rmtree(work_dir, ignore_errors=True)

    return data_urls, diag


def try_parse_logprobs(resp_json: Dict[str, Any]) -> Tuple[List[str], List[float], Optional[float], Optional[float]]:
    try:
        choice0 = resp_json["choices"][0]
    except Exception:
        return [], [], None, None

    lp = choice0.get("logprobs")
    if not lp:
        return [], [], None, None

    content = lp.get("content")
    if isinstance(content, list) and content:
        toks: List[str] = []
        probs: List[float] = []
        logps: List[float] = []
        for item in content:
            tok = item.get("token")
            l = item.get("logprob")
            if tok is None or l is None:
                continue
            toks.append(str(tok))
            logps.append(float(l))
            probs.append(float(math.exp(float(l))))
        if not toks:
            return [], [], None, None
        sent_logp = float(sum(logps))
        mean_logp = float(sent_logp / max(1, len(logps)))
        return toks, probs, sent_logp, mean_logp

    return [], [], None, None


class OpenRouterTaskAdapter:
    def __init__(self):
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY")

        self.url = os.environ.get("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions").strip()
        self.model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o").strip()

        self.referer = os.environ.get("OPENROUTER_REFERER", "").strip()
        self.title = os.environ.get("OPENROUTER_TITLE", "").strip()

        self.frame_fps = float(os.environ.get("FRAME_FPS", "2"))
        self.max_frames = int(os.environ.get("MAX_FRAMES_PER_QUERY", "72"))
        self.frame_short_side = int(os.environ.get("FRAME_SHORT_SIDE", "384"))

        lb_override = os.environ.get("LOOKBACK_OVERRIDE_SEC", "").strip()
        self.lookback_override_sec: Optional[float] = None
        if lb_override:
            try:
                v = float(lb_override)
                if v > 0:
                    self.lookback_override_sec = v
            except Exception:
                self.lookback_override_sec = None

        self.lookback_default_sec = float(os.environ.get("LOOKBACK_DEFAULT_SEC", "20"))

        self.max_tokens = int(os.environ.get("MAX_TOKENS", "2048"))
        self.temperature = float(os.environ.get("TEMPERATURE", "0.0"))
        self.timeout = int(os.environ.get("REQUEST_TIMEOUT_SEC", "120"))
        self.retry = int(os.environ.get("RETRY", "2"))
        self.keep_tmp = os.environ.get("KEEP_TMP_FRAMES", "0").strip() == "1"

        self.progress_enabled = os.environ.get("ADAPTER_PROGRESS", "1").strip() != "0"
        self.progress_desc_prefix = os.environ.get("ADAPTER_PROGRESS_DESC", "").strip()

        self.vision_enabled = _parse_vision_enabled_from_argv(default=True)

        self.system_prompt = (
            "You are an assistant doing strict online egocentric video QA. "
            "You will be given only past frames up to the query time. "
            "Follow the user's required output format EXACTLY."
        )

    def _headers(self) -> Dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.referer:
            h["HTTP-Referer"] = self.referer
        if self.title:
            h["X-Title"] = self.title
        return h

    def _call_openrouter(self, content_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content_list},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "logprobs": True,
        }

        last_err = None
        for attempt in range(self.retry + 1):
            try:
                r = requests.post(self.url, headers=self._headers(), json=payload, timeout=self.timeout)
                if r.status_code >= 400:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:1000]}")
                return r.json()
            except Exception as e:
                last_err = e
                if attempt < self.retry:
                    time.sleep(1.0 * (attempt + 1))
                else:
                    raise RuntimeError(f"OpenRouter call failed after retries: {repr(last_err)}") from last_err

        raise RuntimeError("Unreachable")

    def _infer_task_name(self, qs: Dict[str, Any], queryset_path: Path) -> str:
        for k in ("task", "task_name", "benchmark_task"):
            v = qs.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return queryset_path.parent.name if queryset_path.parent else "unknown"

    def _infer_lookback(self, qs: Dict[str, Any], sample: Dict[str, Any]) -> float:
        if self.lookback_override_sec is not None:
            return float(self.lookback_override_sec)

        for k in ("lookback_sec", "lookback"):
            if k in sample:
                try:
                    v = float(sample[k])
                    if v > 0:
                        return v
                except Exception:
                    pass

        params = qs.get("params")
        if isinstance(params, dict) and ("lookback_sec" in params):
            try:
                v = float(params.get("lookback_sec"))
                if v > 0:
                    return v
            except Exception:
                pass

        return float(self.lookback_default_sec)

    def _infer_window(self, qs: Dict[str, Any], sample: Dict[str, Any]) -> Tuple[float, float, float]:
        """
        Returns (window_start_sec, window_end_sec, lookback_used_sec).

        HARD strict-online safety:
        - window_end_sec is ALWAYS clamped to t_eval (no look-ahead),
            even if helper mistakenly provides a larger value.
        - window_start_sec is clamped to [0, window_end_sec].
        """
        t_eval = float(sample.get("t_eval", 0.0))

        # Only treat explicit window override as valid if the values are not None.
        ws_raw = sample.get("window_start_sec", None)
        we_raw = sample.get("window_end_sec", None)

        has_ws = (ws_raw is not None)
        has_we = (we_raw is not None)

        lookback = self._infer_lookback(qs, sample)

        if has_ws or has_we:
            ws = float(ws_raw) if has_ws else (t_eval - lookback)
            we = float(we_raw) if has_we else t_eval

            # ---- STRICT ONLINE CLAMP (no look-ahead) ----
            we = min(we, t_eval)
            ws = max(0.0, min(ws, we))

            lb = max(0.0, we - ws)
            return ws, we, lb

        # Default: [t_eval - lookback, t_eval]
        ws = max(0.0, t_eval - lookback)
        we = t_eval
        return ws, we, float(lookback)

    def run(self, video_path: Path, queryset_path: Path, pred_flavor: str, out_json_path: Path) -> Path:
        video_path = Path(video_path)
        queryset_path = Path(queryset_path)
        out_json_path = Path(out_json_path)

        qs = json.loads(queryset_path.read_text(encoding="utf-8"))
        task_name = self._infer_task_name(qs, queryset_path)

        dataset = qs.get("dataset", "Ego-Exo4D")
        video_uid = qs.get("video_uid", qs.get("uid", ""))
        meta = qs.get("video_metadata", {}) or {}
        scenario = meta.get("scenario", "")
        take_name = meta.get("take_name", "")

        samples_in = qs.get("samples", []) or []
        if not isinstance(samples_in, list):
            raise ValueError(f"Queryset 'samples' must be a list: {queryset_path}")

        def _key(x: Any) -> Tuple[float, int]:
            t = float(x.get("t_eval", 0.0))
            i = int(x.get("idx", 10**9))
            return (t, i)

        samples_in = sorted([s for s in samples_in if isinstance(s, dict)], key=_key)

        run_id = os.environ.get("RUN_ID", "run001").strip()
        model_name = os.environ.get("MODEL_NAME", self.model).strip() or self.model

        out: Dict[str, Any] = {
            "dataset": dataset,
            "task": task_name,
            "mode": pred_flavor,
            "video_uid": video_uid,
            "model_name": model_name,
            "run_id": run_id,
            "source_video_path": str(video_path),
            "source_frame_fps": self.frame_fps,
            "adapter": {
                "name": self.__class__.__name__,
                "openrouter_model": self.model,
                "frame_fps": self.frame_fps,
                "max_frames_per_query": self.max_frames,
                "frame_short_side": self.frame_short_side,
                "lookback_override_sec": self.lookback_override_sec,
                "lookback_default_sec": self.lookback_default_sec,
                "vision_enabled": self.vision_enabled,
            },
            "video_metadata": {
                "scenario": scenario,
                "take_name": take_name,
                "queryset_path": str(queryset_path),
                "original_video_metadata": meta,
            },
            "params": qs.get("params", {}),
            "samples": [],
        }

        _ensure_dir(out_json_path.parent)

        pbar = None
        if self.progress_enabled and tqdm is not None:
            desc = self.progress_desc_prefix or "Adapter"
            short_id = video_uid or take_name or "video"
            pbar = tqdm(total=len(samples_in), desc=f"{desc} [{short_id}]", unit="qa", dynamic_ncols=True, leave=True)

        for s in samples_in:
            idx = int(s.get("idx", -1))
            t_eval = float(s.get("t_eval", 0.0))
            prompt = s.get("prompt", "")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Missing sample.prompt for idx={idx} in queryset: {queryset_path}")

            start_sec, end_sec, lookback_used = self._infer_window(qs, s)

            # safety net for candidate-style tasks (prefer helper to bake candidates into prompt)
            candidates = s.get("candidates")
            if isinstance(candidates, list) and candidates and str(pred_flavor).lower().startswith("cand"):
                cand_txt = "\n".join([f"- {str(c)}" for c in candidates])
                if "Candidates:" not in prompt:
                    prompt = prompt.rstrip() + "\n\nCandidates:\n" + cand_txt + "\n"

            if pbar is not None:
                pbar.set_postfix_str(f"idx={idx} t={t_eval:.1f}s")

            if self.vision_enabled:
                tmp_name = f"tmp_frames__{_safe_slug(str(video_uid or take_name))}__idx{idx:04d}"
                tmp_dir = out_json_path.parent / "_tmp_frames" / tmp_name

                frame_urls, diag = extract_frames_window(
                    video_path=video_path,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    fps=self.frame_fps,
                    short_side=self.frame_short_side,
                    max_frames=self.max_frames,
                    work_dir=tmp_dir,
                    keep_tmp=self.keep_tmp,
                )

                preface = (
                    f"[Context] Task={task_name} mode={pred_flavor}. "
                    f"Frames are sampled at {self.frame_fps} FPS from {diag['aligned_start_sec']:.1f}s to {end_sec:.1f}s "
                    f"(strict online; no future frames). Lookback_used={lookback_used:.1f}s, cap={self.max_frames} frames."
                )
            else:
                frame_urls = []
                diag = {
                    "start_sec": float(start_sec),
                    "end_sec": float(end_sec),
                    "fps": float(self.frame_fps),
                    "short_side": int(self.frame_short_side),
                    "max_frames": int(self.max_frames),
                    "aligned_start_sec": None,
                    "extracted_frames": 0,
                    "used_frames": 0,
                }
                preface = (
                    f"[Context] Task={task_name} mode={pred_flavor}. "
                    f"No visual frames are provided for this ablation. "
                    f"Answer using the prompt only."
                )

            content_list: List[Dict[str, Any]] = [{"type": "text", "text": preface}]
            for url in frame_urls:
                content_list.append({"type": "image_url", "image_url": {"url": url}})
            content_list.append({"type": "text", "text": prompt})

            resp_json = self._call_openrouter(content_list)

            try:
                resp_text = resp_json["choices"][0]["message"]["content"]
            except Exception:
                resp_text = json.dumps(resp_json)[:2000]

            gen_tokens, gen_token_probs, sent_logp, mean_logp = try_parse_logprobs(resp_json)

            out["samples"].append(
                {
                    "idx": idx,
                    "t_eval": t_eval,
                    "prompt": prompt,
                    "response_text": resp_text,
                    "gen_tokens": gen_tokens,
                    "gen_token_probs": gen_token_probs,
                    "sent_logp": sent_logp,
                    "mean_logp": mean_logp,
                    "diagnostics": {
                        "vision_enabled": self.vision_enabled,
                        "window_start_sec": start_sec,
                        "window_end_sec": end_sec,
                        "lookback_used_sec": lookback_used,
                        "frames_extracted": diag.get("extracted_frames", 0),
                        "frames_used": diag.get("used_frames", 0),
                        "aligned_start_sec": diag.get("aligned_start_sec", None),
                    },
                    "raw": {"openrouter": resp_json} if os.environ.get("SAVE_RAW_RESPONSE", "0").strip() == "1" else None,
                }
            )

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        out_json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_json_path


def create_adapter():
    return OpenRouterTaskAdapter()