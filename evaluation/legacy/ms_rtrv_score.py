#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ms_rtrv_eval.py

One-click evaluation for multi-step retrieval (ms_rtrv).

Supports:
- cand (default)
- cand_conf

Task setting:
- Each anchor event yields a group of 3 queries (n_steps=3), at lags 8/16/24 seconds after anchor end.
- We evaluate:
  (1) Accuracy by step_idx / lag_sec (0/1/2 -> 8/16/24)
  (2) Strong distractor choice rate (option_sources in {"prev","future"}) by step
  (3) Token-prob confidence (ONLY for open-source outputs with gen_tokens/gen_token_probs):
      - mean conf over valid predictions (token-derived only)
      - mean conf over correct predictions (token-derived only)
      - mean conf over wrong predictions (token-derived only)
      - mean conf over valid/correct/wrong predictions ONLY when the answer is inferred by OPTION LETTER
            (i.e., conf_source == "token_ans")
  (4) Monotonic diagnostics per anchor group (CONF ONLY):
      - First segment monotonicity rate:   c0 >= c1   among groups where ALL 3 steps are correct AND all 3 conf available
      - Second segment monotonicity rate:  c1 >= c2   among same set
      - Overall monotonicity rate:         c0 >= c1 >= c2 among same set
  (5) Linear trend slope beta (CONF ONLY):
      - Fit c = alpha + beta * r, where r in {1,2,3} corresponds to steps (0,1,2).
      - Report beta statistics; beta<0 means confidence decreases as lag increases.

[NEW - cand_conf confidence]
- In cand_conf mode, confidence is taken from:
    diagnostics.prompt_and_encoding_debug.cand_conf_probe.p_cond[ANS]
  which is the normalized probability mass over {A,B,C,D}.
- In cand_conf mode, additionally report, for each step:
    - mean cand_conf over valid predictions
    - mean cand_conf over correct predictions
    - mean cand_conf over wrong predictions

Key idea:
- We DO NOT build a global MCQ index.
- For each pred file, load its derived queryset (path resolved by searching pred json for a string containing "queryset").
- Then match pred samples to queryset samples by idx, and evaluate using queryset["samples"][i]["mcq"].

Inputs:
  ~/benchmark_val/testllm/<MODEL_NAME>/ms_rtrv/{cand|cand_conf}/*.json

Outputs:
  ~/benchmark_val/score/<MODEL_NAME>/ms_rtrv_{cand|cand_conf}/{summary.json, details.jsonl, teacher_cache.json}

Run:
  MODEL_NAME=qwen2_5_vl_7b python3 ms_rtrv_eval.py
  MODEL_NAME=qwen2_5_vl_7b python3 ms_rtrv_eval.py --mode cand_conf

Teacher (OpenRouter) is optional:
  read -r -s OPENROUTER_API_KEY
  export OPENROUTER_API_KEY
  export OPENROUTER_MODEL=...
"""

from __future__ import annotations

import os
import re
import json
import time
import hashlib
import argparse
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import requests

try:
    from .mcq_source_stats import (
        source_stats_report_lines,
        summarize_source_stats,
    )
except ImportError:
    from mcq_source_stats import (
        source_stats_report_lines,
        summarize_source_stats,
    )


# -------------------------
# Hard-coded paths / knobs
# -------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini_pro").strip() or "gemini_pro"

# default mode
MODE = "cand"  # "cand" | "cand_conf"

PRED_DIR = Path.home() / "benchmark_val" / "testllm" / MODEL_NAME / "ms_rtrv" / "cand"
OUT_DIR = Path.home() / "benchmark_val" / "score" / MODEL_NAME / "ms_rtrv_cand"

OUT_SUMMARY = OUT_DIR / "summary.json"
OUT_DETAILS = OUT_DIR / "details.jsonl"
OUT_CACHE = OUT_DIR / "teacher_cache.json"

# Teacher (OpenRouter) - optional
ENABLE_TEACHER_FALLBACK = os.environ.get("EGOSAT_ENABLE_LEGACY_TEACHER_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip() if ENABLE_TEACHER_FALLBACK else ""
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-3-pro-preview").strip()
OPENROUTER_URL = os.environ.get("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions").strip()
OPENROUTER_TIMEOUT_SEC = int(float(os.environ.get("OPENROUTER_TIMEOUT_SEC", "60")))
OPENROUTER_RETRY = int(float(os.environ.get("OPENROUTER_RETRY", "2")))

# progress logging knobs
PRED_LOG_EVERY_FILES = 5
SAMPLE_LOG_EVERY = 500

TEACHER_PRINT_OK_SNIPPET = 220
TEACHER_PRINT_ERR_SNIPPET = 600
RESP_SNIPPET = 240

# strong distractor definition (aligned with your generator option_sources)
STRONG_DISTURB_SOURCES = {"prev", "future"}


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# -------------------------
# Regex helpers
# -------------------------
_TAG_RE = {
    "ans": re.compile(r"<\s*ANS\s*>(.*?)<\s*/\s*ANS\s*>", re.IGNORECASE | re.DOTALL),
    "conf": re.compile(r"<\s*CONF\s*>(.*?)<\s*/\s*CONF\s*>", re.IGNORECASE | re.DOTALL),
}

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_ANGLE_BR_RE = re.compile(r"<[^>]*>")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _snippet(s: str, n: int) -> str:
    s = (s or "").replace("\n", "\\n")
    return s if len(s) <= n else (s[:n] + "...")


def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def parse_tag(text: str, which: str) -> Optional[str]:
    rx = _TAG_RE.get(which)
    if not rx:
        return None
    m = rx.search(text or "")
    if not m:
        return None
    return (m.group(1) or "").strip()


def parse_conf_from_text(text: str) -> Optional[float]:
    s = parse_tag(text, "conf")
    if s is None:
        return None
    try:
        v = float(str(s).strip())
    except Exception:
        return None
    if v < 0.0:
        v = 0.0
    if v > 1.0:
        v = 1.0
    return float(v)


def normalize_ans_letter(x: str) -> str:
    s = (x or "").strip().upper()
    s = re.sub(r"[^A-D]", "", s)  # keep only A-D chars
    return s[0] if s and s[0] in "ABCD" else ""


def _ans_to_idx(ans: str) -> int:
    a = normalize_ans_letter(ans)
    if not a:
        return -1
    return ord(a) - ord("A")


def _strip_angle_brackets(text: str) -> str:
    return _ANGLE_BR_RE.sub(" ", text or "")


def _guess_ans_letter_from_clean(clean_text: str) -> str:
    """
    Try to find exactly ONE candidate in {A,B,C,D} from clean text AFTER removing <...>.
    If multiple distinct letters appear, return "" (ambiguous).
    """
    t = _strip_angle_brackets(clean_text)
    if not t.strip():
        return ""

    hits: List[str] = []
    hits += re.findall(r"(?:^|\n)\s*([A-D])\s*[\.\)]\s*", t)
    hits += re.findall(r"\b(?:CHOOSE|ANSWER|ANS|OPTION)\s*[:=\-]?\s*([A-D])\b", t, flags=re.IGNORECASE)
    hits += re.findall(r"\b([A-D])\b", t)

    uniq: List[str] = []
    seen = set()
    for h in hits:
        u = (h or "").strip().upper()
        if u in {"A", "B", "C", "D"} and u not in seen:
            seen.add(u)
            uniq.append(u)

    starters = set(re.findall(r"(?:^|\n)\s*([A-D])\s*[\.\)\|]\s*", t))
    if len(starters) >= 2:
        return ""

    return uniq[0] if len(uniq) == 1 else ""


# -------------------------
# JSON loaders
# -------------------------
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


# -------------------------
# Pred file listing (STRICT)
# -------------------------
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
        if name.endswith("__pred.json") or name.endswith("_pred.json") or "__pred" in name:
            out.append(p)
        else:
            skipped += 1
    _log(f"Pred list: {pred_dir} -> {len(out)} files (skipped_non_pred={skipped})")
    return out


# -------------------------
# Find queryset path in pred (robust)
# -------------------------
def _collect_queryset_candidates(obj: Any, key_path: str = "") -> List[Tuple[str, str]]:
    """
    Recursively collect (key_path, value) where:
      - key contains "queryset" (case-insensitive) AND value is a string
      - OR value string contains "queryset" (case-insensitive)
    """
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
    Resolve the derived queryset path from a pred file.
    Priority:
      1) pred["source_queryset"]
      2) pred["video_metadata"]["queryset_path"]
      3) any nested key/value containing "queryset" (choose first existing file)
    """
    # (1)
    v = pred.get("source_queryset", None)
    if isinstance(v, str) and v.strip():
        p = Path(v).expanduser()
        if p.is_file():
            return str(p)

    # (2)
    vm = pred.get("video_metadata", {}) if isinstance(pred.get("video_metadata"), dict) else {}
    v2 = vm.get("queryset_path", None)
    if isinstance(v2, str) and v2.strip():
        p = Path(v2).expanduser()
        if p.is_file():
            return str(p)

    # (3)
    cands = _collect_queryset_candidates(pred)
    for kp, vv in cands:
        if not isinstance(vv, str) or not vv.strip():
            continue
        p = Path(vv).expanduser()
        if p.is_file():
            return str(p)

    return None


# -------------------------
# Teacher (OpenRouter) - optional
# -------------------------
def openrouter_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }


def _openrouter_chat(system_prompt: str, user_prompt: str, max_tokens: int = 1600) -> Tuple[str, Optional[str]]:
    if not OPENROUTER_API_KEY:
        return "", "missing OPENROUTER_API_KEY"

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": int(max_tokens),
    }

    last_err: Optional[str] = None
    for attempt in range(OPENROUTER_RETRY + 1):
        t0 = time.time()
        try:
            _log(f"Teacher HTTP POST attempt {attempt+1}/{OPENROUTER_RETRY+1} -> {OPENROUTER_URL}")

            r = requests.post(
                OPENROUTER_URL,
                headers=openrouter_headers(),
                json=payload,
                timeout=OPENROUTER_TIMEOUT_SEC,
            )

            dt = time.time() - t0
            status = int(getattr(r, "status_code", 0) or 0)

            if status >= 400:
                body = ""
                try:
                    body = r.text
                except Exception:
                    body = "<no r.text>"
                _log(f"Teacher HTTP ERROR {status} (t={dt:.2f}s): {_snippet(body, TEACHER_PRINT_ERR_SNIPPET)}")
                raise RuntimeError(f"HTTP {status}")

            try:
                resp = r.json()
            except Exception as e:
                body = ""
                try:
                    body = r.text
                except Exception:
                    body = "<no r.text>"
                _log(f"Teacher JSON decode error (t={dt:.2f}s): {repr(e)} | body={_snippet(body, TEACHER_PRINT_ERR_SNIPPET)}")
                raise

            if isinstance(resp, dict) and "error" in resp:
                _log(f"Teacher API ERROR field (t={dt:.2f}s): {_snippet(str(resp.get('error')), TEACHER_PRINT_ERR_SNIPPET)}")
                raise RuntimeError("OpenRouter error field present")

            txt = resp["choices"][0]["message"]["content"]
            _log(f"Teacher OK (t={dt:.2f}s): {_snippet(str(txt), TEACHER_PRINT_OK_SNIPPET)}")
            return str(txt), None

        except Exception as e:
            last_err = repr(e)
            _log(f"Teacher exception: {last_err}")
            if attempt < OPENROUTER_RETRY:
                time.sleep(0.8 * (attempt + 1))
            else:
                return "", last_err

    return "", last_err or "unknown error"


def teacher_map_answer_to_option(response_text: str, options: List[str]) -> Tuple[str, Dict[str, Any]]:
    """
    Same spirit as sh_rtrv:
    - Teacher returns "" ONLY when extremely uncertain / no usable signal.
    - Otherwise choose A/B/C/D.
    """
    diag: Dict[str, Any] = {"used": True, "task": "map_ans", "model": OPENROUTER_MODEL, "ok": False, "raw_text": None, "error": None}

    letters = ["A", "B", "C", "D"]
    opt_lines = []
    for i in range(min(4, len(options))):
        opt_lines.append(f"{letters[i]}. {options[i]}")
    opt_block = "\n".join(opt_lines) if opt_lines else "(no options provided)"

    sys_prompt = (
        "You are a strict grader.\n"
        "Given ONLY (1) the student's response text and (2) four multiple-choice options (A-D), "
        "decide which option the student's response is MOST CONSISTENT WITH.\n\n"
        "Output JSON ONLY with exactly one key:\n"
        '  {"ans":"A"}\n\n'
        "Rules:\n"
        "- ans must be one of A/B/C/D.\n"
        "- You may output ans=\"\" ONLY if you are EXTREMELY uncertain because the response contains virtually no usable evidence.\n"
        "- If there is ANY usable signal (explicit letter choice, or verb/noun/phrase matching one option), you MUST choose A/B/C/D.\n"
        "- If evidence is mixed or ambiguous, still choose the SINGLE best match.\n"
        "- Do NOT default to A.\n"
        "- Output JSON only. No commentary. No extra keys."
    )
    user_prompt = (
        "Options:\n" + opt_block + "\n\n"
        "Student response text:\n-----\n" + (response_text or "").strip() + "\n-----\n\n"
        "Now output JSON."
    )

    _log(f"Teacher call: map_ans (timeout={OPENROUTER_TIMEOUT_SEC}s, model={OPENROUTER_MODEL})")
    txt, err = _openrouter_chat(sys_prompt, user_prompt, max_tokens=96)
    diag["raw_text"] = txt
    diag["error"] = err
    if err:
        return "", diag

    ans_raw = ""
    try:
        obj = json.loads(txt)
        ans_raw = str(obj.get("ans", "") if obj else "")
    except Exception:
        m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                ans_raw = str(obj.get("ans", "") if obj else "")
            except Exception:
                ans_raw = ""

    if ans_raw is None or str(ans_raw).strip() == "":
        diag["ok"] = False
        return "", diag

    ans = normalize_ans_letter(str(ans_raw))
    diag["ok"] = bool(ans)
    return ans, diag


# -------------------------
# Token-prob based confidence extraction
# -------------------------
def _norm_tok(t: str) -> str:
    return (t or "").strip()


def _find_tag_span(tokens: List[str], tag: str) -> Optional[Tuple[int, int]]:
    """
    Return (content_start_idx, content_end_idx) for the FIRST <TAG>...</TAG> span.
    Works with either:
      - "<ANS>" in one token
      - "<", "ANS", ">" split tokens (also tolerates "<", "ANS>")
    Similar for closing: "</ANS>" or "</", "ANS", ">"
    """
    if not tokens:
        return None
    T = [str(x) for x in tokens]
    U = [x.upper() for x in T]
    tagU = tag.upper()

    start_close = None
    end_open = None

    for i in range(len(U)):
        if f"<{tagU}>" in U[i]:
            start_close = i
            break
        if i + 2 < len(U):
            if "<" in U[i] and tagU in U[i + 1] and ">" in U[i + 2]:
                start_close = i + 2
                break
        if i + 1 < len(U):
            if "<" in U[i] and (tagU in U[i + 1] and ">" in U[i + 1]):
                start_close = i + 1
                break
        if i + 1 < len(U):
            if (f"<{tagU}" in U[i]) and ">" in U[i + 1]:
                start_close = i + 1
                break

    if start_close is None:
        return None
    content_start = start_close + 1

    for j in range(content_start, len(U)):
        if f"</{tagU}>" in U[j]:
            end_open = j
            break
        if j + 2 < len(U):
            if "</" in U[j] and tagU in U[j + 1] and ">" in U[j + 2]:
                end_open = j
                break
        if j + 1 < len(U):
            if f"</{tagU}" in U[j] and ">" in U[j + 1]:
                end_open = j
                break

    if end_open is None:
        return None

    if end_open < content_start:
        return None
    return content_start, end_open


def _extract_single_ans_from_tokens(gen_tokens: List[str], gen_probs: List[float]) -> Tuple[str, Optional[float], str]:
    """
    Try to extract a UNIQUE A/B/C/D from gen_tokens with token-prob confidence.

    Priority:
      1) letter inside <ANS>...</ANS> span
      2) prefix before first "<" token (free-form head)

    Returns (ans_letter_or_empty, conf_or_None, method_str)
      method_str in {"token_ans_span","token_prefix","token_multi","token_none"}
    """
    if (not gen_tokens) or (not gen_probs) or len(gen_tokens) != len(gen_probs):
        return "", None, "token_none"

    toks = [str(x) for x in gen_tokens]
    probs = gen_probs

    span = _find_tag_span([_norm_tok(x) for x in toks], tag="ANS")
    if span is not None:
        a, b = span
        found: List[Tuple[int, str]] = []
        for i in range(a, b):
            t = _norm_tok(toks[i]).strip()
            m = re.search(r"([A-D])", t.upper())
            if m:
                found.append((i, m.group(1).upper()))
        uniq = sorted({x[1] for x in found})
        if len(uniq) == 1:
            ii = min([x[0] for x in found if x[1] == uniq[0]])
            try:
                return uniq[0], float(probs[ii]), "token_ans_span"
            except Exception:
                return uniq[0], None, "token_ans_span"
        if len(uniq) >= 2:
            return "", None, "token_multi"

    letters: List[Tuple[int, str]] = []
    for i, t0 in enumerate(toks):
        t = _norm_tok(t0)
        if "<" in t:
            break
        if not t:
            continue
        m = re.match(r"^\s*([A-D])\b", t.upper())
        if m:
            letters.append((i, m.group(1).upper()))
        elif len(t.strip()) == 1 and t.strip().upper() in "ABCD":
            letters.append((i, t.strip().upper()))

    uniq2 = sorted({x[1] for x in letters})
    if len(uniq2) == 1 and letters:
        ii = min([x[0] for x in letters if x[1] == uniq2[0]])
        try:
            return uniq2[0], float(probs[ii]), "token_prefix"
        except Exception:
            return uniq2[0], None, "token_prefix"
    if len(uniq2) >= 2:
        return "", None, "token_multi"

    return "", None, "token_none"


def _normalize_token_words(tok: str) -> List[str]:
    s = (tok or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s.split() if s else []


def _parse_option_to_verb_noun_words(opt: str) -> Tuple[str, List[str]]:
    s = (opt or "").strip()
    if not s:
        return "", []
    if s.upper().startswith("YOU "):
        s = s[4:].strip()
    parts = s.split()
    if not parts:
        return "", []
    verb_raw = parts[0].strip()
    noun_raw = " ".join(parts[1:]).strip() if len(parts) >= 2 else ""

    v = verb_raw.lower()
    if "(" in v:
        v = v.split("(", 1)[0].strip()
    v = v.strip("_").replace("_", " ").strip()
    v_words = _WORD_RE.findall(v)
    verb = v_words[0] if v_words else ""

    n = noun_raw.lower()
    if "(" in n:
        n = n.split("(", 1)[0].strip()
    n = n.strip("_").replace("_", " ").strip()
    noun_words = _WORD_RE.findall(n)

    if noun_words and noun_words[0] in {"none", "null"}:
        return verb, ["none"]

    return verb, noun_words


def _token_word_index(gen_tokens: List[str], gen_probs: List[float]) -> Tuple[Dict[str, List[int]], List[List[str]]]:
    word2idx: Dict[str, List[int]] = {}
    token_words: List[List[str]] = []
    for i, t in enumerate(gen_tokens):
        ws = _normalize_token_words(str(t))
        token_words.append(ws)
        for w in ws:
            word2idx.setdefault(w, []).append(i)
    return word2idx, token_words


def _pick_first_prob_for_word(word: str, word2idx: Dict[str, List[int]], gen_probs: List[float]) -> Optional[float]:
    idxs = word2idx.get(word, [])
    if not idxs:
        return None
    i0 = idxs[0]
    if 0 <= i0 < len(gen_probs):
        try:
            return float(gen_probs[i0])
        except Exception:
            return None
    return None


def _infer_ans_by_token_verbnoun(
    gen_tokens: List[str],
    gen_probs: List[float],
    options4: List[str],
) -> Tuple[str, Optional[float], str]:
    if (not gen_tokens) or (not gen_probs) or len(gen_tokens) != len(gen_probs):
        return "", None, "token_verbnoun_none"
    if not options4:
        return "", None, "token_verbnoun_no_options"

    opt_vn: List[Tuple[str, List[str]]] = []
    verbs: List[str] = []
    nouns_join: List[str] = []
    for o in options4[:4]:
        v, nwords = _parse_option_to_verb_noun_words(o)
        opt_vn.append((v, nwords))
        verbs.append(v)
        nouns_join.append(" ".join(nwords) if nwords else "")

    has_dup = (len(set(verbs)) < len(verbs)) or (len(set(nouns_join)) < len(nouns_join))

    word2idx, _ = _token_word_index(gen_tokens, gen_probs)
    tok_wordset = set(word2idx.keys())

    hits_or: List[bool] = [False, False, False, False]
    hits_and: List[bool] = [False, False, False, False]

    for i in range(min(4, len(opt_vn))):
        v, nwords = opt_vn[i]
        v_hit = bool(v and (v in tok_wordset))
        if not nwords:
            n_hit = False
        else:
            n_hit = all(w in tok_wordset for w in nwords)
        hits_or[i] = bool(v_hit or n_hit)
        hits_and[i] = bool(v_hit and n_hit)

    if has_dup:
        cand = [i for i in range(4) if hits_and[i]]
        if len(cand) != 1:
            return "", None, "token_verbnoun_ambiguous"
        idx = cand[0]
        letter = "ABCD"[idx]
        v, nwords = opt_vn[idx]
        probs_used: List[float] = []
        pv = _pick_first_prob_for_word(v, word2idx, gen_probs) if v else None
        if pv is not None:
            probs_used.append(float(pv))
        for w in nwords:
            pw = _pick_first_prob_for_word(w, word2idx, gen_probs)
            if pw is not None:
                probs_used.append(float(pw))
        conf = (sum(probs_used) / len(probs_used)) if probs_used else None
        return letter, conf, "token_verbnoun_and"
    else:
        cand = [i for i in range(4) if hits_or[i]]
        if len(cand) != 1:
            return "", None, "token_verbnoun_ambiguous"
        idx = cand[0]
        letter = "ABCD"[idx]
        v, nwords = opt_vn[idx]
        probs_used: List[float] = []
        pv = _pick_first_prob_for_word(v, word2idx, gen_probs) if v else None
        if pv is not None:
            probs_used.append(float(pv))
        for w in nwords:
            pw = _pick_first_prob_for_word(w, word2idx, gen_probs)
            if pw is not None:
                probs_used.append(float(pw))
        conf = (sum(probs_used) / len(probs_used)) if probs_used else None
        return letter, conf, "token_verbnoun_or"


# -------------------------
# cand_conf helpers
# -------------------------
def _clamp01(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if v < 0.0:
        v = 0.0
    if v > 1.0:
        v = 1.0
    return float(v)


def _get_cand_conf_pcond_map(ps: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    Return p_cond dict from a cand_conf pred sample, if present.
    Expected path:
      ps["diagnostics"]["prompt_and_encoding_debug"]["cand_conf_probe"]["p_cond"]
    """
    try:
        d = ps.get("diagnostics", {}) if isinstance(ps.get("diagnostics"), dict) else {}
        ped = d.get("prompt_and_encoding_debug", {}) if isinstance(d.get("prompt_and_encoding_debug"), dict) else {}
        probe = ped.get("cand_conf_probe", {}) if isinstance(ped.get("cand_conf_probe"), dict) else {}
        pcond = probe.get("p_cond", None)
        if not isinstance(pcond, dict):
            return None
        out: Dict[str, float] = {}
        for k, v in pcond.items():
            kk = normalize_ans_letter(str(k))
            vv = _clamp01(v)
            if kk and vv is not None:
                out[kk] = float(vv)
        return out if out else None
    except Exception:
        return None


# -------------------------
# Core evaluation
# -------------------------
def _init_step_bucket() -> Dict[str, Any]:
    return {
        "samples_total": 0,
        "samples_valid": 0,
        "samples_invalid": 0,
        "correct": 0,

        "mapped_to_option_sources": 0,
        "correct_mapped": 0,
        "pred_source_counts": {},
        "wrong_source_counts": {},
        "gt_source_counts": {},

        "strong_choice_total": 0,
        "strong_choice_wrong": 0,
        "strong_prev_total": 0,
        "strong_future_total": 0,

        # token-prob confidence (ONLY token-derived; token_ans + token_verbnoun)
        "conf_cnt": 0,
        "conf_sum": 0.0,
        "conf_cnt_correct": 0,
        "conf_sum_correct": 0.0,
        "conf_cnt_wrong": 0,
        "conf_sum_wrong": 0.0,

        # token-prob confidence ONLY when inferred by OPTION LETTER (token_ans)
        "conf_cnt_letter": 0,
        "conf_sum_letter": 0.0,
        "conf_cnt_letter_correct": 0,
        "conf_sum_letter_correct": 0.0,
        "conf_cnt_letter_wrong": 0,
        "conf_sum_letter_wrong": 0.0,

        # cand_conf-specific confidence stats
        "cand_conf_cnt_valid": 0,
        "cand_conf_sum_valid": 0.0,
        "cand_conf_cnt_correct": 0,
        "cand_conf_sum_correct": 0.0,
        "cand_conf_cnt_wrong": 0,
        "cand_conf_sum_wrong": 0.0,
    }


def _inc_count(d: Dict[str, int], k: str, add: int = 1) -> None:
    d[k] = int(d.get(k, 0)) + int(add)


def _mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    try:
        return float(sum(xs) / len(xs))
    except Exception:
        return None


def _pstdev(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    if len(xs) == 1:
        return 0.0
    try:
        return float(statistics.pstdev(xs))
    except Exception:
        return None


def eval_ms_rtrv_cand() -> Dict[str, Any]:
    _ensure_dir(OUT_DIR)
    if OUT_DETAILS.exists():
        OUT_DETAILS.unlink(missing_ok=True)

    pred_files = build_pred_file_list(PRED_DIR)

    teacher_cache: Dict[str, Dict[str, Any]] = {}
    if OUT_CACHE.exists():
        try:
            teacher_cache = json.loads(OUT_CACHE.read_text(encoding="utf-8"))
            if not isinstance(teacher_cache, dict):
                teacher_cache = {}
        except Exception:
            teacher_cache = {}

    num_files_total = 0
    num_files_loaded = 0
    missing_queryset_path = 0
    missing_queryset_file = 0

    total_samples = 0
    valid_samples = 0
    invalid_samples = 0
    correct_total = 0

    teacher_calls = 0
    teacher_cache_hits = 0
    teacher_fail = 0

    step_stats: Dict[int, Dict[str, Any]] = {0: _init_step_bucket(), 1: _init_step_bucket(), 2: _init_step_bucket()}
    lag_values_seen: Dict[int, set] = {0: set(), 1: set(), 2: set()}

    group_track: Dict[str, Dict[int, Dict[str, Any]]] = {}

    _log(f"MS_RTRV({MODE.upper()}): start processing {len(pred_files)} pred files")

    queryset_cache: Dict[str, Dict[str, Any]] = {}

    for fi, pf in enumerate(pred_files, 1):
        num_files_total += 1

        if fi == 1 or (fi % PRED_LOG_EVERY_FILES == 0):
            _log(f"MS_RTRV: file {fi}/{len(pred_files)} -> {pf.name} (samples={total_samples}, correct={correct_total})")

        try:
            pred = json.loads(pf.read_text(encoding="utf-8"))
        except Exception as e:
            _log(f"MS_RTRV: pred json load failed: {pf.name} err={repr(e)}")
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
                _log(f"MS_RTRV: queryset load failed: {qp.name} err={repr(e)}")
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
                _log(f"MS_RTRV: samples processed={total_samples}, correct={correct_total}, valid={valid_samples}, invalid={invalid_samples}")

            p_idx = ps.get("idx", None)
            if not isinstance(p_idx, (int, float)):
                invalid_samples += 1
                continue
            idxi = int(p_idx)

            gt_s = qs_by_idx.get(idxi)
            if gt_s is None:
                invalid_samples += 1
                continue

            step_idx = gt_s.get("step_idx", None)
            lag_sec = gt_s.get("lag_sec", None)
            anchor_group_id = str(gt_s.get("anchor_group_id", "") or "").strip()

            if not isinstance(step_idx, (int, float)):
                step_idx = None
            else:
                step_idx = int(step_idx)

            if step_idx is None and isinstance(lag_sec, (int, float)):
                l = float(lag_sec)
                if abs(l - 8.0) < 1e-6:
                    step_idx = 0
                elif abs(l - 16.0) < 1e-6:
                    step_idx = 1
                elif abs(l - 24.0) < 1e-6:
                    step_idx = 2

            if step_idx not in {0, 1, 2}:
                invalid_samples += 1
                continue

            step_stats[step_idx]["samples_total"] += 1
            if isinstance(lag_sec, (int, float)):
                lag_values_seen[step_idx].add(float(lag_sec))

            mcq = gt_s.get("mcq", {}) if isinstance(gt_s.get("mcq"), dict) else {}
            gt_ans = normalize_ans_letter(str(mcq.get("answer", "") or ""))
            options = mcq.get("options", [])
            option_sources = mcq.get("option_sources", [])

            if not isinstance(options, list):
                options = []
            options4 = [str(x) for x in options[:4]]

            if not isinstance(option_sources, list):
                option_sources = []
            option_sources4 = [str(x) for x in option_sources[:4]] if option_sources else []

            resp_text = str(ps.get("response_text", "") or "")
            clean_text = str(ps.get("clean_response", "") or ps.get("clean", "") or resp_text or "")

            gen_tokens = ps.get("gen_tokens", None)
            gen_probs = ps.get("gen_token_probs", None)
            has_tokens = isinstance(gen_tokens, list) and isinstance(gen_probs, list) and gen_tokens and gen_probs and (len(gen_tokens) == len(gen_probs))

            pred_ans = ""
            pred_method = "none"
            conf_token: Optional[float] = None
            conf_source = "none"  # {"token_ans","token_verbnoun","clean_or_teacher","none"}
            conf_tokenprob_mode = "none"  # {"none","gen_token_prob","cand_conf_pcond"}

            used_teacher = False
            teacher_diag = None

            if has_tokens:
                a1, c1, m1 = _extract_single_ans_from_tokens(list(gen_tokens), list(gen_probs))
                if m1 == "token_multi":
                    pred_ans = ""
                    pred_method = "token_multi"
                elif a1:
                    pred_ans = a1
                    pred_method = m1
                    conf_token = c1
                    conf_source = "token_ans"
                    conf_tokenprob_mode = "gen_token_prob"
                else:
                    a2, c2, m2 = _infer_ans_by_token_verbnoun(list(gen_tokens), list(gen_probs), options4)
                    if a2:
                        pred_ans = a2
                        pred_method = m2
                        conf_token = c2
                        conf_source = "token_verbnoun"
                        conf_tokenprob_mode = "gen_token_prob"
                    else:
                        pred_method = m2

            if not pred_ans:
                a3 = normalize_ans_letter(parse_tag(clean_text, "ans") or "")
                if not a3:
                    a3 = _guess_ans_letter_from_clean(clean_text)
                if a3:
                    pred_ans = a3
                    if pred_method in {"none", "token_none", "token_verbnoun_none"}:
                        pred_method = "clean"
                    else:
                        pred_method = pred_method + "+clean"
                    conf_source = "clean_or_teacher"

            if not pred_ans and pred_method != "token_multi":
                if OPENROUTER_API_KEY and options4:
                    used_teacher = True
                    pred_method = "teacher"
                    _log(
                        "MS_RTRV: need teacher_map_ans because ANS unresolved | "
                        f"resp_snip={_snippet(resp_text, RESP_SNIPPET)}"
                    )
                    opt_join = "\n".join([f"{i}:{str(o)}" for i, o in enumerate(options4)])
                    h = _sha1(resp_text + "\n" + opt_join)

                    if h in teacher_cache and isinstance(teacher_cache[h], dict):
                        teacher_cache_hits += 1
                        pred_ans = normalize_ans_letter(str(teacher_cache[h].get("ans", "")))
                    else:
                        teacher_calls += 1
                        a4, diag4 = teacher_map_answer_to_option(resp_text, options4)
                        teacher_diag = diag4
                        if diag4.get("error"):
                            teacher_fail += 1
                        pred_ans = normalize_ans_letter(a4)
                        teacher_cache[h] = {"ans": pred_ans, "options": options4}

                    conf_source = "clean_or_teacher"
                else:
                    pred_ans = ""

            # cand_conf mode: override confidence with p_cond[ANS]
            if MODE == "cand_conf":
                if pred_ans and pred_ans in {"A", "B", "C", "D"}:
                    pcond_map = _get_cand_conf_pcond_map(ps)
                    if pcond_map is not None and pred_ans in pcond_map:
                        conf_token = float(pcond_map[pred_ans])
                        conf_source = "token_ans"
                        conf_tokenprob_mode = "cand_conf_pcond"

            is_valid = bool(pred_ans) and (pred_ans in {"A", "B", "C", "D"}) and (pred_method != "token_multi")
            if is_valid:
                valid_samples += 1
                step_stats[step_idx]["samples_valid"] += 1
            else:
                invalid_samples += 1
                step_stats[step_idx]["samples_invalid"] += 1

            hit = bool(is_valid and gt_ans and (pred_ans == gt_ans))
            if hit:
                correct_total += 1
                step_stats[step_idx]["correct"] += 1

            pred_source = None
            gt_source = None
            if is_valid and option_sources4 and len(option_sources4) >= 4:
                pi = _ans_to_idx(pred_ans)
                gi = _ans_to_idx(gt_ans)
                if 0 <= pi < len(option_sources4):
                    pred_source = str(option_sources4[pi])
                if 0 <= gi < len(option_sources4):
                    gt_source = str(option_sources4[gi])

            if pred_source is not None:
                step_stats[step_idx]["mapped_to_option_sources"] += 1
                _inc_count(step_stats[step_idx]["pred_source_counts"], pred_source, 1)
                if gt_source is not None:
                    _inc_count(step_stats[step_idx]["gt_source_counts"], gt_source, 1)
                if hit:
                    step_stats[step_idx]["correct_mapped"] += 1
                else:
                    _inc_count(step_stats[step_idx]["wrong_source_counts"], pred_source, 1)

                is_strong = (pred_source in STRONG_DISTURB_SOURCES)
                if is_strong:
                    step_stats[step_idx]["strong_choice_total"] += 1
                    if pred_source == "prev":
                        step_stats[step_idx]["strong_prev_total"] += 1
                    elif pred_source == "future":
                        step_stats[step_idx]["strong_future_total"] += 1
                    if not hit:
                        step_stats[step_idx]["strong_choice_wrong"] += 1

            # token-derived conf (or cand_conf override when MODE == cand_conf)
            if is_valid and conf_source in {"token_ans", "token_verbnoun"} and (conf_token is not None):
                step_stats[step_idx]["conf_cnt"] += 1
                step_stats[step_idx]["conf_sum"] += float(conf_token)
                if hit:
                    step_stats[step_idx]["conf_cnt_correct"] += 1
                    step_stats[step_idx]["conf_sum_correct"] += float(conf_token)
                else:
                    step_stats[step_idx]["conf_cnt_wrong"] += 1
                    step_stats[step_idx]["conf_sum_wrong"] += float(conf_token)

            # only when inferred by option letter (token_ans)
            if is_valid and conf_source == "token_ans" and (conf_token is not None):
                step_stats[step_idx]["conf_cnt_letter"] += 1
                step_stats[step_idx]["conf_sum_letter"] += float(conf_token)
                if hit:
                    step_stats[step_idx]["conf_cnt_letter_correct"] += 1
                    step_stats[step_idx]["conf_sum_letter_correct"] += float(conf_token)
                else:
                    step_stats[step_idx]["conf_cnt_letter_wrong"] += 1
                    step_stats[step_idx]["conf_sum_letter_wrong"] += float(conf_token)

            # cand_conf-specific confidence stats
            if MODE == "cand_conf" and is_valid and (conf_token is not None) and (conf_tokenprob_mode == "cand_conf_pcond"):
                step_stats[step_idx]["cand_conf_cnt_valid"] += 1
                step_stats[step_idx]["cand_conf_sum_valid"] += float(conf_token)
                if hit:
                    step_stats[step_idx]["cand_conf_cnt_correct"] += 1
                    step_stats[step_idx]["cand_conf_sum_correct"] += float(conf_token)
                else:
                    step_stats[step_idx]["cand_conf_cnt_wrong"] += 1
                    step_stats[step_idx]["cand_conf_sum_wrong"] += float(conf_token)

            if anchor_group_id:
                group_track.setdefault(anchor_group_id, {})
                group_track[anchor_group_id][step_idx] = {
                    "valid": bool(is_valid),
                    "correct": bool(hit),
                    "conf_token": float(conf_token) if (conf_source in {"token_ans", "token_verbnoun"} and conf_token is not None and is_valid) else None,
                }

            detail = {
                "file": pf.name,
                "queryset_path": qpath,
                "idx": idxi,
                "t_eval": float(ps.get("t_eval")) if isinstance(ps.get("t_eval"), (int, float)) else None,
                "step_idx": int(step_idx),
                "lag_sec": float(lag_sec) if isinstance(lag_sec, (int, float)) else None,
                "anchor_group_id": anchor_group_id or None,

                "gt_ans": gt_ans,
                "pred_ans": pred_ans if pred_ans else None,
                "pred_method": pred_method,
                "valid": bool(is_valid),
                "wrong": bool(is_valid and not hit),
                "pred_letter": pred_ans if pred_ans else None,
                "answer_letter": gt_ans if gt_ans else None,
                "hit": bool(hit),
                "options": options4,
                "option_sources": option_sources4 if option_sources4 else None,
                "mcq_source": "effective_queryset",
                "mcp_file": mcq.get("mcp_file", None),

                "pred_source": pred_source,
                "gt_source": gt_source,
                "is_disturb": bool(pred_source in STRONG_DISTURB_SOURCES) if pred_source else None,
                "is_strong_distractor": bool(pred_source in STRONG_DISTURB_SOURCES) if pred_source else None,

                "conf_tokenprob": conf_token if (conf_source in {"token_ans", "token_verbnoun"}) else None,
                "conf_source": conf_source,
                "conf_tokenprob_mode": conf_tokenprob_mode,
                "conf_tag_in_response": parse_conf_from_text(clean_text),

                "used_teacher": bool(used_teacher),
                "teacher_diag": teacher_diag,
                "resp_snippet": _snippet(resp_text, RESP_SNIPPET),
            }
            with open(OUT_DETAILS, "a", encoding="utf-8") as f:
                f.write(json.dumps(detail, ensure_ascii=False) + "\n")

    OUT_CACHE.write_text(json.dumps(teacher_cache, ensure_ascii=False, indent=2), encoding="utf-8")

    step_summary: Dict[str, Any] = {}
    for si in [0, 1, 2]:
        st = step_stats[si]
        valid = int(st["samples_valid"])
        total = int(st["samples_total"])
        correct = int(st["correct"])
        acc_valid = (correct / valid) if valid > 0 else 0.0

        mapped = int(st["mapped_to_option_sources"])
        correct_mapped = int(st["correct_mapped"])
        wrong_mapped = int(max(0, mapped - correct_mapped))

        strong_total = int(st["strong_choice_total"])
        strong_wrong = int(st["strong_choice_wrong"])
        strong_rate = (strong_total / mapped) if mapped > 0 else 0.0
        strong_rate_wrong = (strong_wrong / wrong_mapped) if wrong_mapped > 0 else 0.0
        strong_prev = int(st["strong_prev_total"])
        strong_future = int(st["strong_future_total"])
        strong_prev_rate = (strong_prev / mapped) if mapped > 0 else 0.0
        strong_future_rate = (strong_future / mapped) if mapped > 0 else 0.0

        conf_cnt = int(st["conf_cnt"])
        conf_mean = (float(st["conf_sum"]) / conf_cnt) if conf_cnt > 0 else None
        conf_cnt_c = int(st["conf_cnt_correct"])
        conf_mean_c = (float(st["conf_sum_correct"]) / conf_cnt_c) if conf_cnt_c > 0 else None
        conf_cnt_w = int(st["conf_cnt_wrong"])
        conf_mean_w = (float(st["conf_sum_wrong"]) / conf_cnt_w) if conf_cnt_w > 0 else None

        conf_cnt_l = int(st["conf_cnt_letter"])
        conf_mean_l = (float(st["conf_sum_letter"]) / conf_cnt_l) if conf_cnt_l > 0 else None
        conf_cnt_l_c = int(st["conf_cnt_letter_correct"])
        conf_mean_l_c = (float(st["conf_sum_letter_correct"]) / conf_cnt_l_c) if conf_cnt_l_c > 0 else None
        conf_cnt_l_w = int(st["conf_cnt_letter_wrong"])
        conf_mean_l_w = (float(st["conf_sum_letter_wrong"]) / conf_cnt_l_w) if conf_cnt_l_w > 0 else None

        cand_conf_cnt_v = int(st["cand_conf_cnt_valid"])
        cand_conf_mean_v = (float(st["cand_conf_sum_valid"]) / cand_conf_cnt_v) if cand_conf_cnt_v > 0 else None
        cand_conf_cnt_c = int(st["cand_conf_cnt_correct"])
        cand_conf_mean_c = (float(st["cand_conf_sum_correct"]) / cand_conf_cnt_c) if cand_conf_cnt_c > 0 else None
        cand_conf_cnt_w = int(st["cand_conf_cnt_wrong"])
        cand_conf_mean_w = (float(st["cand_conf_sum_wrong"]) / cand_conf_cnt_w) if cand_conf_cnt_w > 0 else None

        lag_list = sorted(list(lag_values_seen[si])) if lag_values_seen[si] else []

        source_summary = summarize_source_stats(
            total_count=total,
            valid_count=valid,
            invalid_count=int(st["samples_invalid"]),
            mapped_count=mapped,
            correct_mapped=correct_mapped,
            selected_source_counts=st["pred_source_counts"],
            wrong_source_counts=st["wrong_source_counts"],
            gt_source_counts=st["gt_source_counts"],
        )

        step_entry = {
            "step_idx": si,
            "lag_values_seen": lag_list,
            "samples_total": total,
            "samples_valid": valid,
            "samples_invalid": int(st["samples_invalid"]),
            "accuracy_over_valid": float(acc_valid),
            "correct": correct,

            "mapped_to_option_sources": mapped,
            "correct_mapped": correct_mapped,
            "wrong_mapped": wrong_mapped,
            "pred_source_counts": st["pred_source_counts"],
            "gt_source_counts": st["gt_source_counts"],

            "strong_distractor_sources": sorted(list(STRONG_DISTURB_SOURCES)),
            "strong_distractor_choice_rate_over_mapped": float(strong_rate),
            "strong_distractor_choice_rate_given_wrong_over_mapped": float(strong_rate_wrong),
            "strong_prev_choice_rate_over_mapped": float(strong_prev_rate),
            "strong_future_choice_rate_over_mapped": float(strong_future_rate),

            "tokenprob_conf_mean_over_valid_token_derived": conf_mean,
            "tokenprob_conf_count_over_valid_token_derived": conf_cnt,
            "tokenprob_conf_mean_over_correct_token_derived": conf_mean_c,
            "tokenprob_conf_count_over_correct_token_derived": conf_cnt_c,
            "tokenprob_conf_mean_over_wrong_token_derived": conf_mean_w,
            "tokenprob_conf_count_over_wrong_token_derived": conf_cnt_w,

            "tokenprob_conf_mean_over_valid_token_derived_letter_matched": conf_mean_l,
            "tokenprob_conf_count_over_valid_token_derived_letter_matched": conf_cnt_l,
            "tokenprob_conf_mean_over_correct_token_derived_letter_matched": conf_mean_l_c,
            "tokenprob_conf_count_over_correct_token_derived_letter_matched": conf_cnt_l_c,
            "tokenprob_conf_mean_over_wrong_token_derived_letter_matched": conf_mean_l_w,
            "tokenprob_conf_count_over_wrong_token_derived_letter_matched": conf_cnt_l_w,

            "cand_conf_mean_over_valid": cand_conf_mean_v,
            "cand_conf_count_over_valid": cand_conf_cnt_v,
            "cand_conf_mean_over_correct": cand_conf_mean_c,
            "cand_conf_count_over_correct": cand_conf_cnt_c,
            "cand_conf_mean_over_wrong": cand_conf_mean_w,
            "cand_conf_count_over_wrong": cand_conf_cnt_w,
        }
        for k, v in source_summary.items():
            step_entry.setdefault(k, v)
        step_summary[str(si)] = step_entry

    EPS_MONO = 1e-6
    groups_total = 0
    groups_all3_correct = 0
    groups_all3_correct_with_conf = 0
    mono_first_ok = 0
    mono_second_ok = 0
    mono_overall_ok = 0

    # collect betas on the same subset as monotonic denominator
    betas: List[float] = []
    betas_z: List[float] = []

    for gid, mp in group_track.items():
        if not isinstance(mp, dict):
            continue
        groups_total += 1

        if not all((i in mp) for i in [0, 1, 2]):
            continue

        if not (bool(mp[0].get("correct")) and bool(mp[1].get("correct")) and bool(mp[2].get("correct"))):
            continue
        groups_all3_correct += 1

        conf0 = mp[0].get("conf_token")
        conf1 = mp[1].get("conf_token")
        conf2 = mp[2].get("conf_token")
        if (conf0 is None) or (conf1 is None) or (conf2 is None):
            continue

        groups_all3_correct_with_conf += 1

        c0 = float(conf0)
        c1 = float(conf1)
        c2 = float(conf2)

        ok01 = bool(c0 + EPS_MONO >= c1)
        ok12 = bool(c1 + EPS_MONO >= c2)
        ok_all = bool(ok01 and ok12)

        if ok01:
            mono_first_ok += 1
        if ok12:
            mono_second_ok += 1
        if ok_all:
            mono_overall_ok += 1

        # linear trend slope beta (r=1,2,3)
        beta = (c2 - c0) / 2.0
        betas.append(float(beta))

        mean_c = (c0 + c1 + c2) / 3.0
        var_c = ((c0 - mean_c) ** 2 + (c1 - mean_c) ** 2 + (c2 - mean_c) ** 2) / 3.0
        std_c = math.sqrt(max(0.0, var_c))
        if std_c > 1e-12:
            z0 = (c0 - mean_c) / std_c
            z1 = (c1 - mean_c) / std_c
            z2 = (c2 - mean_c) / std_c
        else:
            z0, z1, z2 = 0.0, 0.0, 0.0
        beta_z = (z2 - z0) / 2.0
        betas_z.append(float(beta_z))

    denom = groups_all3_correct_with_conf
    monotonic = {
        "eps": EPS_MONO,
        "anchor_groups_total_seen": int(groups_total),

        "anchor_groups_all_3_steps_correct": int(groups_all3_correct),
        "anchor_groups_all_3_steps_correct_with_all_3_conf_token_derived": int(groups_all3_correct_with_conf),

        "first_segment_nonincreasing_count": int(mono_first_ok),
        "second_segment_nonincreasing_count": int(mono_second_ok),
        "overall_nonincreasing_count": int(mono_overall_ok),

        "first_segment_nonincreasing_fraction_over_all3_correct_with_conf": (float(mono_first_ok / denom) if denom > 0 else None),
        "second_segment_nonincreasing_fraction_over_all3_correct_with_conf": (float(mono_second_ok / denom) if denom > 0 else None),
        "overall_nonincreasing_fraction_over_all3_correct_with_conf": (float(mono_overall_ok / denom) if denom > 0 else None),
    }

    beta_mean = _mean(betas)
    beta_med = float(statistics.median(betas)) if betas else None
    beta_std = _pstdev(betas)
    beta_neg_frac = (float(sum(1 for b in betas if b < 0.0) / len(betas)) if betas else None)

    beta_z_mean = _mean(betas_z)
    beta_z_med = float(statistics.median(betas_z)) if betas_z else None
    beta_z_std = _pstdev(betas_z)
    beta_z_neg_frac = (float(sum(1 for b in betas_z if b < 0.0) / len(betas_z)) if betas_z else None)

    linear_trend = {
        "anchor_groups_used": int(len(betas)),
        "beta_mean": beta_mean,
        "beta_median": beta_med,
        "beta_std": beta_std,
        "beta_neg_fraction": beta_neg_frac,

        "beta_z_mean": beta_z_mean,
        "beta_z_median": beta_z_med,
        "beta_z_std": beta_z_std,
        "beta_z_neg_fraction": beta_z_neg_frac,
    }

    all_pred_source_counts: Dict[str, int] = {}
    all_wrong_source_counts: Dict[str, int] = {}
    all_gt_source_counts: Dict[str, int] = {}
    for st in step_stats.values():
        for k, v in st["pred_source_counts"].items():
            _inc_count(all_pred_source_counts, k, int(v))
        for k, v in st["wrong_source_counts"].items():
            _inc_count(all_wrong_source_counts, k, int(v))
        for k, v in st["gt_source_counts"].items():
            _inc_count(all_gt_source_counts, k, int(v))

    overall_source_summary = summarize_source_stats(
        total_count=int(total_samples),
        valid_count=int(valid_samples),
        invalid_count=int(invalid_samples),
        mapped_count=sum(int(st["mapped_to_option_sources"]) for st in step_stats.values()),
        correct_mapped=sum(int(st["correct_mapped"]) for st in step_stats.values()),
        selected_source_counts=all_pred_source_counts,
        wrong_source_counts=all_wrong_source_counts,
        gt_source_counts=all_gt_source_counts,
    )

    summary = {
        "model_name": MODEL_NAME,
        "task": f"ms_rtrv_{MODE}",
        "mode": MODE,
        "confidence_mode": ("cand_conf_pcond" if MODE == "cand_conf" else "gen_token_prob_or_verbnoun"),
        "pred_dir": str(PRED_DIR),
        "out_dir": str(OUT_DIR),

        "num_pred_files_total": int(num_files_total),
        "num_pred_files_loaded": int(num_files_loaded),
        "missing_queryset_path": int(missing_queryset_path),
        "missing_queryset_file": int(missing_queryset_file),

        "samples_total_seen": int(total_samples),
        "samples_valid": int(valid_samples),
        "samples_invalid": int(invalid_samples),
        "accuracy_over_valid": float(correct_total / valid_samples) if valid_samples > 0 else 0.0,
        "correct": int(correct_total),

        "per_step": step_summary,
        "monotonic": monotonic,
        "linear_trend_slope": linear_trend,

        "teacher": {
            "enabled": bool(bool(OPENROUTER_API_KEY)),
            "openrouter_model": OPENROUTER_MODEL,
            "teacher_calls": int(teacher_calls),
            "teacher_cache_hits": int(teacher_cache_hits),
            "teacher_fail": int(teacher_fail),
        },

        "timestamp_unix": time.time(),
    }
    for k, v in overall_source_summary.items():
        summary.setdefault(k, v)

    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _apply_mode_paths(
    mode: str,
    runs_root: Optional[Path] = None,
    pred_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> None:
    global MODE, PRED_DIR, OUT_DIR, OUT_SUMMARY, OUT_DETAILS, OUT_CACHE

    MODE = (mode or "cand").strip().lower()
    if MODE not in {"cand", "cand_conf"}:
        MODE = "cand"

    if pred_dir is not None:
        PRED_DIR = pred_dir.expanduser()
    elif runs_root is not None:
        PRED_DIR = runs_root.expanduser() / MODEL_NAME / "ms_rtrv" / MODE
    elif MODE == "cand_conf":
        PRED_DIR = Path.home() / "benchmark_val" / "testllm" / MODEL_NAME / "ms_rtrv" / "cand_conf"
    else:
        PRED_DIR = Path.home() / "benchmark_val" / "testllm" / MODEL_NAME / "ms_rtrv" / "cand"

    if out_dir is not None:
        OUT_DIR = out_dir.expanduser()
    elif MODE == "cand_conf":
        OUT_DIR = Path.home() / "benchmark_val" / "score" / MODEL_NAME / "ms_rtrv_cand_conf"
    else:
        OUT_DIR = Path.home() / "benchmark_val" / "score" / MODEL_NAME / "ms_rtrv_cand"

    OUT_SUMMARY = OUT_DIR / "summary.json"
    OUT_DETAILS = OUT_DIR / "details.jsonl"
    OUT_CACHE = OUT_DIR / "teacher_cache.json"


def main() -> None:
    global MODEL_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="cand",
        choices=["cand", "cand_conf"],
        help="Evaluation mode: cand (default) or cand_conf",
    )
    parser.add_argument("--flavor", choices=["cand", "cand_conf"], default=None, help="Alias for --mode.")
    parser.add_argument("--model_name", default=None, help="Override MODEL_NAME without changing the environment.")
    parser.add_argument("--runs_root", type=Path, default=None, help="Infer pred_dir as <runs_root>/<model_name>/ms_rtrv/<flavor>.")
    parser.add_argument("--pred_dir", type=Path, default=None, help="Override prediction directory.")
    parser.add_argument("--out_dir", type=Path, default=None, help="Override output directory.")
    args = parser.parse_args()
    if args.model_name:
        MODEL_NAME = str(args.model_name).strip()
    _apply_mode_paths(args.flavor or args.mode, runs_root=args.runs_root, pred_dir=args.pred_dir, out_dir=args.out_dir)

    if not PRED_DIR.exists():
        raise FileNotFoundError(f"PRED_DIR not found: {PRED_DIR}")
    _ensure_dir(OUT_DIR)

    print("\n================ MS_RTRV EVAL ================\n")
    print(f"MODEL_NAME   : {MODEL_NAME}")
    print(f"MODE         : {MODE}")
    print(f"PRED_DIR     : {PRED_DIR}")
    print(f"OUT_DIR      : {OUT_DIR}")
    print(f"TEACHER_EN   : {bool(OPENROUTER_API_KEY)} (OPENROUTER_MODEL={OPENROUTER_MODEL})")
    print("")

    s = eval_ms_rtrv_cand()

    print(f"\n[WROTE] {OUT_SUMMARY}")
    print(f"[WROTE] {OUT_DETAILS}")
    print(f"[WROTE] {OUT_CACHE}")
    print("")

    print("---- Overall ----")
    print(f"Valid samples : {s['samples_valid']} / {s['samples_total_seen']} (invalid={s['samples_invalid']})")
    print(f"Acc(valid)    : {s['accuracy_over_valid']:.4f}  (correct={s['correct']})")
    print("")
    print("---- Source stats overall (valid-only) ----")
    for line in source_stats_report_lines(s):
        print(line)
    print("")

    print("---- Per-step ----")
    for si in ["0", "1", "2"]:
        ps = s["per_step"][si]
        lag_seen = ps.get("lag_values_seen")
        lag_show = lag_seen[0] if isinstance(lag_seen, list) and len(lag_seen) == 1 else lag_seen

        base_msg = (
            f"step={si} lag={lag_show} | "
            f"acc={ps['accuracy_over_valid']:.4f} ({ps['correct']}/{ps['samples_valid']}) "
            f"valid={ps['samples_valid']} invalid={ps['samples_invalid']} "
            f"strong={ps['strong_distractor_choice_rate_over_mapped']:.4f} (mapped={ps['mapped_to_option_sources']}) "
            f"conf_all={ps['tokenprob_conf_mean_over_valid_token_derived']} (n={ps['tokenprob_conf_count_over_valid_token_derived']}) "
            f"conf_ok={ps['tokenprob_conf_mean_over_correct_token_derived']} (n={ps['tokenprob_conf_count_over_correct_token_derived']}) "
            f"conf_wrong={ps['tokenprob_conf_mean_over_wrong_token_derived']} (n={ps['tokenprob_conf_count_over_wrong_token_derived']}) "
            f"| conf_all_letter={ps['tokenprob_conf_mean_over_valid_token_derived_letter_matched']} (n={ps['tokenprob_conf_count_over_valid_token_derived_letter_matched']}) "
            f"conf_ok_letter={ps['tokenprob_conf_mean_over_correct_token_derived_letter_matched']} (n={ps['tokenprob_conf_count_over_correct_token_derived_letter_matched']}) "
            f"conf_wrong_letter={ps['tokenprob_conf_mean_over_wrong_token_derived_letter_matched']} (n={ps['tokenprob_conf_count_over_wrong_token_derived_letter_matched']})"
        )

        if MODE == "cand_conf":
            base_msg += (
                f" | cand_conf_valid={ps['cand_conf_mean_over_valid']} (n={ps['cand_conf_count_over_valid']}) "
                f"cand_conf_ok={ps['cand_conf_mean_over_correct']} (n={ps['cand_conf_count_over_correct']}) "
                f"cand_conf_wrong={ps['cand_conf_mean_over_wrong']} (n={ps['cand_conf_count_over_wrong']})"
            )

        print(base_msg)

    print("\n---- Monotonic diagnostics (ONLY groups with 3/3 correct) ----")
    mono = s["monotonic"]
    print(f"Groups total seen                                   : {mono['anchor_groups_total_seen']}")
    print(f"Groups all 3 steps correct                           : {mono['anchor_groups_all_3_steps_correct']}")
    print(f"Groups all 3 steps correct + all 3 conf available    : {mono['anchor_groups_all_3_steps_correct_with_all_3_conf_token_derived']}")
    print(f"First segment non-increasing fraction (c0>=c1)       : {mono['first_segment_nonincreasing_fraction_over_all3_correct_with_conf']}")
    print(f"Second segment non-increasing fraction (c1>=c2)      : {mono['second_segment_nonincreasing_fraction_over_all3_correct_with_conf']}")
    print(f"Overall non-increasing fraction (c0>=c1>=c2)         : {mono['overall_nonincreasing_fraction_over_all3_correct_with_conf']}")

    lt = s.get("linear_trend_slope", {}) if isinstance(s.get("linear_trend_slope"), dict) else {}
    if lt:
        print("\n---- Linear trend slope beta (ONLY groups with 3/3 correct + 3/3 conf) ----")
        print(f"Groups used                                          : {lt.get('anchor_groups_used')}")
        print(
            f"beta_mean / median / std                             : "
            f"{lt.get('beta_mean')} / {lt.get('beta_median')} / {lt.get('beta_std')}"
        )
        print(f"beta_neg_fraction                                    : {lt.get('beta_neg_fraction')}")
        print(
            f"beta_z_mean / median / std                           : "
            f"{lt.get('beta_z_mean')} / {lt.get('beta_z_median')} / {lt.get('beta_z_std')}"
        )
        print(f"beta_z_neg_fraction                                  : {lt.get('beta_z_neg_fraction')}")

    print("\n[DONE]\n")


if __name__ == "__main__":
    main()
