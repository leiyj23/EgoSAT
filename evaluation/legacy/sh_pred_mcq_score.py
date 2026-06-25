#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sh_pred_eval.py

One-click evaluation for sh_pred (short-horizon prediction) in CAND_FULL mode.

Inputs:
  ~/benchmark_val/testllm/<MODEL_NAME>/sh_pred/cand_full/*.json

Each pred file should contain either:
  - source_queryset: "/.../__derived_sh_pred__cand_full.json"
  - OR video_metadata.queryset_path: ".../__derived_sh_pred__cand_full.json"
We load that queryset and match samples by idx, then compare pred vs queryset.samples[idx].mcq.answer.

Answer parsing (aligned with ms_rtrv_eval spirit):
  1) token-based:
      a) letter inside <ANS>...</ANS> span in gen_tokens
      b) letter from prefix tokens before first "<"
      c) fallback token verb/noun match against options
      - if token_multi => invalid (do NOT fallback to clean/teacher)
  2) clean-text:
      a) parse <ANS>...</ANS> from clean_response/response_text
      b) guess unique A/B/C/D from clean text (after stripping <...>)
  3) optional Teacher(OpenRouter) mapping, only if enabled

Outputs:
  ~/benchmark_val/score/<MODEL_NAME>/sh_pred_cand_full/{summary.json, details.jsonl, teacher_cache.json}

Run:
  MODEL_NAME=gemini_pro python3 sh_pred_eval.py
  MODEL_NAME=timechat_online_7b python3 sh_pred_eval.py

Optional Teacher(OpenRouter):
  read -r -s OPENROUTER_API_KEY
  export OPENROUTER_API_KEY
  export OPENROUTER_MODEL=google/gemini-2.5-pro
"""

from __future__ import annotations

import os
import re
import json
import time
import hashlib
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import requests

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
# Hard-coded paths / knobs
# -------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini_pro").strip() or "gemini_pro"

PRED_DIR = Path.home() / "benchmark_val" / "testllm" / MODEL_NAME / "sh_pred" / "cand_full"
OUT_DIR = Path.home() / "benchmark_val" / "score" / MODEL_NAME / "sh_pred_cand_full"

OUT_SUMMARY = OUT_DIR / "summary.json"
OUT_DETAILS = OUT_DIR / "details.jsonl"
OUT_CACHE = OUT_DIR / "teacher_cache.json"

# Teacher (OpenRouter) - optional
ENABLE_TEACHER_FALLBACK = os.environ.get("EGOSAT_ENABLE_LEGACY_TEACHER_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip() if ENABLE_TEACHER_FALLBACK else ""
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-pro").strip()
OPENROUTER_URL = os.environ.get("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions").strip()
OPENROUTER_TIMEOUT_SEC = int(float(os.environ.get("OPENROUTER_TIMEOUT_SEC", "60")))
OPENROUTER_RETRY = int(float(os.environ.get("OPENROUTER_RETRY", "2")))

# progress logging knobs
PRED_LOG_EVERY_FILES = 10
SAMPLE_LOG_EVERY = 800

RESP_SNIPPET = 240
TEACHER_PRINT_OK_SNIPPET = 220
TEACHER_PRINT_ERR_SNIPPET = 600


# -------------------------
# Logging
# -------------------------
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


def normalize_ans_letter(x: str) -> str:
    s = (x or "").strip().upper()
    s = re.sub(r"[^A-D]", "", s)  # keep only A-D chars
    return s[0] if s and s[0] in "ABCD" else ""


def parse_tag(text: str, which: str) -> Optional[str]:
    rx = _TAG_RE.get(which)
    if not rx:
        return None
    m = rx.search(text or "")
    if not m:
        return None
    return (m.group(1) or "").strip()


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
      - JSONL with exactly ONE JSON object
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
# Pred file listing
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
        if "__pred" in name or name.endswith("_pred.json"):
            out.append(p)
        else:
            # keep permissive: sh_pred dir may contain only pred-like files
            out.append(p)
    _log(f"Pred list: {pred_dir} -> {len(out)} files (skipped_manifest={skipped})")
    return out


# -------------------------
# Find queryset path in pred (robust)
# -------------------------
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
# Token-prob based answer extraction (same spirit as ms_rtrv_eval)
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
    Returns (ans_letter_or_empty, conf_or_None, method_str)
      method_str in {"token_ans_span","token_prefix","token_multi","token_none"}
    """
    if (not gen_tokens) or (not gen_probs) or len(gen_tokens) != len(gen_probs):
        return "", None, "token_none"

    toks = [str(x) for x in gen_tokens]
    probs = gen_probs

    # (1) inside <ANS>...</ANS>
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

    # (2) prefix before first "<"
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


def _token_word_index(gen_tokens: List[str]) -> Dict[str, List[int]]:
    word2idx: Dict[str, List[int]] = {}
    for i, t in enumerate(gen_tokens):
        ws = _normalize_token_words(str(t))
        for w in ws:
            word2idx.setdefault(w, []).append(i)
    return word2idx


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
    """
    Infer unique A/B/C/D by matching option verb/noun words against generated token words.
    Returns (letter, conf, method) where method in {"token_verbnoun_and","token_verbnoun_or", "..._ambiguous", "..._none"}.
    """
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

    word2idx = _token_word_index(gen_tokens)
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
# Core evaluation
# -------------------------
def _init_bucket() -> Dict[str, Any]:
    return {
        "samples_total": 0,
        "samples_valid": 0,
        "samples_invalid": 0,
        "correct": 0,
    }


def eval_sh_pred_cand_full() -> Dict[str, Any]:
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
    queryset_load_fail = 0

    total_samples = 0
    valid_samples = 0
    invalid_samples = 0
    correct_total = 0

    teacher_calls = 0
    teacher_cache_hits = 0
    teacher_fail = 0

    buckets: Dict[str, Dict[str, Any]] = {}
    for st in ["ALL", "NN", "NP", "PN", "PP", "UNK"]:
        buckets[st] = _init_bucket()
    source_stats_by_state: Dict[str, Dict[str, Any]] = {st: init_source_stats() for st in ["ALL", "NN", "NP", "PN", "PP", "UNK"]}

    queryset_cache: Dict[str, Dict[str, Any]] = {}

    _log(f"SH_PRED(CAND_FULL): start processing {len(pred_files)} pred files")

    for fi, pf in enumerate(pred_files, 1):
        num_files_total += 1

        if fi == 1 or (fi % PRED_LOG_EVERY_FILES == 0):
            _log(f"SH_PRED: file {fi}/{len(pred_files)} -> {pf.name} (samples={total_samples}, correct={correct_total})")

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
                _log(f"SH_PRED: samples processed={total_samples}, correct={correct_total}, valid={valid_samples}, invalid={invalid_samples}")

            p_idx = ps.get("idx", None)
            if not isinstance(p_idx, (int, float)):
                invalid_samples += 1
                buckets["ALL"]["samples_total"] += 1
                buckets["ALL"]["samples_invalid"] += 1
                continue
            idxi = int(p_idx)

            gt_s = qs_by_idx.get(idxi)
            if gt_s is None:
                invalid_samples += 1
                buckets["ALL"]["samples_total"] += 1
                buckets["ALL"]["samples_invalid"] += 1
                continue

            state = str(gt_s.get("state", "") or "").strip().upper()
            if state not in {"NN", "NP", "PN", "PP"}:
                state = "UNK"

            buckets["ALL"]["samples_total"] += 1
            buckets[state]["samples_total"] += 1

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

            resp_text = str(ps.get("response_text", "") or "")
            clean_text = str(ps.get("clean_response", "") or ps.get("clean", "") or resp_text or "")

            gen_tokens = ps.get("gen_tokens", None)
            gen_probs = ps.get("gen_token_probs", None)
            has_tokens = isinstance(gen_tokens, list) and isinstance(gen_probs, list) and gen_tokens and gen_probs and (len(gen_tokens) == len(gen_probs))

            pred_ans = ""
            pred_method = "none"
            conf_token: Optional[float] = None
            conf_source = "none"  # {"token_ans","token_verbnoun","clean_or_teacher","none"}

            used_teacher = False
            teacher_diag = None

            # ---------- (1) token-based ----------
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
                else:
                    a2, c2, m2 = _infer_ans_by_token_verbnoun(list(gen_tokens), list(gen_probs), options4)
                    if a2:
                        pred_ans = a2
                        pred_method = m2
                        conf_token = c2
                        conf_source = "token_verbnoun"
                    else:
                        pred_method = m2

            # ---------- (2) clean-text ----------
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

            # ---------- (3) teacher fallback (optional) ----------
            if not pred_ans and pred_method != "token_multi":
                if OPENROUTER_API_KEY and options4:
                    used_teacher = True
                    pred_method = "teacher"
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

            is_valid = bool(pred_ans) and (pred_ans in {"A", "B", "C", "D"}) and (pred_method != "token_multi")
            if is_valid:
                valid_samples += 1
                buckets["ALL"]["samples_valid"] += 1
                buckets[state]["samples_valid"] += 1
            else:
                invalid_samples += 1
                buckets["ALL"]["samples_invalid"] += 1
                buckets[state]["samples_invalid"] += 1

            hit = bool(is_valid and gt_ans and (pred_ans == gt_ans))
            if hit:
                correct_total += 1
                buckets["ALL"]["correct"] += 1
                buckets[state]["correct"] += 1

            source_info_all = update_source_stats(
                source_stats_by_state["ALL"],
                pred_ans,
                gt_ans,
                answer_idx,
                option_sources,
                bool(is_valid),
            )
            source_info_state = update_source_stats(
                source_stats_by_state[state],
                pred_ans,
                gt_ans,
                answer_idx,
                option_sources,
                bool(is_valid),
            )
            source_info = source_info_state or source_info_all

            detail = {
                "file": pf.name,
                "queryset_path": qpath,
                "idx": idxi,
                "state": state,

                "gt_ans": gt_ans,
                "pred_ans": pred_ans if pred_ans else None,
                "answer_letter": gt_ans if gt_ans else None,
                "pred_letter": pred_ans if pred_ans else None,
                "pred_method": pred_method,
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

                "conf_tokenprob": conf_token if (conf_source in {"token_ans", "token_verbnoun"}) else None,
                "conf_source": conf_source,

                "used_teacher": bool(used_teacher),
                "teacher_diag": teacher_diag,
                "resp_snippet": _snippet(resp_text, RESP_SNIPPET),
            }
            with open(OUT_DETAILS, "a", encoding="utf-8") as f:
                f.write(json.dumps(detail, ensure_ascii=False) + "\n")

    OUT_CACHE.write_text(json.dumps(teacher_cache, ensure_ascii=False, indent=2), encoding="utf-8")

    def _acc(correct: int, denom: int) -> float:
        return float(correct / denom) if denom > 0 else 0.0

    def _bucket_summary(name: str, b: Dict[str, Any]) -> Dict[str, Any]:
        total = int(b["samples_total"])
        valid = int(b["samples_valid"])
        invalid = int(b["samples_invalid"])
        correct = int(b["correct"])
        out = {
            "state": name,
            "samples_total": total,
            "samples_valid": valid,
            "samples_invalid": invalid,
            "correct": correct,
            "accuracy_over_total": _acc(correct, total),
            "accuracy_over_valid": _acc(correct, valid),
        }
        st = source_stats_by_state.get(name, init_source_stats())
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

    per_state = {"ALL": _bucket_summary("ALL", buckets["ALL"])}
    for st in ["NN", "NP", "PN", "PP", "UNK"]:
        per_state[st] = _bucket_summary(st, buckets[st])

    summary = {
        "model_name": MODEL_NAME,
        "task": "sh_pred_cand_full",
        "pred_dir": str(PRED_DIR),
        "out_dir": str(OUT_DIR),

        "num_pred_files_total": int(num_files_total),
        "num_pred_files_loaded": int(num_files_loaded),
        "missing_queryset_path": int(missing_queryset_path),
        "missing_queryset_file": int(missing_queryset_file),
        "queryset_load_fail": int(queryset_load_fail),

        "samples_total_seen": int(buckets["ALL"]["samples_total"]),
        "samples_valid": int(buckets["ALL"]["samples_valid"]),
        "samples_invalid": int(buckets["ALL"]["samples_invalid"]),
        "correct": int(buckets["ALL"]["correct"]),
        "accuracy_over_total": _acc(int(buckets["ALL"]["correct"]), int(buckets["ALL"]["samples_total"])),
        "accuracy_over_valid": _acc(int(buckets["ALL"]["correct"]), int(buckets["ALL"]["samples_valid"])),

        "per_state": per_state,

        "teacher": {
            "enabled": bool(bool(OPENROUTER_API_KEY)),
            "openrouter_model": OPENROUTER_MODEL,
            "teacher_calls": int(teacher_calls),
            "teacher_cache_hits": int(teacher_cache_hits),
            "teacher_fail": int(teacher_fail),
        },

        "timestamp_unix": time.time(),
    }
    for k, v in per_state["ALL"].items():
        if k != "state":
            summary.setdefault(k, v)

    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    global MODEL_NAME, PRED_DIR, OUT_DIR, OUT_SUMMARY, OUT_DETAILS, OUT_CACHE

    parser = argparse.ArgumentParser(description="Evaluate sh_pred cand_full and source statistics.")
    parser.add_argument("--model_name", default=None, help="Override MODEL_NAME without changing the environment.")
    parser.add_argument("--runs_root", type=Path, default=None, help="Infer pred_dir as <runs_root>/<model_name>/sh_pred/<flavor>.")
    parser.add_argument("--pred_dir", type=Path, default=None, help="Override prediction directory.")
    parser.add_argument("--out_dir", type=Path, default=None, help="Override output directory.")
    parser.add_argument("--flavor", default="cand_full", help="Prediction flavor for --runs_root inference; default: cand_full.")
    args = parser.parse_args()

    if args.model_name:
        MODEL_NAME = str(args.model_name).strip()
    flavor = str(args.flavor or "cand_full").strip()
    if args.pred_dir is not None:
        PRED_DIR = args.pred_dir.expanduser()
    elif args.runs_root is not None:
        PRED_DIR = args.runs_root.expanduser() / MODEL_NAME / "sh_pred" / flavor
    else:
        PRED_DIR = Path.home() / "benchmark_val" / "testllm" / MODEL_NAME / "sh_pred" / "cand_full"

    if args.out_dir is not None:
        OUT_DIR = args.out_dir.expanduser()
    else:
        OUT_DIR = Path.home() / "benchmark_val" / "score" / MODEL_NAME / "sh_pred_cand_full"

    OUT_SUMMARY = OUT_DIR / "summary.json"
    OUT_DETAILS = OUT_DIR / "details.jsonl"
    OUT_CACHE = OUT_DIR / "teacher_cache.json"

    if not PRED_DIR.exists():
        raise FileNotFoundError(f"PRED_DIR not found: {PRED_DIR}")
    _ensure_dir(OUT_DIR)

    print("\n================ SH_PRED CAND_FULL EVAL ================\n")
    print(f"MODEL_NAME   : {MODEL_NAME}")
    print(f"PRED_DIR     : {PRED_DIR}")
    print(f"OUT_DIR      : {OUT_DIR}")
    print(f"TEACHER_EN   : {bool(OPENROUTER_API_KEY)} (OPENROUTER_MODEL={OPENROUTER_MODEL})")
    print("")

    s = eval_sh_pred_cand_full()

    print(f"\n[WROTE] {OUT_SUMMARY}")
    print(f"[WROTE] {OUT_DETAILS}")
    print(f"[WROTE] {OUT_CACHE}")
    print("")

    print("---- Overall ----")
    print(f"Total samples : {s['samples_total_seen']} (valid={s['samples_valid']} invalid={s['samples_invalid']})")
    print(f"Acc(total)    : {s['accuracy_over_total']:.4f}  (correct={s['correct']})")
    print(f"Acc(valid)    : {s['accuracy_over_valid']:.4f}  (correct={s['correct']})")
    print("")
    print("---- Source stats overall (valid-only) ----")
    for line in source_stats_report_lines(s):
        print(line)
    print("")

    print("---- Per-state (NN/NP/PN/PP) ----")
    for st in ["NN", "NP", "PN", "PP", "UNK"]:
        ps = s["per_state"][st]
        print(
            f"state={st} | "
            f"acc_total={ps['accuracy_over_total']:.4f} "
            f"acc_valid={ps['accuracy_over_valid']:.4f} "
            f"({ps['correct']}/{ps['samples_total']}, valid={ps['samples_valid']}, invalid={ps['samples_invalid']})"
        )

    print("\n[DONE]\n")


if __name__ == "__main__":
    main()
