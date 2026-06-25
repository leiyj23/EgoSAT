#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sh_rtrv_eval.py

One-click evaluation for short-horizon retrieval (sh_rtrv) in TWO modes:
  - open  : parse GT narration -> (verb candidates, noun synonym-set) and match pred <VERB>/<NOUN> (teacher fallback)
  - cand  : read MCQ shuffled file -> match pred <ANS> with GT option (teacher fallback when <ANS> missing)

[Progress logging patch + teacher error visibility patch]
- Adds timestamped logs so you can see:
  * whether it is scanning MCQ_DIR (index building)
  * which pred file it is currently processing
  * which mcq file it resolved for that pred file
  * whether it's about to call OpenRouter teacher (network wait)
- NEW: Print OpenRouter HTTP status / error snippet when teacher call fails,
       and print teacher output snippet when it succeeds.
- NEW: When cand triggers teacher, print why (<ANS> missing/invalid) and a snippet of response_text.

No scoring logic changed.

[FIX]
- build_pred_file_list() now STRICTLY keeps ONLY prediction files (*__pred.json / *_pred.json / contains "__pred")
  and skips derived/queryset jsons like "*__derived_sh_rtrv__cand.json".

[NEW - strong distractor report]
- For CAND mode, report probability the model chooses "strong distractors":
    option_sources in {"prev","future"}.
  We compute this on samples where we can map pred <ANS> to an option_sources entry.
  We also report:
    - breakdown prev vs future
    - rate given wrong
    - distribution of pred_source / gt_source
    - mean parsed <CONF> for strong vs non-strong (if <CONF> exists)

[NEW - robust local ANS recovery before teacher]
(1) If <ANS>A</ANS> not found, try to recover a single A/B/C/D from clean_response,
    but exclude any letters that come from angle-bracket parts <...>.
    If multiple distinct letters are present, we do NOT guess.
(2) If still not found, try to match verb+noun of exactly one option inside clean_response.
    If signals from multiple options appear -> treat as multi-answer invalid (no teacher).
(3) Otherwise, fallback to teacher. Teacher sys_prompt updated so:
    - return "" ONLY when extremely uncertain / no usable signal
    - otherwise choose A/B/C/D
    - DO NOT default to A
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
        build_effective_mcq_by_idx,
        finalize_source_stats,
        init_source_stats,
        resolve_queryset_path_from_pred,
        source_stats_report_lines,
        update_source_stats,
    )
except ImportError:
    from mcq_source_stats import (
        build_effective_mcq_by_idx,
        finalize_source_stats,
        init_source_stats,
        resolve_queryset_path_from_pred,
        source_stats_report_lines,
        update_source_stats,
    )


# -------------------------
# Hard-coded paths / knobs
# -------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini_pro").strip() or "gemini_pro"

# NEW: control which evaluation(s) to run: open|cand|both(all)
SH_RTRV_EVAL_MODE = os.environ.get("SH_RTRV_EVAL_MODE", "both").strip().lower()

# Prediction directories
PRED_OPEN_DIR = Path.home() / "benchmark_val" / "testllm" / MODEL_NAME / "sh_rtrv" / "open"
PRED_CAND_DIR = Path.home() / "benchmark_val" / "testllm" / MODEL_NAME / "sh_rtrv" / "cand"

# GT + taxonomy
GT_FULL_DIR = Path(os.environ.get("EGOSAT_GT_DIR", ""))
TAXONOMY_JSON = Path(os.environ.get("EGOSAT_TAXONOMY_JSON", ""))

# MCQ shuffled (candidate mode). Defaults preserve historical behavior; env/CLI
# can point scoring at a second frozen shuffle root for rebuttal runs.
DEFAULT_MCQ_DIR = Path(os.environ.get("EGOSAT_MCQ_DIR", ""))
MCQ_DIR = Path(os.environ.get("SH_RTRV_MCQ_DIR", str(DEFAULT_MCQ_DIR))).expanduser()
MCQ_DIR_EXPLICIT = bool(os.environ.get("SH_RTRV_MCQ_DIR", "").strip())
STRICT_MCQ_PATH = os.environ.get("STRICT_MCQ_PATH", "0").strip().lower() in {"1", "true", "yes", "y", "t"}

# Output directories
OUT_OPEN_DIR = Path.home() / "benchmark_val" / "score" / MODEL_NAME / "sh_rtrv_open"
OUT_CAND_DIR = Path.home() / "benchmark_val" / "score" / MODEL_NAME / "sh_rtrv_cand"

OUT_OPEN_SUMMARY = OUT_OPEN_DIR / "summary.json"
OUT_OPEN_DETAILS = OUT_OPEN_DIR / "details.jsonl"
OUT_OPEN_CACHE = OUT_OPEN_DIR / "teacher_cache.json"

OUT_CAND_SUMMARY = OUT_CAND_DIR / "summary.json"
OUT_CAND_DETAILS = OUT_CAND_DIR / "details.jsonl"
OUT_CAND_CACHE = OUT_CAND_DIR / "teacher_cache.json"

# t_eval float tolerance (fallback matching)
T_EVAL_EPS = 1e-6

# Teacher (OpenRouter)
ENABLE_TEACHER_FALLBACK = os.environ.get("EGOSAT_ENABLE_LEGACY_TEACHER_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip() if ENABLE_TEACHER_FALLBACK else ""
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-3-pro-preview").strip()
OPENROUTER_URL = os.environ.get("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions").strip()
OPENROUTER_TIMEOUT_SEC = int(float(os.environ.get("OPENROUTER_TIMEOUT_SEC", "60")))
OPENROUTER_RETRY = int(float(os.environ.get("OPENROUTER_RETRY", "2")))

# progress printing knobs (kept minimal; no exports required)
MCQ_INDEX_LOG_EVERY = 500     # print every N mcq files during index building
PRED_LOG_EVERY_FILES = 5      # print every N pred files during eval_cand
SAMPLE_LOG_EVERY = 400        # print every N samples during eval_cand (lightweight)

# teacher log knobs
TEACHER_PRINT_OK_SNIPPET = 220
TEACHER_PRINT_ERR_SNIPPET = 600
RESP_SNIPPET = 240

# NEW: strong distractor definition (aligned with your generator option_sources)
STRONG_DISTURB_SOURCES = {"prev", "future"}


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# -------------------------
# Regex / token helpers
# -------------------------
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

_TAG_RE = {
    "ans": re.compile(r"<\s*ANS\s*>(.*?)<\s*/\s*ANS\s*>", re.IGNORECASE | re.DOTALL),
    "verb": re.compile(r"<\s*VERB\s*>(.*?)<\s*/\s*VERB\s*>", re.IGNORECASE | re.DOTALL),
    "noun": re.compile(r"<\s*NOUN\s*>(.*?)<\s*/\s*NOUN\s*>", re.IGNORECASE | re.DOTALL),
    "conf": re.compile(r"<\s*CONF\s*>(.*?)<\s*/\s*CONF\s*>", re.IGNORECASE | re.DOTALL),
}

# taxonomy parsing: allow multi (...) groups
def _strip_underscores(s: str) -> str:
    return (s or "").strip().strip("_").strip()

def _extract_parenthetical_groups(term: str) -> Tuple[str, List[str]]:
    s = (term or "").strip().lower()
    if not s:
        return "", []
    groups = re.findall(r"\(([^()]*)\)", s)
    base = s.split("(", 1)[0] if "(" in s else s
    base = _strip_underscores(base)
    groups = [g.strip().lower() for g in groups if g is not None]
    return base, groups

def expand_taxonomy_variants(term: str) -> List[str]:
    """
    base always included; any (...) group with comma -> synonyms; groups without comma -> ignored (disambiguation).
    """
    base, groups = _extract_parenthetical_groups(term)
    variants: List[str] = []
    if base:
        variants.append(base)

    syns: List[str] = []
    for g in groups:
        if "," in g:
            for a in g.split(","):
                a = _strip_underscores(a.strip())
                if a:
                    syns.append(a)

    for x in syns:
        if x and x not in variants:
            variants.append(x)

    out: List[str] = []
    seen = set()
    for v in variants:
        v = _strip_underscores(v)
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out

def normalize_and_tokenize(text: str) -> List[str]:
    s = (text or "").lower().replace("_", " ")
    return _WORD_RE.findall(s)

def tokens_to_underscore(toks: List[str]) -> str:
    toks = [t for t in toks if t]
    return "_".join(toks)

def variants_to_token_tuples(variants: List[str]) -> List[Tuple[str, ...]]:
    outs: List[Tuple[str, ...]] = []
    for v in variants:
        toks = normalize_and_tokenize(v)
        if toks:
            outs.append(tuple(toks))
    out2: List[Tuple[str, ...]] = []
    seen = set()
    for t in outs:
        if t not in seen:
            seen.add(t)
            out2.append(t)
    return out2

def parse_tag(text: str, which: str) -> Optional[str]:
    rx = _TAG_RE.get(which)
    if not rx:
        return None
    m = rx.search(text or "")
    if not m:
        return None
    return (m.group(1) or "").strip()

def parse_conf(text: str) -> Optional[float]:
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

def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _refresh_output_paths() -> None:
    global OUT_OPEN_SUMMARY, OUT_OPEN_DETAILS, OUT_OPEN_CACHE
    global OUT_CAND_SUMMARY, OUT_CAND_DETAILS, OUT_CAND_CACHE
    OUT_OPEN_SUMMARY = OUT_OPEN_DIR / "summary.json"
    OUT_OPEN_DETAILS = OUT_OPEN_DIR / "details.jsonl"
    OUT_OPEN_CACHE = OUT_OPEN_DIR / "teacher_cache.json"
    OUT_CAND_SUMMARY = OUT_CAND_DIR / "summary.json"
    OUT_CAND_DETAILS = OUT_CAND_DIR / "details.jsonl"
    OUT_CAND_CACHE = OUT_CAND_DIR / "teacher_cache.json"

def _infer_pred_dirs_from_explicit(pred_dir: Path, mode: str, flavor: str) -> Tuple[Path, Path]:
    p = pred_dir.expanduser()
    cand_name = (flavor or "cand").strip()
    if mode == "open":
        return p, PRED_CAND_DIR
    if mode == "cand":
        return PRED_OPEN_DIR, p
    if p.name == "open":
        return p, p.parent / cand_name
    if p.name == cand_name:
        return p.parent / "open", p
    return p / "open", p / cand_name

def _apply_cli_paths(
    *,
    mode: str,
    model_name: Optional[str],
    runs_root: Optional[Path],
    pred_dir: Optional[Path],
    out_dir: Optional[Path],
    flavor: str,
) -> None:
    global MODEL_NAME, PRED_OPEN_DIR, PRED_CAND_DIR, OUT_OPEN_DIR, OUT_CAND_DIR

    if model_name:
        MODEL_NAME = str(model_name).strip()

    cand_name = (flavor or "cand").strip()
    if pred_dir is not None:
        PRED_OPEN_DIR, PRED_CAND_DIR = _infer_pred_dirs_from_explicit(pred_dir, mode, cand_name)
    elif runs_root is not None:
        base = runs_root.expanduser() / MODEL_NAME / "sh_rtrv"
        PRED_OPEN_DIR = base / "open"
        PRED_CAND_DIR = base / cand_name
    else:
        PRED_OPEN_DIR = Path.home() / "benchmark_val" / "testllm" / MODEL_NAME / "sh_rtrv" / "open"
        PRED_CAND_DIR = Path.home() / "benchmark_val" / "testllm" / MODEL_NAME / "sh_rtrv" / cand_name

    if out_dir is not None:
        root = out_dir.expanduser()
        if mode == "open":
            OUT_OPEN_DIR = root
            OUT_CAND_DIR = Path.home() / "benchmark_val" / "score" / MODEL_NAME / "sh_rtrv_cand"
        elif mode == "cand":
            OUT_OPEN_DIR = Path.home() / "benchmark_val" / "score" / MODEL_NAME / "sh_rtrv_open"
            OUT_CAND_DIR = root
        else:
            OUT_OPEN_DIR = root / "open"
            OUT_CAND_DIR = root / cand_name
    else:
        OUT_OPEN_DIR = Path.home() / "benchmark_val" / "score" / MODEL_NAME / "sh_rtrv_open"
        OUT_CAND_DIR = Path.home() / "benchmark_val" / "score" / MODEL_NAME / "sh_rtrv_cand"
    _refresh_output_paths()

def _snippet(s: str, n: int) -> str:
    s = (s or "")
    s = s.replace("\n", "\\n")
    if len(s) <= n:
        return s
    return s[:n] + "..."

def _ans_to_idx(ans: str) -> int:
    a = (ans or "").strip().upper()
    if not a or a[0] not in "ABCD":
        return -1
    return ord(a[0]) - ord("A")

def _inc_count(d: Dict[str, int], k: str, add: int = 1) -> None:
    d[k] = int(d.get(k, 0)) + int(add)


# -------------------------
# NEW: robust local ANS recovery helpers
# -------------------------
_ANGLE_BR_RE = re.compile(r"<[^>]*>")

def _strip_angle_brackets(text: str) -> str:
    """Remove any <...> chunks entirely, so letters inside tag names don't count."""
    return _ANGLE_BR_RE.sub(" ", text or "")

def _guess_ans_letter_from_clean(clean_text: str) -> str:
    """
    Try to find exactly ONE candidate in {A,B,C,D} from clean text AFTER removing <...>.
    If multiple distinct letters appear, return "" (ambiguous).
    """
    t = _strip_angle_brackets(clean_text)
    if not t.strip():
        return ""

    # patterns that indicate a direct selection
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

    # If it looks like it is listing options (multiple line starters), don't guess.
    starters = set(re.findall(r"(?:^|\n)\s*([A-D])\s*[\.\)\|]\s*", t))
    if len(starters) >= 2:
        return ""

    if len(uniq) == 1:
        return uniq[0]
    return ""

def _parse_option_verb_noun(opt: str) -> Tuple[str, str]:
    """
    Options are typically: "YOU <verb> <noun>".
    Return raw verb, raw noun (strings).
    """
    s = (opt or "").strip()
    if not s:
        return "", ""
    if s.upper().startswith("YOU "):
        s = s[4:].strip()
    parts = s.split()
    if len(parts) >= 2:
        verb = parts[0].strip()
        noun = parts[1].strip()
        return verb, noun
    if len(parts) == 1:
        return parts[0].strip(), ""
    return "", ""

def _base_tokens(term: str) -> List[str]:
    """
    Take base before '(' and normalize underscores -> tokens.
    e.g. hold_(support,_grip,_grasp) -> ["hold"]
    """
    s = (term or "").strip().lower()
    if not s:
        return []
    if "(" in s:
        s = s.split("(", 1)[0].strip()
    s = _strip_underscores(s)
    toks = normalize_and_tokenize(s)
    return toks

def _guess_ans_from_option_terms(clean_text: str, options4: List[str]) -> Tuple[str, bool]:
    """
    Return (ans_letter_or_empty, is_multi_answer_invalid).

    Rule:
      - If exactly one option i has BOTH its verb and noun present in clean response -> choose it.
      - If multiple options contribute verb/noun signals (even if no complete pair) -> multi-answer invalid.
      - Otherwise -> ("", False) to allow teacher fallback.
    """
    t = _strip_angle_brackets(clean_text).lower()
    resp_tokens = normalize_and_tokenize(t)
    tokset = set(resp_tokens)

    verb_hit = [False, False, False, False]
    noun_hit = [False, False, False, False]
    pair_hit = [False, False, False, False]

    opt_keys: List[Tuple[str, str]] = []
    for i in range(4):
        if i >= len(options4):
            opt_keys.append(("", ""))
            continue
        v_raw, n_raw = _parse_option_verb_noun(options4[i])
        v_base = _base_tokens(v_raw)
        n_base = _base_tokens(n_raw)
        v_tok = v_base[0] if v_base else ""
        # noun may be multi-token; require all tokens present
        if v_tok:
            verb_hit[i] = (v_tok in tokset)
        if n_base:
            noun_hit[i] = all(w in tokset for w in n_base)
        pair_hit[i] = bool(verb_hit[i] and noun_hit[i])

        # for duplicate detection
        opt_keys.append((v_tok, tokens_to_underscore(n_base) if n_base else ""))

    pair_idx = [i for i in range(4) if pair_hit[i]]
    if len(pair_idx) == 1:
        return "ABCD"[pair_idx[0]], False

    if len(pair_idx) > 1:
        # If they are actually identical (duplicate options), do NOT mark multi-answer;
        # let teacher force a decision (since label space still A-D).
        keys = {opt_keys[i] for i in pair_idx}
        if len(keys) == 1:
            return "", False
        return "", True  # multi-answer invalid

    # No full pair. Check whether multiple options have ANY verb/noun signal.
    signal_idx = set([i for i in range(4) if verb_hit[i] or noun_hit[i]])
    if len(signal_idx) >= 2:
        # multiple different option signals appear -> invalid per your rule
        keys = {opt_keys[i] for i in signal_idx}
        if len(keys) >= 2:
            return "", True
    return "", False


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
# Taxonomy structures (noun groups)
# -------------------------
class NounGroup:
    def __init__(self, term_raw: str, variants: List[str]):
        self.term_raw = term_raw
        self.variants = variants[:]
        self.variant_tuples = variants_to_token_tuples(variants)
        self.variant_strs = sorted({tokens_to_underscore(list(t)) for t in self.variant_tuples})

def load_taxonomy(tax_path: Path) -> Dict[str, Any]:
    tax = json.loads(tax_path.read_text(encoding="utf-8"))
    nouns = tax.get("nouns", [])
    verbs = tax.get("verbs", [])
    if not isinstance(nouns, list):
        nouns = []
    if not isinstance(verbs, list):
        verbs = []
    return {"nouns": nouns, "verbs": verbs}

def build_noun_groups(nouns: List[Any]) -> Tuple[List[NounGroup], Dict[Tuple[str, ...], int], List[int], set]:
    groups: List[NounGroup] = []
    var_tuple_to_gid: Dict[Tuple[str, ...], int] = {}

    for n in nouns:
        if not isinstance(n, str):
            continue
        vars_ = expand_taxonomy_variants(n)
        g = NounGroup(term_raw=n, variants=vars_)
        if not g.variant_tuples:
            continue
        gid = len(groups)
        groups.append(g)
        for t in g.variant_tuples:
            if t not in var_tuple_to_gid:
                var_tuple_to_gid[t] = gid

    lengths_desc = sorted({len(t) for t in var_tuple_to_gid.keys()}, reverse=True)
    noun_token_vocab: set = set()
    for tup in var_tuple_to_gid.keys():
        for w in tup:
            noun_token_vocab.add(w)

    return groups, var_tuple_to_gid, lengths_desc, noun_token_vocab

def build_verb_token_vocab(verbs: List[Any]) -> set:
    vocab: set = set()
    for v in verbs:
        if not isinstance(v, str):
            continue
        vars_ = expand_taxonomy_variants(v)
        for t in variants_to_token_tuples(vars_):
            for w in t:
                vocab.add(w)
    return vocab


# -------------------------
# GT parsing helpers (same logic style as now_narration)
# -------------------------
_DET_SET = {
    "the", "a", "an", "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    "some", "any", "each", "every", "either", "neither",
    "few", "several", "many", "much", "no", "one", "two", "three", "four", "five", "both", "all",
}

_PREP_SET = {
    "to", "with", "in", "on", "at", "from", "into", "onto", "over", "under",
    "about", "for", "of", "off", "up", "down", "across", "through", "around",
    "between", "during", "before", "after", "without", "within", "by", "as",
}

def _maybe_stem_verb_token(tok: str, vocab: set) -> str:
    t = tok
    if t in vocab:
        return t
    if t.endswith("ing") and len(t) > 5:
        stem = t[:-3]
        if stem in vocab:
            return stem
    if t.endswith("ed") and len(t) > 3:
        stem = t[:-2]
        if stem in vocab:
            return stem
    if t.endswith("es") and len(t) > 3:
        stem = t[:-2]
        if stem in vocab:
            return stem
    if t.endswith("s") and len(t) > 2:
        stem = t[:-1]
        if stem in vocab:
            return stem
    return t

def _maybe_singularize_noun_token(tok: str, noun_token_vocab: set) -> str:
    t = tok
    if t in noun_token_vocab:
        return t
    if t.endswith("s") and len(t) > 3:
        stem = t[:-1]
        if stem in noun_token_vocab:
            return stem
    return t

def find_all_seq_positions(tokens: List[str], seq: List[str]) -> List[int]:
    if not tokens or not seq or len(seq) > len(tokens):
        return []
    out: List[int] = []
    L = len(seq)
    for i in range(0, len(tokens) - L + 1):
        if tokens[i:i+L] == seq:
            out.append(i)
    return out

def primary_verb_tokens(gt_verb: str) -> List[str]:
    v = (gt_verb or "").strip().lower()
    if not v:
        return []
    if "(" in v:
        v = v.split("(", 1)[0].strip()
    v = _strip_underscores(v)
    if not v:
        return []
    return [t for t in v.split("_") if t]

def _filter_occ_overlap(occ: List[Tuple[int, int, int]], verb_pos: Optional[int], verb_len: int) -> List[Tuple[int, int, int]]:
    if not occ:
        return occ
    if verb_pos is None or verb_len <= 0:
        return occ
    v0 = int(verb_pos)
    v1 = int(verb_pos + verb_len)
    out: List[Tuple[int, int, int]] = []
    for s, L, gid in occ:
        n0 = int(s)
        n1 = int(s + L)
        if n0 < v1 and n1 > v0:
            continue
        out.append((s, L, gid))
    return out

def choose_gid_object_prefer_after_verb(
    occ: List[Tuple[int, int, int]],
    verb_pos: Optional[int],
    verb_len: int,
) -> Optional[int]:
    if not occ:
        return None
    if verb_pos is None or verb_len <= 0:
        occ2 = sorted(occ, key=lambda x: (-x[1], x[0]))
        return occ2[0][2]
    v_end = int(verb_pos + verb_len)
    after = [o for o in occ if o[0] >= v_end]
    if after:
        after2 = sorted(after, key=lambda x: (x[0] - v_end, -x[1], x[0]))
        return after2[0][2]
    occ2 = sorted(occ, key=lambda x: (abs(x[0] - verb_pos), -x[1], x[0]))
    return occ2[0][2]

def heuristic_noun_after_verb(tokens_raw: List[str], verb_pos: Optional[int], verb_len: int) -> Optional[str]:
    if verb_pos is None or verb_len <= 0:
        return None
    start = int(verb_pos + verb_len)
    if start >= len(tokens_raw):
        return None
    i = start
    while i < len(tokens_raw) and tokens_raw[i] in _DET_SET:
        i += 1
    if i >= len(tokens_raw):
        return None
    j = i
    while j < len(tokens_raw) and tokens_raw[j] not in _PREP_SET:
        j += 1
    if j <= i:
        return None
    phrase = tokens_raw[i:j]
    if not phrase:
        return None
    return tokens_to_underscore(phrase)

def gt_expand_verb_candidates(gt_verb: str) -> Tuple[List[Tuple[str, ...]], set]:
    vars_ = expand_taxonomy_variants(gt_verb)
    tuples_ = variants_to_token_tuples(vars_)
    strs_ = {tokens_to_underscore(list(t)) for t in tuples_}
    return tuples_, strs_

def gt_extract_noun_candidates_from_narration(
    *,
    gt_text_narr: str,
    gt_verb: str,
    noun_groups: List[NounGroup],
    noun_var_tuple_to_gid: Dict[Tuple[str, ...], int],
    noun_lengths_desc: List[int],
    noun_token_vocab: set,
    verb_token_vocab: set,
) -> Tuple[set, Dict[str, Any]]:
    diag: Dict[str, Any] = {
        "noun_parse": "taxonomy_match",
        "chosen_gid": None,
        "fallback_phrase": None,
        "verb_pos": None,
        "verb_len": None,
        "matched_occ": 0,
    }

    narr_raw = str(gt_text_narr or "")
    narr_tokens_raw = normalize_and_tokenize(narr_raw)
    if len(narr_tokens_raw) >= 2 and narr_tokens_raw[0] == "c" and narr_tokens_raw[1] == "c":
        narr_tokens_raw = narr_tokens_raw[2:]

    vseq = primary_verb_tokens(gt_verb)
    narr_tokens_for_verbpos = [_maybe_stem_verb_token(t, verb_token_vocab) for t in narr_tokens_raw]
    verb_pos = None
    verb_len = 0
    if vseq:
        positions = find_all_seq_positions(narr_tokens_for_verbpos, vseq)
        if positions:
            verb_pos = min(positions)
            verb_len = len(vseq)
    diag["verb_pos"] = verb_pos
    diag["verb_len"] = verb_len

    narr_tokens_for_noun = [_maybe_singularize_noun_token(t, noun_token_vocab) for t in narr_tokens_raw]
    occ: List[Tuple[int, int, int]] = []
    n = len(narr_tokens_for_noun)
    for i in range(n):
        for L in noun_lengths_desc:
            j = i + L
            if j > n:
                continue
            tup = tuple(narr_tokens_for_noun[i:j])
            gid = noun_var_tuple_to_gid.get(tup)
            if gid is not None:
                occ.append((i, L, int(gid)))
    diag["matched_occ"] = int(len(occ))

    occ = _filter_occ_overlap(occ, verb_pos, int(verb_len))
    chosen_gid = choose_gid_object_prefer_after_verb(occ, verb_pos, int(verb_len))
    if chosen_gid is not None:
        diag["chosen_gid"] = int(chosen_gid)
        return set(noun_groups[int(chosen_gid)].variant_strs), diag

    hn = heuristic_noun_after_verb(narr_tokens_raw, verb_pos, int(verb_len))
    if hn:
        diag["noun_parse"] = "heuristic_after_verb"
        diag["fallback_phrase"] = hn
        return {hn}, diag

    diag["noun_parse"] = "none"
    return {"none"}, diag


# -------------------------
# Pred matching helpers
# -------------------------
def normalize_pred_phrase_to_tokens(s: str) -> List[str]:
    return normalize_and_tokenize(s or "")

def stem_pred_verb_tokens(tokens: List[str], vocab: set) -> List[str]:
    return [_maybe_stem_verb_token(t, vocab) for t in tokens]

def singularize_pred_noun_tokens(tokens: List[str], noun_vocab: set) -> List[str]:
    return [_maybe_singularize_noun_token(t, noun_vocab) for t in tokens]

def verb_match(pred_verb: str, gt_candidate_tuples: List[Tuple[str, ...]], verb_vocab: set) -> bool:
    ptoks = normalize_pred_phrase_to_tokens(pred_verb)
    if not ptoks:
        return False
    ptoks = stem_pred_verb_tokens(ptoks, verb_vocab)
    ptup = tuple(ptoks)
    for gt in gt_candidate_tuples:
        if ptup == gt:
            return True
    if len(ptup) == 1:
        for gt in gt_candidate_tuples:
            if len(gt) == 1 and ptup[0] == gt[0]:
                return True
    return False

def noun_match(pred_noun: str, gt_noun_candidate_set: set, noun_vocab: set) -> bool:
    if (pred_noun or "").strip().lower() in {"none", "no", "null"}:
        return "none" in {str(x).strip().lower() for x in gt_noun_candidate_set}
    ptoks = normalize_pred_phrase_to_tokens(pred_noun)
    if not ptoks:
        return False
    ptoks = singularize_pred_noun_tokens(ptoks, noun_vocab)
    pstr = tokens_to_underscore(ptoks)
    gt_norm = {str(x).strip().lower() for x in gt_noun_candidate_set}
    return pstr.lower() in gt_norm

def normalize_ans_letter(x: str) -> str:
    s = (x or "").strip().upper()
    s = re.sub(r"[^A-D]", "", s)  # keep only A-D chars
    if len(s) >= 1 and s[0] in "ABCD":
        return s[0]
    return ""


# -------------------------
# Teacher (OpenRouter)
# -------------------------
def openrouter_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

def _openrouter_chat(system_prompt: str, user_prompt: str, max_tokens: int = 1600) -> Tuple[str, Optional[str]]:
    """
    Returns (text, err_string_or_None).
    Prints HTTP status / error snippet for visibility.
    """
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

def teacher_extract_verb_noun(response_text: str) -> Tuple[str, str, Dict[str, Any]]:
    diag: Dict[str, Any] = {"used": True, "task": "extract_verb_noun", "model": OPENROUTER_MODEL, "ok": False, "raw_text": None, "error": None}

    sys_prompt = (
        "You are a teacher, evaluating a student's performance on MCQ tasks.\n"
        "You extract structured labels from a student's textual answer.\n"
        "Given ONLY the student's response text, output a JSON object with exactly two keys:\n"
        '  {"verb": "...", "noun": "..."}\n'
        "Rules:\n"
        "- verb: the single main action verb in base form if possible (e.g., hold, move, open, take, wipe, walk).\n"
        "  If there is NO clear action verb, set verb to an empty string \"\".\n"
        "- noun: the main object noun or noun phrase (can be multi-word).\n"
        "  If there is NO clear object, set noun to \"none\" (or empty string).\n"
        "- Output JSON only. No extra keys. No commentary."
    )
    user_prompt = "Student response text:\n-----\n" + (response_text or "").strip() + "\n-----\nNow output JSON."

    _log(f"Teacher call: extract_verb_noun (timeout={OPENROUTER_TIMEOUT_SEC}s, model={OPENROUTER_MODEL})")
    txt, err = _openrouter_chat(sys_prompt, user_prompt, max_tokens=128)
    diag["raw_text"] = txt
    diag["error"] = err
    if err:
        return "", "", diag

    try:
        obj = json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
        obj = json.loads(m.group(0)) if m else {}

    verb = str(obj.get("verb", "") if obj else "").strip()
    noun = str(obj.get("noun", "") if obj else "").strip()
    diag["ok"] = True
    return verb, noun, diag

def teacher_map_answer_to_option(response_text: str, options: List[str]) -> Tuple[str, Dict[str, Any]]:
    """
    MODIFIED (ONLY this function):
    - Teacher returns "" ONLY when extremely uncertain / no usable signal.
    - Otherwise teacher must choose A/B/C/D.
    - No 'default to A' rule.
    - If teacher explicitly returns {"ans":""}, we KEEP it empty (do not regex-pick a letter).
    """
    diag: Dict[str, Any] = {"used": True, "task": "map_ans", "model": OPENROUTER_MODEL, "ok": False, "raw_text": None, "error": None}

    letters = ["A", "B", "C", "D"]
    opt_lines = []
    for i in range(min(4, len(options))):
        opt_lines.append(f"{letters[i]}. {options[i]}")
    opt_block = "\n".join(opt_lines) if opt_lines else "(no options provided)"

    # UPDATED per your requirement:
    # - Only return "" when extremely uncertain / no usable signal
    # - Otherwise pick A/B/C/D
    # - Do NOT default to A
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
        "- If evidence is mixed or ambiguous, still choose the SINGLE best match; do NOT return empty just due to uncertainty.\n"
        "- Do NOT default to A. Choose A only when it is actually the best match.\n"
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

    parsed_json = False
    ans_raw = ""
    ans = ""

    # 1) Try strict JSON parse
    try:
        obj = json.loads(txt)
        parsed_json = True
        ans_raw = str(obj.get("ans", "") if obj else "")
    except Exception:
        # 2) Try to extract a JSON object substring if wrapped
        m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                parsed_json = True
                ans_raw = str(obj.get("ans", "") if obj else "")
            except Exception:
                parsed_json = False
                ans_raw = ""

    if parsed_json:
        # IMPORTANT: if teacher explicitly returns empty, keep it empty.
        if (ans_raw is None) or (str(ans_raw).strip() == ""):
            diag["ok"] = False
            return "", diag

        ans = normalize_ans_letter(str(ans_raw))
        if ans:
            diag["ok"] = True
            return ans, diag
        # If teacher produced invalid non-empty, fall through to regex salvage.

    # 3) Salvage from text ONLY when JSON parse failed or ans invalid (non-empty but not A-D)
    m1 = re.search(r'"ans"\s*:\s*"([A-D])"', str(txt), flags=re.IGNORECASE)
    if m1:
        ans = m1.group(1).upper()
    else:
        m2 = re.search(r"\b([A-D])\b", str(txt).upper())
        ans = m2.group(1) if m2 else ""

    ans = normalize_ans_letter(ans)
    diag["ok"] = bool(ans)
    return ans, diag


# -------------------------
# File indexing helpers
# -------------------------
def _parse_triplet_from_filename(p: Path) -> Tuple[str, str, str]:
    name = p.name
    parts = name.split("__")
    if len(parts) >= 3:
        video_uid = parts[0].strip()
        clip_id = parts[1].strip()
        clip_uid = parts[2].split(".", 1)[0].strip()
        return video_uid, clip_id, clip_uid
    return "", "", ""

def build_gt_full_index(gt_full_dir: Path) -> Dict[Tuple[str, str, str], Path]:
    out: Dict[Tuple[str, str, str], Path] = {}
    files = sorted([p for p in gt_full_dir.rglob("*.jsonl") if p.is_file()], key=lambda x: x.as_posix())
    for p in files:
        try:
            obj = safe_load_json_or_one_jsonl(p)
            video_uid = str(obj.get("video_uid", "")).strip()
            meta = obj.get("video_metadata", {}) if isinstance(obj.get("video_metadata"), dict) else {}
            clip_id = str(meta.get("clip_id", "")).strip()
            clip_uid = str(meta.get("clip_uid", "")).strip()
            if not (video_uid and clip_id and clip_uid):
                v2, c2, u2 = _parse_triplet_from_filename(p)
                video_uid = video_uid or v2
                clip_id = clip_id or c2
                clip_uid = clip_uid or u2
            if video_uid and clip_id and clip_uid:
                out[(video_uid, clip_id, clip_uid)] = p
        except Exception:
            continue
    return out

def build_mcq_index(mcq_dir: Path) -> Dict[Tuple[str, str, str], Path]:
    out: Dict[Tuple[str, str, str], Path] = {}
    t0 = time.time()
    files = sorted(
        [p for p in mcq_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}],
        key=lambda x: x.as_posix(),
    )
    _log(f"MCQ index: scanning {len(files)} files under {mcq_dir}")

    for i, p in enumerate(files, 1):
        if i == 1 or (i % MCQ_INDEX_LOG_EVERY == 0):
            _log(f"MCQ index progress: {i}/{len(files)} (last={p.name})")

        v, c, u = _parse_triplet_from_filename(p)
        if v and c and u:
            out[(v, c, u)] = p
            continue

        try:
            obj = safe_load_json_or_one_jsonl(p)
            video_uid = str(obj.get("video_uid", "")).strip()
            clip_id = str(obj.get("clip_id", "")).strip()
            clip_uid = ""
            gt_file = str(obj.get("gt_file", "")).strip()
            if gt_file:
                clip_uid = _parse_triplet_from_filename(Path(gt_file))[2]
            if video_uid and clip_id and clip_uid:
                out[(video_uid, clip_id, clip_uid)] = p
        except Exception:
            continue

    _log(f"MCQ index: built {len(out)} keys in {time.time() - t0:.2f}s")
    return out

def build_pred_file_list(pred_dir: Path) -> List[Path]:
    """
    FIXED:
    - ONLY include prediction output files.
    - Previously, it appended all *.json (including derived/queryset jsons).
    """
    if not pred_dir.exists():
        return []
    t0 = time.time()
    files = sorted([p for p in pred_dir.rglob("*.json") if p.is_file()], key=lambda x: x.as_posix())
    out: List[Path] = []
    skipped = 0
    for p in files:
        name = p.name.lower()
        if "manifest" in name:
            skipped += 1
            continue
        # STRICT keep pred only
        if name.endswith("__pred.json") or name.endswith("_pred.json") or "__pred" in name:
            out.append(p)
        else:
            skipped += 1
            # do not include derived/queryset jsons
            continue
    _log(f"Pred list: {pred_dir} -> {len(out)} files (scan {time.time() - t0:.2f}s, skipped_non_pred={skipped})")
    return out


# -------------------------
# Matching GT sample for a pred sample (idx first, then t_eval)
# -------------------------
def match_gt_sample(gt_samples: List[Dict[str, Any]], pred_sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    gt_by_idx: Dict[int, Dict[str, Any]] = {}
    gt_by_t: Dict[float, Dict[str, Any]] = {}

    for s in gt_samples:
        if not isinstance(s, dict):
            continue
        idx = s.get("idx", None)
        if isinstance(idx, (int, float)):
            gt_by_idx[int(idx)] = s
        t = s.get("t_eval", None)
        if isinstance(t, (int, float)):
            gt_by_t[float(t)] = s

    p_idx = pred_sample.get("idx", None)
    p_t = pred_sample.get("t_eval", None)

    if isinstance(p_idx, (int, float)) and int(p_idx) in gt_by_idx:
        return gt_by_idx[int(p_idx)]

    if isinstance(p_t, (int, float)):
        tt = float(p_t)
        if tt in gt_by_t:
            return gt_by_t[tt]
        best = None
        best_d = 1e9
        for t0, s0 in gt_by_t.items():
            d = abs(t0 - tt)
            if d < best_d:
                best_d = d
                best = s0
        if best is not None and best_d <= T_EVAL_EPS:
            return best

    return None


# -------------------------
# Extract GT fields for sh_rtrv open (robust key fallback)
# -------------------------
def get_gt_verb(sample: Dict[str, Any]) -> str:
    v = sample.get("gt_verb", None)
    if isinstance(v, str) and v.strip():
        return v.strip()
    g = sample.get("gt", None)
    if isinstance(g, dict):
        v2 = g.get("verb", None)
        if isinstance(v2, str) and v2.strip():
            return v2.strip()
    return ""

def get_gt_narration(sample: Dict[str, Any]) -> str:
    for k in ("gt_text_narr", "narration_text", "narration", "gt_text", "gt_text_canonical"):
        v = sample.get(k, None)
        if isinstance(v, str) and v.strip():
            return v
    g = sample.get("gt", None)
    if isinstance(g, dict):
        for k in ("text", "narration", "gt_text_narr"):
            v = g.get(k, None)
            if isinstance(v, str) and v.strip():
                return v
    return ""


# -------------------------
# Pretty print
# -------------------------
def _print_block(title: str) -> None:
    print("\n" + "=" * 18 + f" {title} " + "=" * 18 + "\n", flush=True)


# -------------------------
# Evaluation: OPEN
# -------------------------
def eval_open(
    *,
    noun_groups: List[NounGroup],
    noun_var_tuple_to_gid: Dict[Tuple[str, ...], int],
    noun_lengths_desc: List[int],
    noun_token_vocab: set,
    verb_token_vocab: set,
) -> Dict[str, Any]:
    _ensure_dir(OUT_OPEN_DIR)

    pred_files = build_pred_file_list(PRED_OPEN_DIR)
    _log("GT index: building full GT index ...")
    t0 = time.time()
    gt_index = build_gt_full_index(GT_FULL_DIR)
    _log(f"GT index: built {len(gt_index)} keys in {time.time() - t0:.2f}s")

    teacher_cache: Dict[str, Dict[str, str]] = {}
    if OUT_OPEN_CACHE.exists():
        try:
            teacher_cache = json.loads(OUT_OPEN_CACHE.read_text(encoding="utf-8"))
            if not isinstance(teacher_cache, dict):
                teacher_cache = {}
        except Exception:
            teacher_cache = {}

    if OUT_OPEN_DETAILS.exists():
        OUT_OPEN_DETAILS.unlink(missing_ok=True)

    num_files_total = 0
    num_files_matched = 0
    missing_gt_files = 0
    missing_gt_samples = 0

    total_samples = 0
    verb_correct = 0
    noun_correct = 0
    pair_correct = 0

    pred_missing_verb_tag = 0
    pred_missing_noun_tag = 0
    teacher_calls = 0
    teacher_cache_hits = 0
    teacher_fail = 0

    for pf in pred_files:
        num_files_total += 1
        try:
            pred = json.loads(pf.read_text(encoding="utf-8"))
        except Exception:
            continue

        video_uid = str(pred.get("video_uid", "")).strip()
        meta = pred.get("video_metadata", {}) if isinstance(pred.get("video_metadata"), dict) else {}
        om = meta.get("original_video_metadata", {}) if isinstance(meta.get("original_video_metadata"), dict) else {}
        clip_id = str(om.get("clip_id", "")).strip()
        clip_uid = str(om.get("clip_uid", "")).strip()

        key = (video_uid, clip_id, clip_uid)
        gt_path = gt_index.get(key)
        if gt_path is None:
            missing_gt_files += 1
            continue

        gt = safe_load_json_or_one_jsonl(gt_path)
        gt_samples = gt.get("samples", [])
        if not isinstance(gt_samples, list):
            gt_samples = []

        pred_samples = pred.get("samples", [])
        if not isinstance(pred_samples, list):
            continue

        num_files_matched += 1

        for ps in pred_samples:
            if not isinstance(ps, dict):
                continue
            total_samples += 1

            gt_s = match_gt_sample(gt_samples, ps)
            if gt_s is None:
                missing_gt_samples += 1
                continue

            resp_text = str(ps.get("response_text", "") or "")

            pred_verb_raw = parse_tag(resp_text, "verb")
            pred_noun_raw = parse_tag(resp_text, "noun")

            if pred_verb_raw is None:
                pred_missing_verb_tag += 1
            if pred_noun_raw is None:
                pred_missing_noun_tag += 1

            pred_verb = (pred_verb_raw or "").strip()
            pred_noun = (pred_noun_raw or "").strip()

            used_teacher = False
            teacher_diag = None

            if (pred_verb_raw is None) or (pred_noun_raw is None):
                used_teacher = True
                h = _sha1(resp_text)
                if h in teacher_cache and isinstance(teacher_cache[h], dict):
                    teacher_cache_hits += 1
                    pred_verb = str(teacher_cache[h].get("verb", "")).strip()
                    pred_noun = str(teacher_cache[h].get("noun", "")).strip()
                else:
                    teacher_calls += 1
                    v2, n2, diag2 = teacher_extract_verb_noun(resp_text)
                    teacher_diag = diag2
                    if diag2.get("error"):
                        teacher_fail += 1
                    pred_verb = (v2 or "").strip()
                    pred_noun = (n2 or "").strip()
                    teacher_cache[h] = {"verb": pred_verb, "noun": pred_noun}

            gt_verb = get_gt_verb(gt_s)
            gt_text = get_gt_narration(gt_s)

            gt_verb_cand_tuples, gt_verb_cand_strs = gt_expand_verb_candidates(gt_verb)
            gt_noun_cands, gt_noun_diag = gt_extract_noun_candidates_from_narration(
                gt_text_narr=gt_text,
                gt_verb=gt_verb,
                noun_groups=noun_groups,
                noun_var_tuple_to_gid=noun_var_tuple_to_gid,
                noun_lengths_desc=noun_lengths_desc,
                noun_token_vocab=noun_token_vocab,
                verb_token_vocab=verb_token_vocab,
            )

            local_verb_vocab = set(verb_token_vocab)
            for t in gt_verb_cand_tuples:
                for w in t:
                    local_verb_vocab.add(w)

            local_noun_vocab = set(noun_token_vocab)
            for s0 in gt_noun_cands:
                for w in normalize_and_tokenize(str(s0)):
                    local_noun_vocab.add(w)

            v_hit = verb_match(pred_verb, gt_verb_cand_tuples, local_verb_vocab)
            n_hit = noun_match(pred_noun, gt_noun_cands, local_noun_vocab)
            p_hit = bool(v_hit and n_hit)

            if v_hit:
                verb_correct += 1
            if n_hit:
                noun_correct += 1
            if p_hit:
                pair_correct += 1

            detail = {
                "file": pf.name,
                "video_uid": video_uid,
                "clip_id": clip_id,
                "clip_uid": clip_uid,
                "idx": int(ps.get("idx")) if isinstance(ps.get("idx"), (int, float)) else None,
                "t_eval": float(ps.get("t_eval")) if isinstance(ps.get("t_eval"), (int, float)) else None,
                "gt_verb": gt_verb,
                "gt_verb_candidates": sorted(gt_verb_cand_strs),
                "gt_noun_candidates": sorted({str(x) for x in gt_noun_cands}),
                "gt_noun_diag": gt_noun_diag,
                "pred_verb": pred_verb,
                "pred_noun": pred_noun,
                "used_teacher": used_teacher,
                "teacher_diag": teacher_diag,
                "verb_hit": bool(v_hit),
                "noun_hit": bool(n_hit),
                "pair_hit": bool(p_hit),
            }
            with open(OUT_OPEN_DETAILS, "a", encoding="utf-8") as f:
                f.write(json.dumps(detail, ensure_ascii=False) + "\n")

    OUT_OPEN_CACHE.write_text(json.dumps(teacher_cache, ensure_ascii=False, indent=2), encoding="utf-8")

    verb_acc = (verb_correct / total_samples) if total_samples > 0 else 0.0
    noun_acc = (noun_correct / total_samples) if total_samples > 0 else 0.0
    pair_acc = (pair_correct / total_samples) if total_samples > 0 else 0.0

    summary = {
        "model_name": MODEL_NAME,
        "task": "sh_rtrv_open",
        "gt_full_dir": str(GT_FULL_DIR),
        "pred_dir": str(PRED_OPEN_DIR),
        "out_dir": str(OUT_OPEN_DIR),
        "taxonomy_json": str(TAXONOMY_JSON),
        "num_pred_files_total": int(num_files_total),
        "num_pred_files_matched_gt": int(num_files_matched),
        "missing_gt_files": int(missing_gt_files),
        "missing_gt_samples": int(missing_gt_samples),
        "num_pred_samples_evaluated": int(total_samples),
        "verb_accuracy": float(verb_acc),
        "noun_accuracy": float(noun_acc),
        "pair_accuracy": float(pair_acc),
        "pred_missing_verb_tag": int(pred_missing_verb_tag),
        "pred_missing_noun_tag": int(pred_missing_noun_tag),
        "teacher_calls": int(teacher_calls),
        "teacher_cache_hits": int(teacher_cache_hits),
        "teacher_fail": int(teacher_fail),
        "openrouter_model": OPENROUTER_MODEL,
        "timestamp_unix": time.time(),
    }

    OUT_OPEN_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# -------------------------
# Evaluation: CAND
# -------------------------
def eval_cand() -> Dict[str, Any]:
    _ensure_dir(OUT_CAND_DIR)

    _log("CAND: building pred file list ...")
    pred_files = build_pred_file_list(PRED_CAND_DIR)

    _log("CAND: building MCQ index (this can be slow on shared disks) ...")
    mcq_index = build_mcq_index(MCQ_DIR)

    teacher_cache: Dict[str, Dict[str, Any]] = {}
    if OUT_CAND_CACHE.exists():
        try:
            teacher_cache = json.loads(OUT_CAND_CACHE.read_text(encoding="utf-8"))
            if not isinstance(teacher_cache, dict):
                teacher_cache = {}
        except Exception:
            teacher_cache = {}

    if OUT_CAND_DETAILS.exists():
        OUT_CAND_DETAILS.unlink(missing_ok=True)

    num_files_total = 0
    num_files_matched_mcq = 0
    missing_mcq_files = 0
    missing_mcq_samples = 0

    total_samples = 0
    ans_correct = 0

    pred_missing_ans_tag = 0
    teacher_calls = 0
    teacher_cache_hits = 0
    teacher_fail = 0

    # NEW: strong distractor stats (computed on samples where pred_ans maps to option_sources)
    num_samples_with_mcq = 0
    num_samples_mapped = 0
    ans_correct_mapped = 0

    pred_source_counts: Dict[str, int] = {}
    gt_source_counts: Dict[str, int] = {}
    mcq_source_counts: Dict[str, int] = {}

    strong_choice_total = 0
    strong_choice_wrong = 0
    strong_prev_total = 0
    strong_future_total = 0

    source_stats = init_source_stats()

    conf_sum_mapped = 0.0
    conf_cnt_mapped = 0
    conf_sum_strong = 0.0
    conf_cnt_strong = 0
    conf_sum_nonstrong = 0.0
    conf_cnt_nonstrong = 0

    queryset_cache: Dict[str, Dict[str, Any]] = {}

    _log(f"CAND: start processing {len(pred_files)} pred files")

    for fi, pf in enumerate(pred_files, 1):
        num_files_total += 1

        if fi == 1 or (fi % PRED_LOG_EVERY_FILES == 0):
            _log(f"CAND: file {fi}/{len(pred_files)} -> {pf.name} (matched={num_files_matched_mcq}, missing_mcq_files={missing_mcq_files})")

        try:
            pred = json.loads(pf.read_text(encoding="utf-8"))
        except Exception as e:
            _log(f"CAND: pred json load failed: {pf.name} err={repr(e)}")
            continue

        video_uid = str(pred.get("video_uid", "")).strip()
        meta = pred.get("video_metadata", {}) if isinstance(pred.get("video_metadata"), dict) else {}
        om = meta.get("original_video_metadata", {}) if isinstance(meta.get("original_video_metadata"), dict) else {}
        clip_id = str(om.get("clip_id", "")).strip()
        clip_uid = str(om.get("clip_uid", "")).strip()

        params = pred.get("params", {}) if isinstance(pred.get("params"), dict) else {}
        mcp_path = str(params.get("mcp_path", "") or "").strip()

        mcq_by_idx: Dict[int, Dict[str, Any]] = {}
        mcq_path = None
        mcq_source = "mcq_dir"
        qpath = resolve_queryset_path_from_pred(pred)
        if qpath:
            try:
                if qpath in queryset_cache:
                    qs = queryset_cache[qpath]
                else:
                    qs = safe_load_json_or_one_jsonl(Path(qpath).expanduser())
                    queryset_cache[qpath] = qs
                mcq_by_idx = build_effective_mcq_by_idx(qs)
                if mcq_by_idx:
                    mcq_source = "effective_queryset"
            except Exception:
                if STRICT_MCQ_PATH:
                    raise
                mcq_by_idx = {}

        if not mcq_by_idx:
            if mcp_path:
                p2 = Path(mcp_path).expanduser()
                if p2.is_file():
                    mcq_path = p2
                    mcq_source = "pred.params.mcp_path"
                elif STRICT_MCQ_PATH:
                    raise FileNotFoundError(f"params.mcp_path not found and STRICT_MCQ_PATH=1: {p2}")

            if mcq_path is None:
                if STRICT_MCQ_PATH and not MCQ_DIR_EXPLICIT:
                    raise FileNotFoundError(
                        "STRICT_MCQ_PATH=1 requires effective queryset, pred.params.mcp_path, "
                        f"or an explicit --mcq-dir/SH_RTRV_MCQ_DIR; refusing implicit fallback to default MCQ_DIR={MCQ_DIR}"
                    )
                mcq_path = mcq_index.get((video_uid, clip_id, clip_uid))

            if mcq_path is None or (not Path(mcq_path).is_file()):
                missing_mcq_files += 1
                if fi == 1 or (fi % PRED_LOG_EVERY_FILES == 0):
                    _log(f"CAND: missing mcq for key={(video_uid, clip_id, clip_uid)} (mcp_path={mcp_path})")
                continue

            if fi == 1 or (fi % PRED_LOG_EVERY_FILES == 0):
                _log(f"CAND: resolved mcq_path = {Path(mcq_path).name} source={mcq_source}")

            mcq = safe_load_json_or_one_jsonl(Path(mcq_path))
            mcq_samples = mcq.get("samples", [])
            if not isinstance(mcq_samples, list):
                mcq_samples = []

            for s in mcq_samples:
                if not isinstance(s, dict):
                    continue
                idx = s.get("idx", None)
                if isinstance(idx, (int, float)):
                    mcq_by_idx[int(idx)] = s

        _inc_count(mcq_source_counts, mcq_source, 1)

        pred_samples = pred.get("samples", [])
        if not isinstance(pred_samples, list):
            continue

        num_files_matched_mcq += 1

        for ps in pred_samples:
            if not isinstance(ps, dict):
                continue
            total_samples += 1

            if total_samples == 1 or (total_samples % SAMPLE_LOG_EVERY == 0):
                _log(f"CAND: samples processed = {total_samples} (acc={ans_correct}/{total_samples})")

            p_idx = ps.get("idx", None)
            if not isinstance(p_idx, (int, float)) or int(p_idx) not in mcq_by_idx:
                missing_mcq_samples += 1
                continue

            ms = mcq_by_idx[int(p_idx)]
            num_samples_with_mcq += 1

            options = ms.get("options", [])
            if not isinstance(options, list):
                options = []
            options4 = options[:4]

            option_sources = ms.get("option_sources", [])
            if not isinstance(option_sources, list):
                option_sources = []

            gt_ans = ""
            if isinstance(ms.get("answer", None), str):
                gt_ans = normalize_ans_letter(ms.get("answer", ""))
            if not gt_ans:
                sm = ms.get("shuffle_meta", {}) if isinstance(ms.get("shuffle_meta"), dict) else {}
                if isinstance(sm.get("gt_option", None), str):
                    gt_ans = normalize_ans_letter(sm.get("gt_option", ""))

            resp_text = str(ps.get("response_text", "") or "")
            clean_text = str(ps.get("clean_response", "") or ps.get("clean", "") or resp_text or "")
            pred_conf = parse_conf(resp_text)

            pred_ans_raw = parse_tag(resp_text, "ans")
            pred_ans = normalize_ans_letter(pred_ans_raw or "")

            used_teacher = False
            teacher_diag = None
            pred_ans_method = "tag" if pred_ans else "none"

            # ---------------- NEW: local recovery before teacher ----------------
            if pred_ans_raw is None or not pred_ans:
                pred_missing_ans_tag += 1

                # (1) Find a single A/B/C/D from clean_text (excluding <...>)
                a1 = _guess_ans_letter_from_clean(clean_text)
                if a1:
                    pred_ans = a1
                    pred_ans_method = "clean_letter"
                else:
                    # (2) Match verb+noun against exactly one option
                    a2, is_multi = _guess_ans_from_option_terms(clean_text, options4)
                    if is_multi:
                        # multi-answer invalid: directly mark as invalid (no teacher)
                        pred_ans = ""
                        pred_ans_method = "multi_answer_invalid"
                    elif a2:
                        pred_ans = a2
                        pred_ans_method = "verbnoun_match"
                    else:
                        # (3) Teacher fallback
                        used_teacher = True
                        pred_ans_method = "teacher"

                        _log(
                            "CAND: need teacher_map_ans because "
                            + ("<ANS> missing" if pred_ans_raw is None else f"<ANS> invalid raw={repr(pred_ans_raw)}")
                            + f" | after_local_recover=FAIL | resp_snip={_snippet(resp_text, RESP_SNIPPET)}"
                        )

                        opt_join = "\n".join([f"{i}:{str(o)}" for i, o in enumerate(options4)])
                        h = _sha1(resp_text + "\n" + opt_join)

                        if h in teacher_cache and isinstance(teacher_cache[h], dict):
                            teacher_cache_hits += 1
                            pred_ans = normalize_ans_letter(str(teacher_cache[h].get("ans", "")))
                            _log(f"CAND: teacher cache hit -> ans={pred_ans}")
                        else:
                            teacher_calls += 1
                            a3, diag3 = teacher_map_answer_to_option(resp_text, options4)
                            teacher_diag = diag3
                            if diag3.get("error"):
                                teacher_fail += 1
                                _log(f"CAND: teacher_map_ans failed -> error={diag3.get('error')}")
                            pred_ans = normalize_ans_letter(a3)
                            teacher_cache[h] = {"ans": pred_ans, "options": options4}
                            _log(f"CAND: teacher_map_ans result -> ans={pred_ans}")
            # -------------------------------------------------------------------

            hit = bool(pred_ans and gt_ans and pred_ans == gt_ans)
            is_valid = bool(pred_ans and pred_ans in {"A", "B", "C", "D"} and gt_ans)
            if hit:
                ans_correct += 1

            source_info = update_source_stats(
                source_stats,
                pred_ans,
                gt_ans,
                ms.get("answer_idx", None),
                option_sources,
                bool(is_valid),
            )

            # ---------- strong distractor stats (only when pred_ans maps to option_sources) ----------
            pred_source = None
            gt_source = None
            pred_ans_idx = _ans_to_idx(pred_ans)
            gt_ans_idx = _ans_to_idx(gt_ans)

            if option_sources and len(option_sources) >= 4:
                if 0 <= pred_ans_idx < len(option_sources):
                    pred_source = str(option_sources[pred_ans_idx])
                if 0 <= gt_ans_idx < len(option_sources):
                    gt_source = str(option_sources[gt_ans_idx])

            if pred_source is not None:
                num_samples_mapped += 1
                _inc_count(pred_source_counts, pred_source, 1)
                if gt_source is not None:
                    _inc_count(gt_source_counts, gt_source, 1)

                if hit:
                    ans_correct_mapped += 1

                if pred_conf is not None:
                    conf_sum_mapped += float(pred_conf)
                    conf_cnt_mapped += 1

                is_strong = (pred_source in STRONG_DISTURB_SOURCES)
                if is_strong:
                    strong_choice_total += 1
                    if pred_source == "prev":
                        strong_prev_total += 1
                    elif pred_source == "future":
                        strong_future_total += 1

                    if (not hit) and (pred_ans_idx >= 0):
                        strong_choice_wrong += 1

                    if pred_conf is not None:
                        conf_sum_strong += float(pred_conf)
                        conf_cnt_strong += 1
                else:
                    if pred_conf is not None:
                        conf_sum_nonstrong += float(pred_conf)
                        conf_cnt_nonstrong += 1
            # -------------------------------------------------------------------------------

            detail = {
                "file": pf.name,
                "video_uid": video_uid,
                "clip_id": clip_id,
                "clip_uid": clip_uid,
                "mcq_source": mcq_source,
                "queryset_path": qpath,
                "mcq_path": str(mcq_path) if mcq_path is not None else None,
                "idx": int(p_idx),
                "t_eval": float(ps.get("t_eval")) if isinstance(ps.get("t_eval"), (int, float)) else None,
                "gt_ans": gt_ans,
                "pred_ans_raw": pred_ans_raw,
                "pred_ans": pred_ans,
                "pred_ans_method": pred_ans_method,
                "valid": bool(is_valid),
                "wrong": bool(is_valid and not hit),
                "pred_letter": pred_ans if pred_ans else None,
                "answer_letter": gt_ans if gt_ans else None,
                "ans_hit": bool(hit),
                "used_teacher": used_teacher,
                "teacher_diag": teacher_diag,
                "options": options4,
                "option_sources": option_sources[:4] if option_sources else None,
                "gt_source": gt_source,
                "pred_source": pred_source,
                "is_disturb": source_info.get("is_disturb"),
                "is_strong_distractor": source_info.get("is_strong_distractor"),
                "mcp_file": ms.get("mcp_file", None),
                "pred_conf": pred_conf,
                "resp_snippet": _snippet(resp_text, RESP_SNIPPET),
            }
            with open(OUT_CAND_DETAILS, "a", encoding="utf-8") as f:
                f.write(json.dumps(detail, ensure_ascii=False) + "\n")

    OUT_CAND_CACHE.write_text(json.dumps(teacher_cache, ensure_ascii=False, indent=2), encoding="utf-8")

    ans_acc = (ans_correct / total_samples) if total_samples > 0 else 0.0

    # strong distractor rates (over mapped samples only)
    mapped = int(num_samples_mapped)
    wrong_mapped = int(max(0, mapped - ans_correct_mapped))
    strong_rate = (float(strong_choice_total) / float(mapped)) if mapped > 0 else 0.0
    strong_rate_wrong = (float(strong_choice_wrong) / float(wrong_mapped)) if wrong_mapped > 0 else 0.0
    strong_prev_rate = (float(strong_prev_total) / float(mapped)) if mapped > 0 else 0.0
    strong_future_rate = (float(strong_future_total) / float(mapped)) if mapped > 0 else 0.0

    mean_conf_mapped = (conf_sum_mapped / conf_cnt_mapped) if conf_cnt_mapped > 0 else None
    mean_conf_strong = (conf_sum_strong / conf_cnt_strong) if conf_cnt_strong > 0 else None
    mean_conf_nonstrong = (conf_sum_nonstrong / conf_cnt_nonstrong) if conf_cnt_nonstrong > 0 else None
    source_summary = finalize_source_stats(source_stats)

    summary = {
        "model_name": MODEL_NAME,
        "task": "sh_rtrv_cand",
        "pred_dir": str(PRED_CAND_DIR),
        "mcq_dir": str(MCQ_DIR),
        "out_dir": str(OUT_CAND_DIR),
        "num_pred_files_total": int(num_files_total),
        "num_pred_files_matched_mcq": int(num_files_matched_mcq),
        "missing_mcq_files": int(missing_mcq_files),
        "missing_mcq_samples": int(missing_mcq_samples),
        "num_pred_samples_evaluated": int(total_samples),
        "answer_accuracy": float(ans_acc),
        "pred_missing_ans_tag_or_invalid": int(pred_missing_ans_tag),
        "teacher_calls": int(teacher_calls),
        "teacher_cache_hits": int(teacher_cache_hits),
        "teacher_fail": int(teacher_fail),

        # strong distractor report
        "strong_distractor_sources": sorted(list(STRONG_DISTURB_SOURCES)),
        "num_samples_with_mcq": int(num_samples_with_mcq),
        "num_samples_mapped_to_option_sources": int(num_samples_mapped),
        "answer_correct_mapped": int(ans_correct_mapped),
        "wrong_mapped": int(wrong_mapped),
        "pred_source_counts": pred_source_counts,
        "gt_source_counts": gt_source_counts,
        "mcq_source_counts": mcq_source_counts,
        "strong_distractor_choice_count": int(strong_choice_total),
        "strong_distractor_choice_rate_over_mapped": float(strong_rate),
        "strong_distractor_choice_rate_given_wrong_over_mapped": float(strong_rate_wrong),
        "strong_prev_choice_count": int(strong_prev_total),
        "strong_future_choice_count": int(strong_future_total),
        "strong_prev_choice_rate_over_mapped": float(strong_prev_rate),
        "strong_future_choice_rate_over_mapped": float(strong_future_rate),
        "mean_conf_over_mapped": mean_conf_mapped,
        "mean_conf_strong_over_mapped": mean_conf_strong,
        "mean_conf_nonstrong_over_mapped": mean_conf_nonstrong,

        "openrouter_model": OPENROUTER_MODEL,
        "timestamp_unix": time.time(),
    }
    for k, v in source_summary.items():
        summary.setdefault(k, v)

    OUT_CAND_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["open", "cand", "both", "all"],
        default=None,
        help="Override SH_RTRV_EVAL_MODE for this invocation.",
    )
    parser.add_argument(
        "--mcq-dir",
        type=Path,
        default=None,
        help="Override candidate MCQ directory. Env alternative: SH_RTRV_MCQ_DIR.",
    )
    parser.add_argument("--model_name", default=None, help="Override MODEL_NAME without changing the environment.")
    parser.add_argument("--runs_root", type=Path, default=None, help="Infer pred_dir as <runs_root>/<model_name>/sh_rtrv/<flavor>.")
    parser.add_argument("--pred_dir", type=Path, default=None, help="Override prediction directory; mode=cand uses it as the cand pred dir.")
    parser.add_argument("--out_dir", type=Path, default=None, help="Override output directory; mode=cand writes summary/details/cache here.")
    parser.add_argument("--flavor", default="cand", help="Prediction flavor for --runs_root inference; default: cand.")
    parser.add_argument(
        "--strict-mcq-path",
        action="store_true",
        help="Fail fast if params.mcp_path or MCQ_DIR is missing. Env alternative: STRICT_MCQ_PATH=1.",
    )
    args = parser.parse_args()

    global SH_RTRV_EVAL_MODE, MCQ_DIR, MCQ_DIR_EXPLICIT, STRICT_MCQ_PATH
    if args.mode is not None:
        SH_RTRV_EVAL_MODE = args.mode
    if args.mcq_dir is not None:
        MCQ_DIR = args.mcq_dir.expanduser()
        MCQ_DIR_EXPLICIT = True
    if args.strict_mcq_path:
        STRICT_MCQ_PATH = True

    mode = (SH_RTRV_EVAL_MODE or "both").strip().lower()
    if mode in {"all"}:
        mode = "both"
    if mode not in {"open", "cand", "both"}:
        raise ValueError(f"Invalid SH_RTRV_EVAL_MODE={SH_RTRV_EVAL_MODE!r}. Use open|cand|both.")

    _apply_cli_paths(
        mode=mode,
        model_name=args.model_name,
        runs_root=args.runs_root,
        pred_dir=args.pred_dir,
        out_dir=args.out_dir,
        flavor=args.flavor,
    )

    # NOTE: checks are now conditional based on mode (required for new mode switch)
    if mode in {"open", "both"}:
        if not TAXONOMY_JSON.is_file():
            raise FileNotFoundError(f"TAXONOMY_JSON not found: {TAXONOMY_JSON}")
        if not GT_FULL_DIR.exists():
            raise FileNotFoundError(f"GT_FULL_DIR not found: {GT_FULL_DIR}")
    if mode in {"cand", "both"}:
        if not MCQ_DIR.exists():
            raise FileNotFoundError(f"MCQ_DIR not found: {MCQ_DIR}")

    noun_groups = []
    noun_var_tuple_to_gid = {}
    noun_lengths_desc = []
    noun_token_vocab = set()
    verb_token_vocab = set()

    if mode in {"open", "both"}:
        tax = load_taxonomy(TAXONOMY_JSON)
        noun_groups, noun_var_tuple_to_gid, noun_lengths_desc, noun_token_vocab = build_noun_groups(tax["nouns"])
        verb_token_vocab = build_verb_token_vocab(tax.get("verbs", []))

        _print_block("SH_RTRV OPEN EVAL")
        print(f"MODEL_NAME     : {MODEL_NAME}", flush=True)
        print(f"EVAL_MODE      : {mode}", flush=True)
        print(f"PRED_OPEN_DIR  : {PRED_OPEN_DIR}", flush=True)
        print(f"OUT_OPEN_DIR   : {OUT_OPEN_DIR}", flush=True)
        print(f"GT_FULL_DIR    : {GT_FULL_DIR}", flush=True)
        print(f"TAXONOMY_JSON  : {TAXONOMY_JSON}", flush=True)
        open_sum = eval_open(
            noun_groups=noun_groups,
            noun_var_tuple_to_gid=noun_var_tuple_to_gid,
            noun_lengths_desc=noun_lengths_desc,
            noun_token_vocab=noun_token_vocab,
            verb_token_vocab=verb_token_vocab,
        )
        print(f"[WROTE] {OUT_OPEN_SUMMARY}", flush=True)
        print(f"[WROTE] {OUT_OPEN_DETAILS}", flush=True)
        print(f"[WROTE] {OUT_OPEN_CACHE}", flush=True)
        print("", flush=True)
        print(f"Pair accuracy (main): {open_sum['pair_accuracy']:.4f}  ({open_sum['num_pred_samples_evaluated']} samples)", flush=True)
        print(f"Verb acc           : {open_sum['verb_accuracy']:.4f}", flush=True)
        print(f"Noun acc           : {open_sum['noun_accuracy']:.4f}", flush=True)
        print(f"Teacher calls      : {open_sum['teacher_calls']} (cache hits {open_sum['teacher_cache_hits']})", flush=True)

    if mode in {"cand", "both"}:
        _print_block("SH_RTRV CAND EVAL")
        print(f"MODEL_NAME     : {MODEL_NAME}", flush=True)
        print(f"EVAL_MODE      : {mode}", flush=True)
        print(f"PRED_CAND_DIR  : {PRED_CAND_DIR}", flush=True)
        print(f"MCQ_DIR        : {MCQ_DIR}", flush=True)
        print(f"OUT_CAND_DIR   : {OUT_CAND_DIR}", flush=True)
        print(f"STRICT_MCQ_PATH: {STRICT_MCQ_PATH}", flush=True)
        cand_sum = eval_cand()
        print(f"[WROTE] {OUT_CAND_SUMMARY}", flush=True)
        print(f"[WROTE] {OUT_CAND_DETAILS}", flush=True)
        print(f"[WROTE] {OUT_CAND_CACHE}", flush=True)
        print("", flush=True)
        print(f"Answer accuracy (main): {cand_sum['answer_accuracy']:.4f}  ({cand_sum['num_pred_samples_evaluated']} samples)", flush=True)
        print(f"Teacher calls        : {cand_sum['teacher_calls']} (cache hits {cand_sum['teacher_cache_hits']})", flush=True)
        print(f"MCQ source counts    : {cand_sum.get('mcq_source_counts')}", flush=True)

        # strong distractor report (prev+future)
        n_map = int(cand_sum.get("num_samples_mapped_to_option_sources", 0) or 0)
        strong_rate = float(cand_sum.get("strong_distractor_choice_rate_over_mapped", 0.0) or 0.0)
        strong_rate_wrong = float(cand_sum.get("strong_distractor_choice_rate_given_wrong_over_mapped", 0.0) or 0.0)
        prev_rate = float(cand_sum.get("strong_prev_choice_rate_over_mapped", 0.0) or 0.0)
        future_rate = float(cand_sum.get("strong_future_choice_rate_over_mapped", 0.0) or 0.0)

        print("", flush=True)
        print("---- Unified source stats (valid-only) ----", flush=True)
        for line in source_stats_report_lines(cand_sum):
            print(line, flush=True)
        print("", flush=True)
        print(f"Strong distractor sources      : {cand_sum.get('strong_distractor_sources')}", flush=True)
        print(f"Mapped samples (ANS->source)   : {n_map}", flush=True)
        print(f"Strong distractor choice rate  : {strong_rate:.4f}  (prev={prev_rate:.4f}, future={future_rate:.4f})", flush=True)
        print(f"Strong distractor rate | wrong : {strong_rate_wrong:.4f}  (wrong_mapped={cand_sum.get('wrong_mapped')})", flush=True)
        print(f"Pred source counts             : {cand_sum.get('pred_source_counts')}", flush=True)
        print(f"GT source counts               : {cand_sum.get('gt_source_counts')}", flush=True)
        if cand_sum.get("mean_conf_over_mapped") is not None:
            print(f"Mean <CONF> over mapped        : {cand_sum.get('mean_conf_over_mapped')}", flush=True)
        if cand_sum.get("mean_conf_strong_over_mapped") is not None:
            print(f"Mean <CONF> strong             : {cand_sum.get('mean_conf_strong_over_mapped')}", flush=True)
        if cand_sum.get("mean_conf_nonstrong_over_mapped") is not None:
            print(f"Mean <CONF> non-strong         : {cand_sum.get('mean_conf_nonstrong_over_mapped')}", flush=True)

    print("\n[DONE]\n", flush=True)


if __name__ == "__main__":
    main()
