#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
now_narration_eval.py

Supports TWO modes via CLI:
  --now_narration_mode open   : original open-ended eval (STATE + VERB/NOUN/Pairs), unchanged logic
  --now_narration_mode cand   : candidate eval with TWO sub-tasks:
      (1) cand_state : evaluate STATE only (INTERACTION vs NO_INTERACTION) vs GT region (segment/gap)
          - If <STATE> tag not parsable to INTERACTION/NO_INTERACTION, fallback to clean_response:
              * if contains ONLY one of {"interaction", "no interaction"/"no_interaction"} -> take it
              * if both appear or neither appear -> invalid answer
          - Report valid answer ratio and conditional confusion matrix over valid answers.
          - NEW (qwen2_5_vl_7b only): dump 5 samples for (GT=INTERACTION & PRED=INTERACTION)
            and 5 for (GT=NO_INTERACTION & PRED=NO_INTERACTION) into ./temp/ as JSON,
            each entry is the full per-idx sample block from pred["samples"].

      (2) cand_mcq   : MCQ over interaction samples only (from shuffled mcq files)
          - Expect <ANS>A|B|C|D</ANS>
          - If missing/invalid:
              (a) recover a single A/B/C/D from clean_response (excluding <...>)
              (b) recover by matching verb+noun of exactly one option in clean_response
                  - if multi-answer ambiguous => invalid (no teacher)
              (c) fallback to OpenRouter teacher to map response -> A/B/C/D
          - strong distractor report using option_sources
              * strong distractors: option_sources in {"prev","future"}

Paths:
  GT (shared): pass explicitly with --gt_dir or EGOSAT_GT_DIR
  Taxonomy (open): pass explicitly with --taxonomy_json or EGOSAT_TAXONOMY_JSON

  OPEN pred dir:
    ~/benchmark_val/testllm/<MODEL_NAME>/now_narration/open

  CAND pred dirs:
    ~/benchmark_val/testllm/<MODEL_NAME>/now_narration/cand_state
    ~/benchmark_val/testllm/<MODEL_NAME>/now_narration/cand_mcq

  MCQ dir (cand_mcq):
    ~/benchmark_val/mcq_shuffled/now_narration_action/val

Outputs:
  open:
    ~/benchmark_val/score/<MODEL_NAME>/now_narration_open/{summary.json, details.jsonl, teacher_cache.json}
  cand_state:
    ~/benchmark_val/score/<MODEL_NAME>/now_narration_cand_state/{summary.json, details.jsonl}
    + (qwen2_5_vl_7b only) ./temp/{now_narration_cand_state_clean_interaction_5.json,
                                   now_narration_cand_state_clean_no_interaction_5.json}
  cand_mcq:
    ~/benchmark_val/score/<MODEL_NAME>/now_narration_cand_mcq/{summary.json, details.jsonl, teacher_cache.json}

Run:
  python3 now_narration_eval.py --now_narration_mode open
  python3 now_narration_eval.py --now_narration_mode cand
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
        finalize_source_stats,
        init_source_stats,
        source_stats_report_lines,
        update_source_stats,
    )
except ImportError:
    from mcq_source_stats import (
        finalize_source_stats,
        init_source_stats,
        source_stats_report_lines,
        update_source_stats,
    )


# -------------------------
# Hard-coded paths / knobs
# -------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini_pro").strip() or "gemini_pro"

GT_DIR = Path(os.environ.get("EGOSAT_GT_DIR", ""))
TAXONOMY_JSON = Path(os.environ.get("EGOSAT_TAXONOMY_JSON", ""))

# OPEN preds
PRED_OPEN_DIR = Path.home() / "benchmark_val" / "testllm" / MODEL_NAME / "now_narration" / "open"
OUT_OPEN_DIR = Path.home() / "benchmark_val" / "score" / MODEL_NAME / "now_narration_open"
OUT_OPEN_SUMMARY_JSON = OUT_OPEN_DIR / "summary.json"
OUT_OPEN_DETAILS_JSONL = OUT_OPEN_DIR / "details.jsonl"
OUT_OPEN_TEACHER_CACHE_JSON = OUT_OPEN_DIR / "teacher_cache.json"

# CAND preds
PRED_CAND_STATE_DIR = Path.home() / "benchmark_val" / "testllm" / MODEL_NAME / "now_narration" / "cand_state"
PRED_CAND_MCQ_DIR = Path.home() / "benchmark_val" / "testllm" / MODEL_NAME / "now_narration" / "cand_mcq"

OUT_CAND_STATE_DIR = Path.home() / "benchmark_val" / "score" / MODEL_NAME / "now_narration_cand_state"
OUT_CAND_STATE_SUMMARY_JSON = OUT_CAND_STATE_DIR / "summary.json"
OUT_CAND_STATE_DETAILS_JSONL = OUT_CAND_STATE_DIR / "details.jsonl"

OUT_CAND_MCQ_DIR = Path.home() / "benchmark_val" / "score" / MODEL_NAME / "now_narration_cand_mcq"
OUT_CAND_MCQ_SUMMARY_JSON = OUT_CAND_MCQ_DIR / "summary.json"
OUT_CAND_MCQ_DETAILS_JSONL = OUT_CAND_MCQ_DIR / "details.jsonl"
OUT_CAND_MCQ_TEACHER_CACHE_JSON = OUT_CAND_MCQ_DIR / "teacher_cache.json"

# MCQ shuffled dir. Defaults preserve the historical behavior; env/CLI can point
# scoring at a second frozen shuffle root for rebuttal runs.
DEFAULT_MCQ_DIR = Path(os.environ.get("EGOSAT_MCQ_DIR", ""))
MCQ_DIR = Path(os.environ.get("NOW_NARRATION_MCQ_DIR", str(DEFAULT_MCQ_DIR))).expanduser()
MCQ_DIR_EXPLICIT = bool(os.environ.get("NOW_NARRATION_MCQ_DIR", "").strip())
STRICT_MCQ_PATH = os.environ.get("STRICT_MCQ_PATH", "0").strip().lower() in {"1", "true", "yes", "y", "t"}

# float matching tolerance (t_eval)
T_EVAL_EPS = 1e-6

# Teacher (OpenRouter)
ENABLE_TEACHER_FALLBACK = os.environ.get("EGOSAT_ENABLE_LEGACY_TEACHER_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip() if ENABLE_TEACHER_FALLBACK else ""
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()
OPENROUTER_URL = os.environ.get("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions").strip()
OPENROUTER_TIMEOUT_SEC = int(float(os.environ.get("OPENROUTER_TIMEOUT_SEC", "60")))
OPENROUTER_RETRY = int(float(os.environ.get("OPENROUTER_RETRY", "2")))

# strong distractor definition (aligned with your generator option_sources)
STRONG_DISTURB_SOURCES = {"prev", "future"}


# -------------------------
# Regex / token helpers
# -------------------------
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_TAG_RE = {
    "state": re.compile(r"<\s*STATE\s*>(.*?)<\s*/\s*STATE\s*>", re.IGNORECASE | re.DOTALL),
    "verb": re.compile(r"<\s*VERB\s*>(.*?)<\s*/\s*VERB\s*>", re.IGNORECASE | re.DOTALL),
    "noun": re.compile(r"<\s*NOUN\s*>(.*?)<\s*/\s*NOUN\s*>", re.IGNORECASE | re.DOTALL),
    "ans": re.compile(r"<\s*ANS\s*>(.*?)<\s*/\s*ANS\s*>", re.IGNORECASE | re.DOTALL),
    "conf": re.compile(r"<\s*CONF\s*>(.*?)<\s*/\s*CONF\s*>", re.IGNORECASE | re.DOTALL),
}

# For cand_mcq local recovery
_ANGLE_BR_RE = re.compile(r"<[^>]*>")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _refresh_open_output_paths() -> None:
    global OUT_OPEN_SUMMARY_JSON, OUT_OPEN_DETAILS_JSONL, OUT_OPEN_TEACHER_CACHE_JSON
    OUT_OPEN_SUMMARY_JSON = OUT_OPEN_DIR / "summary.json"
    OUT_OPEN_DETAILS_JSONL = OUT_OPEN_DIR / "details.jsonl"
    OUT_OPEN_TEACHER_CACHE_JSON = OUT_OPEN_DIR / "teacher_cache.json"


def _refresh_cand_output_paths() -> None:
    global OUT_CAND_STATE_SUMMARY_JSON, OUT_CAND_STATE_DETAILS_JSONL
    global OUT_CAND_MCQ_SUMMARY_JSON, OUT_CAND_MCQ_DETAILS_JSONL, OUT_CAND_MCQ_TEACHER_CACHE_JSON
    OUT_CAND_STATE_SUMMARY_JSON = OUT_CAND_STATE_DIR / "summary.json"
    OUT_CAND_STATE_DETAILS_JSONL = OUT_CAND_STATE_DIR / "details.jsonl"
    OUT_CAND_MCQ_SUMMARY_JSON = OUT_CAND_MCQ_DIR / "summary.json"
    OUT_CAND_MCQ_DETAILS_JSONL = OUT_CAND_MCQ_DIR / "details.jsonl"
    OUT_CAND_MCQ_TEACHER_CACHE_JSON = OUT_CAND_MCQ_DIR / "teacher_cache.json"


def _infer_cand_pred_dirs_from_root(runs_root: Path, flavor: str) -> Tuple[Path, Path]:
    base = runs_root.expanduser() / MODEL_NAME / "now_narration"
    f = (flavor or "cand").strip()
    if f == "cand":
        return base / "cand_state", base / "cand_mcq"
    return base / f"{f}_state", base / f"{f}_mcq"


def _infer_cand_pred_dirs_from_explicit(pred_dir: Path) -> Tuple[Path, Path]:
    p = pred_dir.expanduser()
    if p.name == "cand_state":
        return p, p.parent / "cand_mcq"
    if p.name == "cand_mcq":
        return p.parent / "cand_state", p
    return p / "cand_state", p / "cand_mcq"


def _apply_cli_paths(
    *,
    mode: str,
    model_name: Optional[str],
    runs_root: Optional[Path],
    pred_dir: Optional[Path],
    out_dir: Optional[Path],
    flavor: str,
) -> None:
    global MODEL_NAME, PRED_OPEN_DIR, OUT_OPEN_DIR
    global PRED_CAND_STATE_DIR, PRED_CAND_MCQ_DIR, OUT_CAND_STATE_DIR, OUT_CAND_MCQ_DIR

    if model_name:
        MODEL_NAME = str(model_name).strip()

    if mode == "open":
        if pred_dir is not None:
            PRED_OPEN_DIR = pred_dir.expanduser()
        elif runs_root is not None:
            PRED_OPEN_DIR = runs_root.expanduser() / MODEL_NAME / "now_narration" / "open"
        else:
            PRED_OPEN_DIR = Path.home() / "benchmark_val" / "testllm" / MODEL_NAME / "now_narration" / "open"

        if out_dir is not None:
            OUT_OPEN_DIR = out_dir.expanduser()
        else:
            OUT_OPEN_DIR = Path.home() / "benchmark_val" / "score" / MODEL_NAME / "now_narration_open"
        _refresh_open_output_paths()
        return

    if pred_dir is not None:
        PRED_CAND_STATE_DIR, PRED_CAND_MCQ_DIR = _infer_cand_pred_dirs_from_explicit(pred_dir)
    elif runs_root is not None:
        PRED_CAND_STATE_DIR, PRED_CAND_MCQ_DIR = _infer_cand_pred_dirs_from_root(runs_root, flavor)
    else:
        PRED_CAND_STATE_DIR = Path.home() / "benchmark_val" / "testllm" / MODEL_NAME / "now_narration" / "cand_state"
        PRED_CAND_MCQ_DIR = Path.home() / "benchmark_val" / "testllm" / MODEL_NAME / "now_narration" / "cand_mcq"

    if out_dir is not None:
        root = out_dir.expanduser()
        OUT_CAND_STATE_DIR = root / "cand_state"
        OUT_CAND_MCQ_DIR = root / "cand_mcq"
    else:
        OUT_CAND_STATE_DIR = Path.home() / "benchmark_val" / "score" / MODEL_NAME / "now_narration_cand_state"
        OUT_CAND_MCQ_DIR = Path.home() / "benchmark_val" / "score" / MODEL_NAME / "now_narration_cand_mcq"
    _refresh_cand_output_paths()


def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def safe_load_json_or_one_jsonl(path: Path) -> Dict[str, Any]:
    """
    Accept:
      - .json: dict
      - .jsonl: must contain exactly ONE JSON object (either single-line JSONL or pretty JSON saved as .jsonl).
    """
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

    # jsonl fallback
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    recs = [json.loads(ln) for ln in lines]
    if len(recs) != 1 or not isinstance(recs[0], dict):
        raise ValueError(f"Expected exactly 1 JSON object in JSONL: {p}, got {len(recs)}")
    return recs[0]


def normalize_and_tokenize(text: str) -> List[str]:
    """
    Lowercase, replace '_' with space, keep only [a-z0-9]+ tokens.
    """
    s = (text or "").lower().replace("_", " ")
    return _WORD_RE.findall(s)


def tokens_to_underscore(toks: List[str]) -> str:
    toks = [t for t in toks if t]
    return "_".join(toks)


def _strip_underscores(s: str) -> str:
    return (s or "").strip().strip("_").strip()


def _extract_parenthetical_groups(term: str) -> Tuple[str, List[str]]:
    """
    Robustly handle terms like:
      - pen_(marker,_pen)
      - pepper_(vegetable)_(capsicum,_pepper)
      - jack_(tool)_(jack,_lift)
    Strategy:
      - base = substring before first '(' (trim trailing '_')
      - groups = all (...) contents in order
    """
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
    Expand taxonomy-style strings into variant surface forms.

    Rules:
      - base is always included if non-empty
      - any (...) group that contains comma is treated as synonym list
      - (...) groups without comma are treated as disambiguation notes and ignored
      - de-dup, keep order
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
        else:
            # disambiguation note: ignore
            pass

    for x in syns:
        if x and x not in variants:
            variants.append(x)

    # final cleanup
    out: List[str] = []
    seen = set()
    for v in variants:
        v = _strip_underscores(v)
        if not v:
            continue
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def variants_to_token_tuples(variants: List[str]) -> List[Tuple[str, ...]]:
    outs: List[Tuple[str, ...]] = []
    for v in variants:
        toks = normalize_and_tokenize(v)
        if toks:
            outs.append(tuple(toks))
    # de-dup keep order
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


# -------------------------
# Taxonomy structures (open mode)
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
# GT noun parsing from narration (open mode)
# -------------------------
_DET_SET = {
    "the", "a", "an",
    "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    "some", "any", "each", "every", "either", "neither",
    "few", "several", "many", "much", "no",
    "one", "two", "three", "four", "five",
    "both", "all",
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
    """
    "hold_(support,_grip,_grasp)" -> ["hold"]
    "wash_up" -> ["wash", "up"]
    """
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
    """
    occ: (start, length, gid)
    """
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
    """
    Returns (noun_candidate_set, diagnostics).
    noun_candidate_set is a set of underscore-joined variants, e.g. {"pen","marker"}.
    """
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

    # locate verb in narration
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

    # noun match scan (with simple plural -> singular if in noun vocab)
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
        # IMPORTANT: candidate set is ALL synonyms/variants for this taxonomy group
        return set(noun_groups[int(chosen_gid)].variant_strs), diag

    # fallback
    hn = heuristic_noun_after_verb(narr_tokens_raw, verb_pos, int(verb_len))
    if hn:
        diag["noun_parse"] = "heuristic_after_verb"
        diag["fallback_phrase"] = hn
        return {hn}, diag

    diag["noun_parse"] = "none"
    return {"none"}, diag


def gt_expand_verb_candidates(gt_verb: str) -> Tuple[List[Tuple[str, ...]], set]:
    """
    Expand GT verb string like hold_(support,_grip,_grasp) into candidates:
      {"hold","support","grip","grasp"}
    Returns (candidate_tuples, candidate_str_set)
    """
    vars_ = expand_taxonomy_variants(gt_verb)
    tuples_ = variants_to_token_tuples(vars_)
    strs_ = {tokens_to_underscore(list(t)) for t in tuples_}
    return tuples_, strs_


# -------------------------
# Prediction matching helpers (open mode)
# -------------------------
def normalize_state(x: Optional[str]) -> str:
    s = (x or "").strip().lower().replace(" ", "_")
    if not s:
        return "UNKNOWN"
    if "no_interaction" in s:
        return "NO_INTERACTION"
    if "interaction" in s:
        return "INTERACTION"
    if s in {"no", "none", "null", "0"}:
        return "NO_INTERACTION"
    if s in {"yes", "1"}:
        return "INTERACTION"
    return "UNKNOWN"


def infer_state_from_noun(pred_noun: str) -> str:
    """
    Per open-mode rule:
      - If no noun / noun == none -> NO_INTERACTION
      - Else -> INTERACTION
    """
    n = (pred_noun or "").strip().lower()
    if not n or n in {"none", "null", "no"}:
        return "NO_INTERACTION"
    return "INTERACTION"


def gt_state_from_region(sample: Dict[str, Any]) -> str:
    r = (sample.get("region") or "").strip().lower()
    if r == "segment":
        return "INTERACTION"
    if r == "gap":
        return "NO_INTERACTION"
    vi = sample.get("visible_interaction", None)
    if isinstance(vi, (int, float)):
        return "INTERACTION" if int(vi) == 1 else "NO_INTERACTION"
    return "UNKNOWN"


def normalize_pred_phrase_to_tokens(s: str) -> List[str]:
    return normalize_and_tokenize(s or "")


def stem_pred_verb_tokens(tokens: List[str], vocab: set) -> List[str]:
    return [_maybe_stem_verb_token(t, vocab) for t in tokens]


def singularize_pred_noun_tokens(tokens: List[str], noun_vocab: set) -> List[str]:
    return [_maybe_singularize_noun_token(t, noun_vocab) for t in tokens]


def verb_match(pred_verb: str, gt_candidate_tuples: List[Tuple[str, ...]], verb_vocab: set) -> bool:
    # hit if pred matches ANY candidate (set semantics)
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
    # hit if pred matches ANY candidate (set semantics)
    if (pred_noun or "").strip().lower() in {"none", "no", "null"}:
        return "none" in {str(x).strip().lower() for x in gt_noun_candidate_set}

    ptoks = normalize_pred_phrase_to_tokens(pred_noun)
    if not ptoks:
        return False
    ptoks = singularize_pred_noun_tokens(ptoks, noun_vocab)
    pstr = tokens_to_underscore(ptoks)

    gt_norm = {str(x).strip().lower() for x in gt_noun_candidate_set}
    return pstr.lower() in gt_norm


# -------------------------
# Teacher (OpenRouter) verb+noun extraction (OPEN MODE ONLY)
# -------------------------
def openrouter_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }


def teacher_extract_verb_noun(response_text: str) -> Tuple[str, str, Dict[str, Any]]:
    """
    OPEN MODE teacher:
    Extract main verb and main object noun phrase from response_text ONLY (no vision, no GT).
    IMPORTANT: teacher is allowed to output empty verb and/or empty noun.
    Returns: (verb, noun, diag)
    """
    diag: Dict[str, Any] = {
        "used": True,
        "model": OPENROUTER_MODEL,
        "ok": False,
        "raw_text": None,
        "error": None,
    }

    if not OPENROUTER_API_KEY:
        diag["error"] = "missing OPENROUTER_API_KEY"
        return "", "", diag

    sys_prompt = (
        "You extract structured labels from a student's textual answer.\n"
        "Given ONLY the student's response text, output a JSON object with exactly two keys:\n"
        '  {"verb": "...", "noun": "..."}\n'
        "Rules:\n"
        "- verb: the single main action verb in base form if possible (e.g., hold, move, open, take, wipe, walk).\n"
        "  If there is NO clear action verb, set verb to an empty string \"\".\n"
        "- noun: the main object noun or noun phrase (can be multi-word).\n"
        "  If there is NO clear object being interacted with, set noun to \"none\" (or \"\").\n"
        "- Output JSON only. No extra keys. No commentary."
    )

    user_prompt = (
        "Student response text:\n"
        "-----\n"
        f"{(response_text or '').strip()}\n"
        "-----\n"
        "Now output JSON."
    )

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 128,
    }

    last_err = None
    for attempt in range(OPENROUTER_RETRY + 1):
        try:
            r = requests.post(
                OPENROUTER_URL,
                headers=openrouter_headers(),
                json=payload,
                timeout=OPENROUTER_TIMEOUT_SEC,
            )
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:1000]}")
            resp = r.json()
            txt = resp["choices"][0]["message"]["content"]
            diag["raw_text"] = txt

            try:
                obj = json.loads(txt)
            except Exception:
                m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
                obj = json.loads(m.group(0)) if m else {}

            verb = str(obj.get("verb", "") if obj else "").strip()
            noun = str(obj.get("noun", "") if obj else "").strip()

            if noun.strip().lower() in {"", "null"}:
                noun = ""

            diag["ok"] = True
            return verb, noun, diag

        except Exception as e:
            last_err = e
            if attempt < OPENROUTER_RETRY:
                time.sleep(0.8 * (attempt + 1))
            else:
                diag["error"] = repr(last_err)
                return "", "", diag

    diag["error"] = repr(last_err)
    return "", "", diag


# -------------------------
# CAND helpers: state fallback parsing
# -------------------------
_NO_INTER_RE = re.compile(r"\bno[_ ]interaction\b", re.IGNORECASE)
# "interaction" that is NOT immediately preceded by "no_" or "no " (avoid NO_INTERACTION containing interaction)
_INTER_ONLY_RE = re.compile(r"(?<!no[_ ])\binteraction\b", re.IGNORECASE)


def cand_state_from_clean_text(clean_text: str) -> Tuple[str, str]:
    """
    Returns (state, method):
      state in {INTERACTION, NO_INTERACTION, INVALID}
      method in {"clean_no_interaction", "clean_interaction", "clean_ambiguous", "clean_none"}
    Rule per user:
      - if exactly one appears -> take it
      - if both appear or neither appear -> invalid
    """
    s = clean_text or ""
    has_no = _NO_INTER_RE.search(s) is not None
    has_inter = _INTER_ONLY_RE.search(s) is not None

    if has_no and has_inter:
        return "INVALID", "clean_ambiguous"
    if has_no:
        return "NO_INTERACTION", "clean_no_interaction"
    if has_inter:
        return "INTERACTION", "clean_interaction"
    return "INVALID", "clean_none"


# -------------------------
# CAND helpers: MCQ answer parsing (from sh_rtrv_eval logic)
# -------------------------
def _strip_angle_brackets(text: str) -> str:
    return _ANGLE_BR_RE.sub(" ", text or "")


def _guess_ans_letter_from_clean(clean_text: str) -> str:
    """
    Recover exactly ONE A/B/C/D from clean text AFTER removing <...>.
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

    if len(uniq) == 1:
        return uniq[0]
    return ""


def _parse_option_verb_noun(opt: str) -> Tuple[str, str]:
    """
    Options typically: "YOU <verb> <noun>".
    Return raw verb, raw noun (strings).
    """
    s = (opt or "").strip()
    if not s:
        return "", ""
    if s.upper().startswith("YOU "):
        s = s[4:].strip()
    parts = s.split()
    if len(parts) >= 2:
        return parts[0].strip(), parts[1].strip()
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
    return normalize_and_tokenize(s)


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

        if v_tok:
            verb_hit[i] = (v_tok in tokset)
        if n_base:
            noun_hit[i] = all(w in tokset for w in n_base)
        pair_hit[i] = bool(verb_hit[i] and noun_hit[i])

        opt_keys.append((v_tok, tokens_to_underscore(n_base) if n_base else ""))

    pair_idx = [i for i in range(4) if pair_hit[i]]
    if len(pair_idx) == 1:
        return "ABCD"[pair_idx[0]], False

    if len(pair_idx) > 1:
        keys = {opt_keys[i] for i in pair_idx}
        if len(keys) == 1:
            return "", False
        return "", True  # multi-answer invalid

    signal_idx = set([i for i in range(4) if verb_hit[i] or noun_hit[i]])
    if len(signal_idx) >= 2:
        keys = {opt_keys[i] for i in signal_idx}
        if len(keys) >= 2:
            return "", True
    return "", False


def normalize_ans_letter(x: str) -> str:
    s = (x or "").strip().upper()
    s = re.sub(r"[^A-D]", "", s)
    if len(s) >= 1 and s[0] in "ABCD":
        return s[0]
    return ""


def _ans_to_idx(ans: str) -> int:
    a = (ans or "").strip().upper()
    if not a or a[0] not in "ABCD":
        return -1
    return ord(a[0]) - ord("A")


def _inc_count(d: Dict[str, int], k: str, add: int = 1) -> None:
    d[k] = int(d.get(k, 0)) + int(add)


# -------------------------
# Teacher (OpenRouter) for CAND MCQ mapping
# -------------------------
def _openrouter_chat(system_prompt: str, user_prompt: str, max_tokens: int = 96) -> Tuple[str, Optional[str]]:
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
        try:
            r = requests.post(
                OPENROUTER_URL,
                headers=openrouter_headers(),
                json=payload,
                timeout=OPENROUTER_TIMEOUT_SEC,
            )
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:1000]}")
            resp = r.json()
            if isinstance(resp, dict) and "error" in resp:
                raise RuntimeError(f"OpenRouter error: {resp.get('error')}")
            txt = resp["choices"][0]["message"]["content"]
            return str(txt), None
        except Exception as e:
            last_err = repr(e)
            if attempt < OPENROUTER_RETRY:
                time.sleep(0.8 * (attempt + 1))
            else:
                return "", last_err
    return "", last_err or "unknown error"


def teacher_map_answer_to_option(response_text: str, options4: List[str]) -> Tuple[str, Dict[str, Any]]:
    """
    Teacher returns A/B/C/D (or empty only if extremely no evidence).
    Output JSON only: {"ans":"A"}
    """
    diag: Dict[str, Any] = {
        "used": True,
        "task": "map_ans",
        "model": OPENROUTER_MODEL,
        "ok": False,
        "raw_text": None,
        "error": None,
    }

    letters = ["A", "B", "C", "D"]
    opt_lines = []
    for i in range(min(4, len(options4))):
        opt_lines.append(f"{letters[i]}. {options4[i]}")
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
        "- If evidence is mixed or ambiguous, still choose the SINGLE best match; do NOT return empty just due to uncertainty.\n"
        "- Do NOT default to A. Choose A only when it is actually the best match.\n"
        "- Output JSON only. No commentary. No extra keys."
    )

    user_prompt = (
        "Options:\n" + opt_block + "\n\n"
        "Student response text:\n-----\n" + (response_text or "").strip() + "\n-----\n\n"
        "Now output JSON."
    )

    txt, err = _openrouter_chat(sys_prompt, user_prompt, max_tokens=96)
    diag["raw_text"] = txt
    diag["error"] = err
    if err:
        return "", diag

    parsed_json = False
    ans_raw = ""
    try:
        obj = json.loads(txt)
        parsed_json = True
        ans_raw = str(obj.get("ans", "") if obj else "")
    except Exception:
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
        if (ans_raw is None) or (str(ans_raw).strip() == ""):
            diag["ok"] = False
            return "", diag
        ans = normalize_ans_letter(str(ans_raw))
        if ans:
            diag["ok"] = True
            return ans, diag

    # salvage only if JSON parse failed or invalid non-empty
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
def build_gt_file_index(gt_dir: Path) -> Dict[Tuple[str, str, str], Path]:
    """
    Index GT files by (video_uid, clip_id, clip_uid) using file content metadata (robust).
    """
    out: Dict[Tuple[str, str, str], Path] = {}
    files = sorted(
        [p for p in gt_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}],
        key=lambda x: x.as_posix(),
    )
    for p in files:
        try:
            gt = safe_load_json_or_one_jsonl(p)
            video_uid = str(gt.get("video_uid", "")).strip()
            meta = gt.get("video_metadata", {}) if isinstance(gt.get("video_metadata"), dict) else {}
            clip_id = str(meta.get("clip_id", "")).strip()
            clip_uid = str(meta.get("clip_uid", "")).strip()
            if video_uid and clip_id and clip_uid:
                out[(video_uid, clip_id, clip_uid)] = p
        except Exception:
            continue
    return out


def _parse_triplet_from_filename(p: Path) -> Tuple[str, str, str]:
    """
    Try to parse (video_uid, clip_id, clip_uid) from filenames like:
      <model>__<video_uid>__<clip_id>__<clip_uid>__pred__cand_state.json
      <video_uid>__<clip_id>__<clip_uid>.jsonl
    """
    name = p.name
    parts = name.split("__")
    # For pred: model__video__clip_id__clip_uid__...
    if len(parts) >= 4 and re.fullmatch(r"[0-9]+", parts[2].strip()):
        return parts[1].strip(), parts[2].strip(), parts[3].split(".", 1)[0].strip()
    # For gt/mcq: video__clip_id__clip_uid(.jsonl)
    if len(parts) >= 3 and re.fullmatch(r"[0-9]+", parts[1].strip()):
        return parts[0].strip(), parts[1].strip(), parts[2].split(".", 1)[0].strip()
    return "", "", ""


def build_pred_file_list(pred_dir: Path) -> List[Path]:
    """
    Keep ONLY prediction files (*__pred*.json) and skip manifest/derived/queryset files.
    """
    if not pred_dir.exists():
        return []
    files = sorted([p for p in pred_dir.rglob("*.json") if p.is_file()], key=lambda x: x.as_posix())
    out: List[Path] = []
    for p in files:
        name = p.name.lower()
        if "manifest" in name:
            continue
        if name.endswith("__pred.json") or name.endswith("_pred.json") or "__pred" in name:
            out.append(p)
    return out


def build_mcq_index(mcq_dir: Path) -> Dict[Tuple[str, str, str], Path]:
    out: Dict[Tuple[str, str, str], Path] = {}
    files = sorted(
        [p for p in mcq_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}],
        key=lambda x: x.as_posix(),
    )
    for p in files:
        v, c, u = _parse_triplet_from_filename(p)
        if v and c and u:
            out[(v, c, u)] = p
            continue
        # fallback: read content if needed
        try:
            obj = safe_load_json_or_one_jsonl(p)
            video_uid = str(obj.get("video_uid", "")).strip()
            clip_id = str(obj.get("clip_id", "")).strip()
            clip_uid = ""
            gt_file = str(obj.get("gt_file", "")).strip()
            if gt_file:
                _, _, u2 = _parse_triplet_from_filename(Path(gt_file))
                clip_uid = u2
            if video_uid and clip_id and clip_uid:
                out[(video_uid, clip_id, clip_uid)] = p
        except Exception:
            continue
    return out


def resolve_queryset_path_from_pred(pred: Dict[str, Any]) -> Optional[str]:
    """
    Prefer the effective queryset recorded by runner/adapter, if present.
    This lets second-shuffle scoring use queryset.samples[*].mcq.answer instead
    of falling back to a global MCQ_DIR.
    """
    v = pred.get("source_queryset", None)
    if isinstance(v, str) and v.strip():
        return v.strip()
    vm = pred.get("video_metadata", {}) if isinstance(pred.get("video_metadata"), dict) else {}
    v2 = vm.get("queryset_path", None)
    if isinstance(v2, str) and v2.strip():
        return v2.strip()
    return None


def build_effective_mcq_by_idx(qs: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    samples = qs.get("samples", [])
    if not isinstance(samples, list):
        return out
    for s in samples:
        if not isinstance(s, dict):
            continue
        idx = s.get("idx", None)
        if not isinstance(idx, (int, float)):
            continue
        mcq = s.get("mcq", {}) if isinstance(s.get("mcq"), dict) else {}
        if not mcq:
            continue
        item = dict(mcq)
        item.setdefault("idx", int(idx))
        out[int(idx)] = item
    return out


def pretty_confusion(cm: Dict[str, Dict[str, int]]) -> str:
    rows = ["INTERACTION", "NO_INTERACTION", "UNKNOWN"]
    cols = ["INTERACTION", "NO_INTERACTION", "UNKNOWN"]
    lines = []
    header = "GT\\PRED".ljust(14) + "".join([c.rjust(16) for c in cols])
    lines.append(header)
    for r in rows:
        line = r.ljust(14)
        for c in cols:
            line += str(cm.get(r, {}).get(c, 0)).rjust(16)
        lines.append(line)
    return "\n".join(lines)


def pretty_confusion_2x2(cm: Dict[str, Dict[str, int]]) -> str:
    rows = ["INTERACTION", "NO_INTERACTION"]
    cols = ["INTERACTION", "NO_INTERACTION"]
    lines = []
    header = "GT\\PRED".ljust(14) + "".join([c.rjust(16) for c in cols])
    lines.append(header)
    for r in rows:
        line = r.ljust(14)
        for c in cols:
            line += str(cm.get(r, {}).get(c, 0)).rjust(16)
        lines.append(line)
    return "\n".join(lines)


# -------------------------
# OPEN evaluation (original logic; moved into a function, unchanged)
# -------------------------
def eval_open() -> None:
    _ensure_dir(OUT_OPEN_DIR)

    if not GT_DIR.exists():
        raise FileNotFoundError(f"GT_DIR not found: {GT_DIR}")
    if not TAXONOMY_JSON.is_file():
        raise FileNotFoundError(f"TAXONOMY_JSON not found: {TAXONOMY_JSON}")
    if not PRED_OPEN_DIR.exists():
        raise FileNotFoundError(f"PRED_DIR not found: {PRED_OPEN_DIR}")

    tax = load_taxonomy(TAXONOMY_JSON)
    noun_groups, noun_var_tuple_to_gid, noun_lengths_desc, noun_token_vocab = build_noun_groups(tax["nouns"])
    verb_token_vocab = build_verb_token_vocab(tax.get("verbs", []))

    gt_index = build_gt_file_index(GT_DIR)
    pred_files = build_pred_file_list(PRED_OPEN_DIR)

    # Teacher cache
    teacher_cache: Dict[str, Dict[str, str]] = {}
    if OUT_OPEN_TEACHER_CACHE_JSON.exists():
        try:
            teacher_cache = json.loads(OUT_OPEN_TEACHER_CACHE_JSON.read_text(encoding="utf-8"))
            if not isinstance(teacher_cache, dict):
                teacher_cache = {}
        except Exception:
            teacher_cache = {}

    # Metrics accumulators
    cm: Dict[str, Dict[str, int]] = {}

    def cm_add(gt_s: str, pr_s: str) -> None:
        cm.setdefault(gt_s, {})
        cm[gt_s][pr_s] = int(cm[gt_s].get(pr_s, 0)) + 1

    num_files_total = 0
    num_files_matched = 0
    num_samples_total = 0

    state_correct = 0
    state_total = 0

    gt_inter_total = 0
    verb_correct = 0
    noun_correct = 0
    pair_correct = 0

    # conditional (only when state correct AND gt is interaction)
    gt_inter_state_ok = 0
    verb_correct_cond = 0
    noun_correct_cond = 0
    pair_correct_cond = 0

    missing_gt_files = 0
    missing_gt_samples = 0
    pred_missing_state = 0
    pred_missing_verb_tag = 0
    pred_missing_noun_tag = 0
    teacher_calls = 0
    teacher_cache_hits = 0
    teacher_fail = 0

    # output details
    if OUT_OPEN_DETAILS_JSONL.exists():
        OUT_OPEN_DETAILS_JSONL.unlink(missing_ok=True)

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

        # build gt maps
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

        pred_samples = pred.get("samples", [])
        if not isinstance(pred_samples, list):
            continue

        num_files_matched += 1

        for ps in pred_samples:
            if not isinstance(ps, dict):
                continue
            num_samples_total += 1

            p_idx = ps.get("idx", None)
            p_t = ps.get("t_eval", None)

            gt_s = None
            if isinstance(p_idx, (int, float)) and int(p_idx) in gt_by_idx:
                gt_s = gt_by_idx[int(p_idx)]
            elif isinstance(p_t, (int, float)):
                if float(p_t) in gt_by_t:
                    gt_s = gt_by_t[float(p_t)]
                else:
                    tt = float(p_t)
                    best = None
                    best_d = 1e9
                    for t0, s0 in gt_by_t.items():
                        d = abs(t0 - tt)
                        if d < best_d:
                            best_d = d
                            best = s0
                    if best is not None and best_d <= T_EVAL_EPS:
                        gt_s = best

            if gt_s is None:
                missing_gt_samples += 1
                continue

            gt_state = gt_state_from_region(gt_s)
            gt_region = (gt_s.get("region") or "").strip().lower()

            resp_text = str(ps.get("response_text", "") or "")

            # parse tags
            pred_state_raw = parse_tag(resp_text, "state")
            pred_verb_raw = parse_tag(resp_text, "verb")
            pred_noun_raw = parse_tag(resp_text, "noun")

            if pred_state_raw is None:
                pred_missing_state += 1
            if pred_verb_raw is None:
                pred_missing_verb_tag += 1
            if pred_noun_raw is None:
                pred_missing_noun_tag += 1

            pred_verb = (pred_verb_raw or "").strip()
            pred_noun = (pred_noun_raw or "").strip()

            used_teacher = False
            teacher_diag = None

            # If verb/noun missing, call teacher
            need_teacher_for_verbnoun = (pred_verb_raw is None) or (pred_noun_raw is None)

            if need_teacher_for_verbnoun:
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

            # State:
            if pred_state_raw is not None:
                pred_state = normalize_state(pred_state_raw)
            else:
                pred_state = infer_state_from_noun(pred_noun)

            cm_add(gt_state, pred_state)

            state_total += 1
            if pred_state == gt_state and gt_state != "UNKNOWN":
                state_correct += 1

            # Only evaluate verb/noun/pair on GT interaction samples
            if gt_state != "INTERACTION":
                detail = {
                    "file": pf.name,
                    "video_uid": video_uid,
                    "clip_id": clip_id,
                    "clip_uid": clip_uid,
                    "idx": int(p_idx) if isinstance(p_idx, (int, float)) else None,
                    "t_eval": float(p_t) if isinstance(p_t, (int, float)) else None,
                    "gt_region": gt_region,
                    "gt_state": gt_state,
                    "pred_state": pred_state,
                    "used_teacher": used_teacher,
                    "teacher_diag": teacher_diag,
                    "verb_eval": None,
                    "noun_eval": None,
                    "pair_eval": None,
                }
                with open(OUT_OPEN_DETAILS_JSONL, "a", encoding="utf-8") as f:
                    f.write(json.dumps(detail, ensure_ascii=False) + "\n")
                continue

            gt_inter_total += 1

            # GT verb candidates (synonym set)
            gt_verb = str(gt_s.get("gt_verb", "") or "").strip()
            gt_verb_cand_tuples, gt_verb_cand_strs = gt_expand_verb_candidates(gt_verb)

            # GT noun candidates (synonym set from taxonomy group matched in narration)
            gt_text_narr = str(gt_s.get("gt_text_narr", "") or "")
            gt_noun_cands, gt_noun_diag = gt_extract_noun_candidates_from_narration(
                gt_text_narr=gt_text_narr,
                gt_verb=gt_verb,
                noun_groups=noun_groups,
                noun_var_tuple_to_gid=noun_var_tuple_to_gid,
                noun_lengths_desc=noun_lengths_desc,
                noun_token_vocab=noun_token_vocab,
                verb_token_vocab=verb_token_vocab,
            )

            # STRICT: if state is wrong for an interaction GT sample, count all as wrong
            if pred_state != "INTERACTION":
                v_hit = False
                n_hit = False
                p_hit = False
            else:
                local_verb_vocab = set(verb_token_vocab)
                for t in gt_verb_cand_tuples:
                    for w in t:
                        local_verb_vocab.add(w)
                v_hit = verb_match(pred_verb, gt_verb_cand_tuples, local_verb_vocab)

                local_noun_vocab = set(noun_token_vocab)
                for s0 in gt_noun_cands:
                    for w in normalize_and_tokenize(str(s0)):
                        local_noun_vocab.add(w)
                n_hit = noun_match(pred_noun, gt_noun_cands, local_noun_vocab)

                p_hit = bool(v_hit and n_hit)

            if v_hit:
                verb_correct += 1
            if n_hit:
                noun_correct += 1
            if p_hit:
                pair_correct += 1

            if pred_state == "INTERACTION":
                gt_inter_state_ok += 1
                if v_hit:
                    verb_correct_cond += 1
                if n_hit:
                    noun_correct_cond += 1
                if p_hit:
                    pair_correct_cond += 1

            detail = {
                "file": pf.name,
                "video_uid": video_uid,
                "clip_id": clip_id,
                "clip_uid": clip_uid,
                "idx": int(p_idx) if isinstance(p_idx, (int, float)) else None,
                "t_eval": float(p_t) if isinstance(p_t, (int, float)) else None,
                "gt_region": gt_region,
                "gt_state": gt_state,
                "pred_state": pred_state,
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
            with open(OUT_OPEN_DETAILS_JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps(detail, ensure_ascii=False) + "\n")

    # persist teacher cache
    OUT_OPEN_TEACHER_CACHE_JSON.write_text(json.dumps(teacher_cache, ensure_ascii=False, indent=2), encoding="utf-8")

    # compute final metrics
    state_acc = (state_correct / state_total) if state_total > 0 else 0.0
    verb_acc = (verb_correct / gt_inter_total) if gt_inter_total > 0 else 0.0
    noun_acc = (noun_correct / gt_inter_total) if gt_inter_total > 0 else 0.0
    pair_acc = (pair_correct / gt_inter_total) if gt_inter_total > 0 else 0.0

    verb_acc_cond = (verb_correct_cond / gt_inter_state_ok) if gt_inter_state_ok > 0 else 0.0
    noun_acc_cond = (noun_correct_cond / gt_inter_state_ok) if gt_inter_state_ok > 0 else 0.0
    pair_acc_cond = (pair_correct_cond / gt_inter_state_ok) if gt_inter_state_ok > 0 else 0.0

    summary = {
        "model_name": MODEL_NAME,
        "task": "now_narration_open",
        "gt_dir": str(GT_DIR),
        "pred_dir": str(PRED_OPEN_DIR),
        "taxonomy_json": str(TAXONOMY_JSON),
        "num_pred_files_total": int(num_files_total),
        "num_pred_files_matched_gt": int(num_files_matched),
        "missing_gt_files": int(missing_gt_files),
        "missing_gt_samples": int(missing_gt_samples),
        "num_pred_samples_evaluated": int(num_samples_total),
        "state_confusion": cm,
        "state_accuracy": float(state_acc),
        "gt_interaction_samples": int(gt_inter_total),
        "verb_accuracy_strict": float(verb_acc),
        "noun_accuracy_strict": float(noun_acc),
        "pair_accuracy_strict": float(pair_acc),
        "gt_interaction_state_predicted_interaction": int(gt_inter_state_ok),
        "verb_accuracy_cond_on_state_correct": float(verb_acc_cond),
        "noun_accuracy_cond_on_state_correct": float(noun_acc_cond),
        "pair_accuracy_cond_on_state_correct": float(pair_acc_cond),
        "pred_missing_state_tag": int(pred_missing_state),
        "pred_missing_verb_tag": int(pred_missing_verb_tag),
        "pred_missing_noun_tag": int(pred_missing_noun_tag),
        "teacher_calls": int(teacher_calls),
        "teacher_cache_hits": int(teacher_cache_hits),
        "teacher_fail": int(teacher_fail),
        "openrouter_model": OPENROUTER_MODEL,
        "timestamp_unix": time.time(),
    }

    OUT_OPEN_SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # console print
    print("\n================ NOW_NARRATION OPEN EVAL ================\n")
    print(f"MODEL_NAME: {MODEL_NAME}")
    print(f"GT_DIR    : {GT_DIR}")
    print(f"PRED_DIR  : {PRED_OPEN_DIR}")
    print(f"OUT_DIR   : {OUT_OPEN_DIR}")
    print("")
    print(f"Pred files total        : {num_files_total}")
    print(f"Pred files matched GT   : {num_files_matched}")
    print(f"Missing GT files        : {missing_gt_files}")
    print(f"Missing GT samples      : {missing_gt_samples}")
    print(f"Pred samples evaluated  : {num_samples_total}")
    print("")
    print("---- State confusion (GT region->state vs pred state) ----")
    print(pretty_confusion(cm))
    print(f"\nState accuracy: {state_acc:.4f} ({state_correct}/{state_total})")
    print("")
    print("---- Verb/Noun/Pair (STRICT; only GT INTERACTION, state mismatch => counts wrong) ----")
    print(f"GT interaction samples: {gt_inter_total}")
    print(f"Verb acc  : {verb_acc:.4f} ({verb_correct}/{gt_inter_total})")
    print(f"Noun acc  : {noun_acc:.4f} ({noun_correct}/{gt_inter_total})")
    print(f"Pair acc  : {pair_acc:.4f} ({pair_correct}/{gt_inter_total})")
    print("")
    print("---- Conditional (given pred_state==INTERACTION on GT INTERACTION) ----")
    print(f"Interaction samples with pred_state==INTERACTION: {gt_inter_state_ok}")
    print(f"Verb acc|state_ok : {verb_acc_cond:.4f} ({verb_correct_cond}/{gt_inter_state_ok})")
    print(f"Noun acc|state_ok : {noun_acc_cond:.4f} ({noun_correct_cond}/{gt_inter_state_ok})")
    print(f"Pair acc|state_ok : {pair_acc_cond:.4f} ({pair_correct_cond}/{gt_inter_state_ok})")
    print("")
    print("---- Extraction diagnostics ----")
    print(f"Missing <STATE> tags: {pred_missing_state}")
    print(f"Missing <VERB>  tags: {pred_missing_verb_tag}")
    print(f"Missing <NOUN>  tags: {pred_missing_noun_tag}")
    print(f"Teacher calls        : {teacher_calls}")
    print(f"Teacher cache hits   : {teacher_cache_hits}")
    print(f"Teacher failures     : {teacher_fail}")
    print("")
    print(f"[WROTE] {OUT_OPEN_SUMMARY_JSON}")
    print(f"[WROTE] {OUT_OPEN_DETAILS_JSONL}")
    print(f"[WROTE] {OUT_OPEN_TEACHER_CACHE_JSON}")
    print("\n=========================================================\n")


# -------------------------
# CAND: cand_state evaluation
# -------------------------
def eval_cand_state() -> None:
    _ensure_dir(OUT_CAND_STATE_DIR)

    if not GT_DIR.exists():
        raise FileNotFoundError(f"GT_DIR not found: {GT_DIR}")
    if not PRED_CAND_STATE_DIR.exists():
        raise FileNotFoundError(f"PRED_CAND_STATE_DIR not found: {PRED_CAND_STATE_DIR}")

    gt_index = build_gt_file_index(GT_DIR)
    pred_files = build_pred_file_list(PRED_CAND_STATE_DIR)

    if OUT_CAND_STATE_DETAILS_JSONL.exists():
        OUT_CAND_STATE_DETAILS_JSONL.unlink(missing_ok=True)

    num_files_total = 0
    num_files_matched = 0
    missing_gt_files = 0
    missing_gt_samples = 0

    total_samples = 0          # samples with matched GT
    valid_samples = 0          # samples with valid pred state
    invalid_samples = 0        # invalid pred state
    cm_valid: Dict[str, Dict[str, int]] = {}

    def cm_add(gt_s: str, pr_s: str) -> None:
        cm_valid.setdefault(gt_s, {})
        cm_valid[gt_s][pr_s] = int(cm_valid[gt_s].get(pr_s, 0)) + 1

    # NEW (qwen2_5_vl_7b only): collect examples where (GT==PRED) for INTERACTION and NO_INTERACTION
    dump_enabled = (MODEL_NAME == "qwen2_5_vl_7b")
    match_interaction_blocks: List[Dict[str, Any]] = []
    match_no_interaction_blocks: List[Dict[str, Any]] = []

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

        if not (video_uid and clip_id and clip_uid):
            v2, c2, u2 = _parse_triplet_from_filename(pf)
            video_uid = video_uid or v2
            clip_id = clip_id or c2
            clip_uid = clip_uid or u2

        key = (video_uid, clip_id, clip_uid)
        gt_path = gt_index.get(key)
        if gt_path is None:
            missing_gt_files += 1
            continue

        gt = safe_load_json_or_one_jsonl(gt_path)
        gt_samples = gt.get("samples", [])
        if not isinstance(gt_samples, list):
            gt_samples = []

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

        pred_samples = pred.get("samples", [])
        if not isinstance(pred_samples, list):
            continue

        num_files_matched += 1

        for ps in pred_samples:
            if not isinstance(ps, dict):
                continue

            p_idx = ps.get("idx", None)
            p_t = ps.get("t_eval", None)

            gt_s = None
            if isinstance(p_idx, (int, float)) and int(p_idx) in gt_by_idx:
                gt_s = gt_by_idx[int(p_idx)]
            elif isinstance(p_t, (int, float)):
                tt = float(p_t)
                if tt in gt_by_t:
                    gt_s = gt_by_t[tt]
                else:
                    best = None
                    best_d = 1e9
                    for t0, s0 in gt_by_t.items():
                        d = abs(t0 - tt)
                        if d < best_d:
                            best_d = d
                            best = s0
                    if best is not None and best_d <= T_EVAL_EPS:
                        gt_s = best

            if gt_s is None:
                missing_gt_samples += 1
                continue

            total_samples += 1
            gt_state = gt_state_from_region(gt_s)
            gt_region = (gt_s.get("region") or "").strip().lower()

            resp_text = str(ps.get("response_text", "") or "")
            clean_text = str(ps.get("clean_response", "") or ps.get("clean", "") or resp_text or "")

            pred_state_raw = parse_tag(resp_text, "state")
            pred_state = "INVALID"
            pred_method = "invalid"

            if pred_state_raw is not None:
                ns = normalize_state(pred_state_raw)
                if ns in {"INTERACTION", "NO_INTERACTION"}:
                    pred_state = ns
                    pred_method = "tag"
                else:
                    # fallback to clean
                    ps2, m2 = cand_state_from_clean_text(clean_text)
                    pred_state = ps2
                    pred_method = m2
            else:
                # fallback to clean
                ps2, m2 = cand_state_from_clean_text(clean_text)
                pred_state = ps2
                pred_method = m2

            is_valid = pred_state in {"INTERACTION", "NO_INTERACTION"}
            if is_valid:
                valid_samples += 1
                cm_add(gt_state, pred_state)
            else:
                invalid_samples += 1

            # NEW: dump sample blocks (qwen2_5_vl_7b only) for the matrix true-positive/true-negative cells
            # i.e., (GT=INTERACTION & PRED=INTERACTION) and (GT=NO_INTERACTION & PRED=NO_INTERACTION)
            if dump_enabled and is_valid and gt_state in {"INTERACTION", "NO_INTERACTION"} and pred_state == gt_state:
                if gt_state == "INTERACTION" and len(match_interaction_blocks) < 5:
                    match_interaction_blocks.append(ps)
                elif gt_state == "NO_INTERACTION" and len(match_no_interaction_blocks) < 5:
                    match_no_interaction_blocks.append(ps)

            detail = {
                "file": pf.name,
                "video_uid": video_uid,
                "clip_id": clip_id,
                "clip_uid": clip_uid,
                "idx": int(p_idx) if isinstance(p_idx, (int, float)) else None,
                "t_eval": float(p_t) if isinstance(p_t, (int, float)) else None,
                "gt_region": gt_region,
                "gt_state": gt_state,
                "pred_state_raw": pred_state_raw,
                "pred_state": pred_state,
                "pred_state_method": pred_method,
                "valid": bool(is_valid),
            }
            with open(OUT_CAND_STATE_DETAILS_JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps(detail, ensure_ascii=False) + "\n")

    # NEW: write dumped blocks to ./temp (qwen2_5_vl_7b only)
    if dump_enabled:
        temp_dir = Path.cwd() / "temp"
        _ensure_dir(temp_dir)
        out_a = temp_dir / "now_narration_cand_state_clean_interaction_5.json"
        out_b = temp_dir / "now_narration_cand_state_clean_no_interaction_5.json"
        out_a.write_text(json.dumps(match_interaction_blocks, ensure_ascii=False, indent=2), encoding="utf-8")
        out_b.write_text(json.dumps(match_no_interaction_blocks, ensure_ascii=False, indent=2), encoding="utf-8")

    valid_ratio = (valid_samples / total_samples) if total_samples > 0 else 0.0

    summary = {
        "model_name": MODEL_NAME,
        "task": "now_narration_cand_state",
        "gt_dir": str(GT_DIR),
        "pred_dir": str(PRED_CAND_STATE_DIR),
        "out_dir": str(OUT_CAND_STATE_DIR),
        "num_pred_files_total": int(num_files_total),
        "num_pred_files_matched_gt": int(num_files_matched),
        "missing_gt_files": int(missing_gt_files),
        "missing_gt_samples": int(missing_gt_samples),
        "num_samples_evaluated_with_gt": int(total_samples),
        "num_valid_answers": int(valid_samples),
        "num_invalid_answers": int(invalid_samples),
        "valid_answer_ratio": float(valid_ratio),
        "confusion_valid_only": cm_valid,
        "timestamp_unix": time.time(),
    }

    OUT_CAND_STATE_SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n================ NOW_NARRATION CAND_STATE EVAL ================\n")
    print(f"MODEL_NAME: {MODEL_NAME}")
    print(f"GT_DIR    : {GT_DIR}")
    print(f"PRED_DIR  : {PRED_CAND_STATE_DIR}")
    print(f"OUT_DIR   : {OUT_CAND_STATE_DIR}")
    print("")
    print(f"Pred files total        : {num_files_total}")
    print(f"Pred files matched GT   : {num_files_matched}")
    print(f"Missing GT files        : {missing_gt_files}")
    print(f"Missing GT samples      : {missing_gt_samples}")
    print(f"Samples evaluated (GT matched): {total_samples}")
    print(f"Valid answers           : {valid_samples}")
    print(f"Invalid answers         : {invalid_samples}")
    print(f"Valid answer ratio      : {valid_ratio:.4f}")
    print("")
    print("---- Confusion (VALID answers only) ----")
    print(pretty_confusion_2x2(cm_valid))
    print("")
    if dump_enabled:
        print("---- Dumped GT==PRED blocks (qwen2_5_vl_7b only) ----")
        print(f"[WROTE] {Path.cwd() / 'temp' / 'now_narration_cand_state_clean_interaction_5.json'}")
        print(f"[WROTE] {Path.cwd() / 'temp' / 'now_narration_cand_state_clean_no_interaction_5.json'}")
        print(f"Collected GT=INTERACTION & PRED=INTERACTION       : {len(match_interaction_blocks)}/5")
        print(f"Collected GT=NO_INTERACTION & PRED=NO_INTERACTION : {len(match_no_interaction_blocks)}/5")
        print("")
    print(f"[WROTE] {OUT_CAND_STATE_SUMMARY_JSON}")
    print(f"[WROTE] {OUT_CAND_STATE_DETAILS_JSONL}")
    print("\n==============================================================\n")


# -------------------------
# CAND: cand_mcq evaluation
# -------------------------
def eval_cand_mcq() -> None:
    _ensure_dir(OUT_CAND_MCQ_DIR)

    if not PRED_CAND_MCQ_DIR.exists():
        raise FileNotFoundError(f"PRED_CAND_MCQ_DIR not found: {PRED_CAND_MCQ_DIR}")
    if not MCQ_DIR.exists():
        raise FileNotFoundError(f"MCQ_DIR not found: {MCQ_DIR}")

    pred_files = build_pred_file_list(PRED_CAND_MCQ_DIR)
    mcq_index = build_mcq_index(MCQ_DIR)
    queryset_cache: Dict[str, Dict[str, Any]] = {}

    teacher_cache: Dict[str, Dict[str, Any]] = {}
    if OUT_CAND_MCQ_TEACHER_CACHE_JSON.exists():
        try:
            teacher_cache = json.loads(OUT_CAND_MCQ_TEACHER_CACHE_JSON.read_text(encoding="utf-8"))
            if not isinstance(teacher_cache, dict):
                teacher_cache = {}
        except Exception:
            teacher_cache = {}

    if OUT_CAND_MCQ_DETAILS_JSONL.exists():
        OUT_CAND_MCQ_DETAILS_JSONL.unlink(missing_ok=True)

    num_files_total = 0
    num_files_matched_mcq = 0
    missing_mcq_files = 0
    missing_mcq_samples = 0

    total_samples = 0
    valid_answers = 0
    invalid_answers = 0
    correct_over_all = 0
    correct_over_valid = 0

    pred_missing_ans_tag_or_invalid = 0
    teacher_calls = 0
    teacher_cache_hits = 0
    teacher_fail = 0

    # strong distractor stats (computed on samples where we can map pred_ans to option_sources)
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

        if not (video_uid and clip_id and clip_uid):
            v2, c2, u2 = _parse_triplet_from_filename(pf)
            video_uid = video_uid or v2
            clip_id = clip_id or c2
            clip_uid = clip_uid or u2

        mcq_by_idx: Dict[int, Dict[str, Any]] = {}
        mcq_path: Optional[Path] = None
        qpath = resolve_queryset_path_from_pred(pred)
        mcq_source = "mcq_dir"

        if qpath:
            qp = Path(qpath).expanduser()
            if qp.is_file():
                try:
                    if qpath in queryset_cache:
                        qs = queryset_cache[qpath]
                    else:
                        qs = safe_load_json_or_one_jsonl(qp)
                        queryset_cache[qpath] = qs
                    mcq_by_idx = build_effective_mcq_by_idx(qs)
                    if mcq_by_idx:
                        mcq_source = "effective_queryset"
                except Exception:
                    if STRICT_MCQ_PATH:
                        raise
                    mcq_by_idx = {}
            elif STRICT_MCQ_PATH:
                raise FileNotFoundError(f"effective queryset path from prediction not found: {qp}")

        if not mcq_by_idx:
            if STRICT_MCQ_PATH and not MCQ_DIR_EXPLICIT:
                raise FileNotFoundError(
                    "STRICT_MCQ_PATH=1 requires an effective queryset or explicit --mcq-dir/NOW_NARRATION_MCQ_DIR; "
                    f"refusing implicit fallback to default MCQ_DIR={MCQ_DIR}"
                )
            mcq_path = mcq_index.get((video_uid, clip_id, clip_uid))
            if mcq_path is None or (not Path(mcq_path).is_file()):
                missing_mcq_files += 1
                continue

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

        pred_samples = pred.get("samples", [])
        if not isinstance(pred_samples, list):
            continue

        num_files_matched_mcq += 1
        _inc_count(mcq_source_counts, mcq_source, 1)

        for ps in pred_samples:
            if not isinstance(ps, dict):
                continue
            total_samples += 1

            p_idx = ps.get("idx", None)
            if not isinstance(p_idx, (int, float)) or int(p_idx) not in mcq_by_idx:
                missing_mcq_samples += 1
                continue

            ms = mcq_by_idx[int(p_idx)]

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
            pred_ans_method = "tag" if pred_ans else "none"

            used_teacher = False
            teacher_diag = None

            if pred_ans_raw is None or not pred_ans:
                pred_missing_ans_tag_or_invalid += 1

                # (1) recover a single letter
                a1 = _guess_ans_letter_from_clean(clean_text)
                if a1:
                    pred_ans = a1
                    pred_ans_method = "clean_letter"
                else:
                    # (2) recover by option verb+noun signals
                    a2, is_multi = _guess_ans_from_option_terms(clean_text, options4)
                    if is_multi:
                        pred_ans = ""
                        pred_ans_method = "multi_answer_invalid"
                    elif a2:
                        pred_ans = a2
                        pred_ans_method = "verbnoun_match"
                    else:
                        # (3) teacher fallback
                        used_teacher = True
                        pred_ans_method = "teacher"

                        opt_join = "\n".join([f"{i}:{str(o)}" for i, o in enumerate(options4)])
                        h = _sha1(resp_text + "\n" + opt_join)

                        if h in teacher_cache and isinstance(teacher_cache[h], dict):
                            teacher_cache_hits += 1
                            pred_ans = normalize_ans_letter(str(teacher_cache[h].get("ans", "")))
                        else:
                            teacher_calls += 1
                            a3, diag3 = teacher_map_answer_to_option(resp_text, options4)
                            teacher_diag = diag3
                            if diag3.get("error"):
                                teacher_fail += 1
                            pred_ans = normalize_ans_letter(a3)
                            teacher_cache[h] = {"ans": pred_ans, "options": options4}

            is_valid = bool(pred_ans and pred_ans in {"A", "B", "C", "D"} and gt_ans)
            hit = bool(is_valid and pred_ans == gt_ans)

            if is_valid:
                valid_answers += 1
                if hit:
                    correct_over_valid += 1
            else:
                invalid_answers += 1

            if hit:
                correct_over_all += 1

            source_info = update_source_stats(
                source_stats,
                pred_ans,
                gt_ans,
                ms.get("answer_idx", None),
                option_sources,
                bool(is_valid),
            )

            # strong distractor stats (only when ANS -> option_sources is possible)
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

            detail = {
                "file": pf.name,
                "video_uid": video_uid,
                "clip_id": clip_id,
                "clip_uid": clip_uid,
                "mcq_source": mcq_source,
                "queryset_path": qpath,
                "mcq_path": str(mcq_path) if mcq_path is not None else None,
                "idx": int(p_idx) if isinstance(p_idx, (int, float)) else None,
                "t_eval": float(ps.get("t_eval")) if isinstance(ps.get("t_eval"), (int, float)) else None,
                "gt_ans": gt_ans,
                "pred_ans_raw": pred_ans_raw,
                "pred_ans": pred_ans,
                "pred_ans_method": pred_ans_method,
                "valid": bool(is_valid),
                "wrong": bool(is_valid and not hit),
                "ans_hit": bool(hit),
                "pred_letter": pred_ans if pred_ans else None,
                "answer_letter": gt_ans if gt_ans else None,
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
            }
            with open(OUT_CAND_MCQ_DETAILS_JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps(detail, ensure_ascii=False) + "\n")

    OUT_CAND_MCQ_TEACHER_CACHE_JSON.write_text(json.dumps(teacher_cache, ensure_ascii=False, indent=2), encoding="utf-8")

    valid_ratio = (valid_answers / total_samples) if total_samples > 0 else 0.0
    acc_over_all = (correct_over_all / total_samples) if total_samples > 0 else 0.0
    acc_over_valid = (correct_over_valid / valid_answers) if valid_answers > 0 else 0.0

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
        "task": "now_narration_cand_mcq",
        "pred_dir": str(PRED_CAND_MCQ_DIR),
        "mcq_dir": str(MCQ_DIR),
        "out_dir": str(OUT_CAND_MCQ_DIR),
        "num_pred_files_total": int(num_files_total),
        "num_pred_files_matched_mcq": int(num_files_matched_mcq),
        "missing_mcq_files": int(missing_mcq_files),
        "missing_mcq_samples": int(missing_mcq_samples),
        "num_pred_samples_seen": int(total_samples),
        "num_valid_answers": int(valid_answers),
        "num_invalid_answers": int(invalid_answers),
        "valid_answer_ratio": float(valid_ratio),
        "answer_accuracy_over_all": float(acc_over_all),
        "answer_accuracy_over_valid": float(acc_over_valid),
        "pred_missing_ans_tag_or_invalid": int(pred_missing_ans_tag_or_invalid),
        "teacher_calls": int(teacher_calls),
        "teacher_cache_hits": int(teacher_cache_hits),
        "teacher_fail": int(teacher_fail),

        # strong distractor report
        "strong_distractor_sources": sorted(list(STRONG_DISTURB_SOURCES)),
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

    OUT_CAND_MCQ_SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n================ NOW_NARRATION CAND_MCQ EVAL ================\n")
    print(f"MODEL_NAME: {MODEL_NAME}")
    print(f"PRED_DIR  : {PRED_CAND_MCQ_DIR}")
    print(f"MCQ_DIR   : {MCQ_DIR}")
    print(f"STRICT_MCQ_PATH: {STRICT_MCQ_PATH}")
    print(f"OUT_DIR   : {OUT_CAND_MCQ_DIR}")
    print("")
    print(f"Pred files total        : {num_files_total}")
    print(f"Pred files matched MCQ  : {num_files_matched_mcq}")
    print(f"MCQ source counts       : {mcq_source_counts}")
    print(f"Missing MCQ files       : {missing_mcq_files}")
    print(f"Missing MCQ samples     : {missing_mcq_samples}")
    print(f"Pred samples seen       : {total_samples}")
    print(f"Valid answers           : {valid_answers}")
    print(f"Invalid answers         : {invalid_answers}")
    print(f"Valid answer ratio      : {valid_ratio:.4f}")
    print(f"Acc over ALL            : {acc_over_all:.4f} ({correct_over_all}/{total_samples})")
    print(f"Acc over VALID          : {acc_over_valid:.4f} ({correct_over_valid}/{valid_answers})")
    print("")
    print("---- Teacher diagnostics ----")
    print(f"Missing/invalid <ANS>   : {pred_missing_ans_tag_or_invalid}")
    print(f"Teacher calls           : {teacher_calls}")
    print(f"Teacher cache hits      : {teacher_cache_hits}")
    print(f"Teacher failures        : {teacher_fail}")
    print("")
    print("---- Strong distractor report (mapped samples only) ----")
    for line in source_stats_report_lines(summary):
        print(line)
    print("")
    print(f"Strong distractor sources      : {sorted(list(STRONG_DISTURB_SOURCES))}")
    print(f"Mapped samples (ANS->source)   : {mapped}")
    print(f"Strong distractor choice rate  : {strong_rate:.4f}  (prev={strong_prev_rate:.4f}, future={strong_future_rate:.4f})")
    print(f"Strong distractor rate | wrong : {strong_rate_wrong:.4f}  (wrong_mapped={wrong_mapped})")
    print(f"Pred source counts             : {pred_source_counts}")
    print(f"GT source counts               : {gt_source_counts}")
    if mean_conf_mapped is not None:
        print(f"Mean <CONF> over mapped        : {mean_conf_mapped}")
    if mean_conf_strong is not None:
        print(f"Mean <CONF> strong             : {mean_conf_strong}")
    if mean_conf_nonstrong is not None:
        print(f"Mean <CONF> non-strong         : {mean_conf_nonstrong}")
    print("")
    print(f"[WROTE] {OUT_CAND_MCQ_SUMMARY_JSON}")
    print(f"[WROTE] {OUT_CAND_MCQ_DETAILS_JSONL}")
    print(f"[WROTE] {OUT_CAND_MCQ_TEACHER_CACHE_JSON}")
    print("\n============================================================\n")


# -------------------------
# Main
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--now_narration_mode",
        type=str,
        default="open",
        choices=["open", "cand"],
        help="open: original open-ended eval; cand: run cand_state + cand_mcq evals",
    )
    parser.add_argument(
        "--mcq-dir",
        type=Path,
        default=None,
        help="Override cand_mcq MCQ directory. Env alternative: NOW_NARRATION_MCQ_DIR.",
    )
    parser.add_argument("--model_name", default=None, help="Override MODEL_NAME without changing the environment.")
    parser.add_argument("--runs_root", type=Path, default=None, help="Infer prediction dirs from <runs_root>/<model_name>/now_narration/.")
    parser.add_argument("--pred_dir", type=Path, default=None, help="Override prediction root; cand mode expects cand_state/cand_mcq below it.")
    parser.add_argument("--out_dir", type=Path, default=None, help="Override output root; cand mode writes cand_state/ and cand_mcq/ below it.")
    parser.add_argument("--flavor", default="cand", help="Candidate flavor for --runs_root inference; default: cand.")
    parser.add_argument(
        "--strict-mcq-path",
        action="store_true",
        help="Fail fast if an effective queryset path or MCQ directory is missing. Env alternative: STRICT_MCQ_PATH=1.",
    )
    args = parser.parse_args()

    global MCQ_DIR, MCQ_DIR_EXPLICIT, STRICT_MCQ_PATH
    _apply_cli_paths(
        mode=args.now_narration_mode,
        model_name=args.model_name,
        runs_root=args.runs_root,
        pred_dir=args.pred_dir,
        out_dir=args.out_dir,
        flavor=args.flavor,
    )
    if args.mcq_dir is not None:
        MCQ_DIR = args.mcq_dir.expanduser()
        MCQ_DIR_EXPLICIT = True
    if args.strict_mcq_path:
        STRICT_MCQ_PATH = True

    if args.now_narration_mode == "open":
        print("\n================ NOW_NARRATION PATHS ================\n")
        print(f"MODEL_NAME: {MODEL_NAME}")
        print(f"PRED_DIR  : {PRED_OPEN_DIR}")
        print(f"OUT_DIR   : {OUT_OPEN_DIR}")
        print("")
        eval_open()
    else:
        print("\n================ NOW_NARRATION PATHS ================\n")
        print(f"MODEL_NAME          : {MODEL_NAME}")
        print(f"PRED_CAND_STATE_DIR : {PRED_CAND_STATE_DIR}")
        print(f"PRED_CAND_MCQ_DIR   : {PRED_CAND_MCQ_DIR}")
        print(f"MCQ_DIR             : {MCQ_DIR}")
        print(f"OUT_CAND_STATE_DIR  : {OUT_CAND_STATE_DIR}")
        print(f"OUT_CAND_MCQ_DIR    : {OUT_CAND_MCQ_DIR}")
        print(f"STRICT_MCQ_PATH     : {STRICT_MCQ_PATH}")
        print("")
        eval_cand_state()
        eval_cand_mcq()


if __name__ == "__main__":
    main()
