#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOW-style helper (model-agnostic) for Ego4D interval clips.

This helper assumes the RUNNER already CUTS interval clip [interval_start_sec, interval_end_sec]
from the full <video_uid>.mp4, and then runs inference ON THAT CLIP.

Therefore:
  - sample.t_eval should remain RELATIVE to the interval clip (0..interval_len_sec)
  - time offset should be 0 by default (NOW_TIME_MODE defaults to "clip")

Extended:
  - Support sh_rtrv (past retrieval) in open-ended and candidate/MCQ flavors.
  - Support ms_rtrv (multistep past retrieval) in open-ended and candidate/MCQ flavors.
  - Support ms_pred (multistep prediction) in open-ended and candidate/MCQ flavors.
  - Candidate flavor reads shuffled MCQ from MCP_ROOT/{train,val}/<same-stem-as-GT>.json(.jsonl)
    where <TASK> is sh_rtrv or ms_rtrv or ms_pred.

Env:
  - PRED_FLAVOR: open | cand  (default open)
  - SH_RTRV_LOOKBACK_SEC: float (default 20)   [for sh_rtrv only]
  - MS_PRED_CONTEXT_SEC: float (default 20)    [for ms_pred only]
  - MCP_ROOT: root of shuffled mcq (wrapper should set this; fallback is
        <PROJECT_ROOT>/data/mcq_shuffled/<TASK>)
  - MCP_SPLIT_SUBDIR: 1/0 whether MCP_ROOT has train/val subdirs (default 1)

IMPORTANT:
  - In cand mode, we DO NOT reshuffle options. We trust the shuffled MCQ file order for full reproducibility.

NEW (uniform downsampling + global stopping, task-specific):
  - now_narration(open / cand_state): keep 2 out of every 3 samples (uniform), across GT files; stop flag after cumulative >= 800.
  - now_narration(cand_mcq): keep 2 out of every 3 MCQ samples (uniform), across GT files; stop flag after cumulative >= 800. (independent stream)
  - ms_rtrv/ms_pred: samples are bound by anchor_idx in contiguous groups (typically 3); keep 2 out of every 3 groups (uniform), across GT files; stop flag after cumulative >= 800.
  - helper writes helper.stop_after_this_file in output JSON for runner to stop entering next GT after finishing this file.

NEW (now_state_switch cand_state):
  - now_state_switch supports pred_flavor:
      * open       -> use NOW_XML_TEMPLATE_V1 (same as now_narration open)
      * cand_state -> use NOW_STATE_ONLY_CAND_TEMPLATE (same as now_narration cand_state)
"""

from __future__ import annotations

import os
import json
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from caption_cache import build_segment_key, build_usage_key, caption_record_path


# ----------------------------
# NOW templates (unchanged)
# ----------------------------

NOW_XML_TEMPLATE_V1 = """[now narration | t={t_eval:.1f}s] Based on the egocentric video up to now, answer THREE fields:

1) Visible interaction now:
- Output INTERACTION if the wearer is clearly interacting with an object that is visible and verifiable NOW.
- Output NO_INTERACTION if there is no clear/valid object interaction now (e.g., walking, looking around, pausing, the object of change is not visible).

2) A short description (one short sentence) of what is happening right now.

3) Confidence (0 to 1):
- Output a number in [0, 1] indicating how confident you are that STATE+VERB+NOUN are all correct and verifiable in the visible frames.

Output EXACTLY in one line:
<NOW><STATE>STATE_HERE</STATE><VERB>verb_here</VERB><NOUN>noun_here_or_none</NOUN><DESC>short_concise_sentence</DESC><CONF>0.00</CONF></NOW>

Rules:
- STATE_HERE must be either INTERACTION or NO_INTERACTION.
- If STATE_HERE is NO_INTERACTION, set NOUN to "none".
- VERB should be one main verb (e.g., open, take, wipe, look, walk, stand).
- DESC should be a short natural sentence consistent with STATE_HERE.
- CONF must be a decimal number between 0 and 1 (inclusive), preferably with two digits (e.g., 0.73).
- CONF semantics: the probability that STATE+VERB+NOUN are all correct and verifiable in the visible frames.
- If STATE_HERE is NO_INTERACTION (e.g., walking/looking/waiting), CONF should not be overly high unless you are very certain.
- Do NOT output anything outside the tags."""


# ----------------------------
# NEW: now_narration cand templates (state-only + action MCQ)
# ----------------------------

NOW_STATE_ONLY_CAND_TEMPLATE = """[now narration | STATE only | t={t_eval:.1f}s] Based on the egocentric video up to now, decide whether there is a visible, verifiable object interaction RIGHT NOW.

Definitions:
- Output INTERACTION if the wearer is clearly interacting with a visible object NOW (hands/tools manipulating, holding, operating, wearing/putting on, etc.).
- Output NO_INTERACTION if there is no clear/valid visible object interaction now (e.g., walking, looking around, pausing, the object of change is not visible).

Output EXACTLY in ONE line:
<NOW><STATE>INTERACTION</STATE><CONF>0.00</CONF></NOW>

Rules:
- STATE must be exactly INTERACTION or NO_INTERACTION (uppercase).
- CONF must be a decimal number in [0, 1] (inclusive), preferably with two digits.
- Do NOT output anything outside the tags.
- Do NOT output VERB/NOUN/DESC or any extra explanation."""


NOW_ACTION_MCQ_TEMPLATE = """[now narration | ACTION MCQ | t={t_eval:.1f}s] Based on the egocentric video up to now, the wearer IS interacting with an object RIGHT NOW. Choose the ONE option (A/B/C/D) that best matches the current visible interaction.

Then output:
- the chosen option letter (ANS),
- the verb and noun copied EXACTLY from the chosen option,
- confidence.

Output EXACTLY in ONE line:
<NOW><ANS>A</ANS><VERB>verb_here</VERB><NOUN>noun_here</NOUN><CONF>0.00</CONF></NOW>

Rules:
- You MUST choose exactly one of A/B/C/D.
- VERB and NOUN must be copied EXACTLY from the chosen option (same wording, no tense changes).
- CONF must be a decimal number in [0, 1] (inclusive), preferably with two digits.
- Do NOT output anything outside the tags.
- Do NOT add any extra explanation.

Options:
A. {A_OPT}
B. {B_OPT}
C. {C_OPT}
D. {D_OPT}"""


# ----------------------------
# sh_rtrv templates (unchanged)
# ----------------------------

PAST_RTRV_OPEN_TEMPLATE = """[past retrieval | t={t_eval:.1f}s | lookback={lookback_sec:.1f}s] Based on the egocentric video up to now, retrieve what MOST RECENTLY happened within the last {lookback_sec:.1f}s (from t-{lookback_sec:.1f}s to t).

Answer FOUR fields:

1) Verb (canonical form, NO tense):
- Output ONE main verb in base form (e.g., open, take, cut, wipe, move, put, pick, pour, look, walk).
- Do NOT use tense changes (no -ing, no -ed, no "was/were/did").

2) Noun (canonical phrase, NO tense):
- Output a short noun phrase describing the main object (e.g., "concrete", "brick mold", "knife", "manure").
- Keep it simple; avoid full sentences.
- If you cannot verify a specific object/action in the last window, set NOUN to "none" and VERB to "none".

3) Short description (simple declarative sentence):
- Output ONE short sentence, preferably starting with "YOU".
- Keep it a simple statement without tense changes.
- If VERB="none" or NOUN="none", set DESC to "YOU do nothing".

4) Confidence (0 to 1):
- Output a number in [0, 1] indicating how confident you are that VERB+NOUN+DESC are correct and supported by frames within the last {lookback_sec:.1f}s.

Output EXACTLY in one line:
<PAST><VERB>verb_here</VERB><NOUN>noun_here_or_none</NOUN><DESC>short_concise_sentence</DESC><CONF>0.00</CONF></PAST>

Rules:
- VERB must be a single base-form verb token (no tense).
- NOUN must be a short noun phrase or "none".
- DESC must be a short simple declarative sentence; avoid tense markers ("was", "were", "-ed", "-ing").
- CONF must be a decimal number between 0 and 1 (inclusive), preferably with two digits.
- Do NOT output anything outside the tags."""


PAST_RTRV_CAND_TEMPLATE = """[past retrieval | candidate | t={t_eval:.1f}s | lookback={lookback_sec:.1f}s] Based on the egocentric video up to now, choose the ONE option (A/B/C/D) that best matches what MOST RECENTLY happened within the last {lookback_sec:.1f}s (from t-{lookback_sec:.1f}s to t).

Then output:
- the chosen option letter (ANS),
- the verb and noun copied EXACTLY from the chosen option,
- a short simple declarative sentence consistent with the chosen option,
- confidence.

Output EXACTLY in one line:
<PAST><ANS>A</ANS><VERB>verb_here</VERB><NOUN>noun_here_or_none</NOUN><DESC>short_concise_sentence</DESC><CONF>0.00</CONF></PAST>

Rules:
- You MUST choose exactly one of A/B/C/D.
- VERB and NOUN must be copied EXACTLY from the chosen option (same wording, no tense changes).
- DESC must be ONE short simple declarative sentence, preferably: "YOU <VERB> <NOUN>" (or "YOU do nothing" if NOUN is "none").
- Do NOT add tense changes (no -ing, no -ed, no "was/were/did") anywhere.
- CONF must be a decimal number in [0, 1], preferably with two digits.
- CONF semantics: probability that the chosen option is correct and supported by frames within the last {lookback_sec:.1f}s.
- Do NOT output anything outside the tags.

Options:
A. {A_OPT}
B. {B_OPT}
C. {C_OPT}
D. {D_OPT}"""


# ----------------------------
# ms_rtrv templates (NEW, per your spec)
# ----------------------------

MS_RTRV_OPEN_TEMPLATE = """[multistep past retrieval | t={t_eval:.1f}s | lag={lookback_sec:.1f}s] Based on the egocentric video up to now, recall what happened about {lookback_sec:.1f}s ago (i.e., at time t-{lookback_sec:.1f}s).

You should answer the main action/event around that past moment (not the most recent event in the last window). If you cannot verify a specific action/object around t-{lookback_sec:.1f}s, set VERB="none" and NOUN="none".

Answer FOUR fields:

1) Verb (canonical form, NO tense):
- Output ONE main verb in base form (e.g., open, take, cut, wipe, move, put, pick, pour, look, walk).
- Do NOT use tense changes (no -ing, no -ed, no "was/were/did").

2) Noun (canonical phrase, NO tense):
- Output a short noun phrase describing the main object (e.g., "concrete", "brick mold", "knife", "manure").
- Keep it simple; avoid full sentences.
- If you cannot verify a specific object/action around t-{lookback_sec:.1f}s, set NOUN to "none" and VERB to "none".

3) Short description (simple declarative sentence):
- Output ONE short sentence, preferably starting with "YOU".
- Keep it a simple statement without tense changes.
- If VERB="none" or NOUN="none", set DESC to "YOU do nothing".

4) Confidence (0 to 1):
- Output a number in [0, 1] indicating how confident you are that VERB+NOUN+DESC are correct and supported by frames around t-{lookback_sec:.1f}s.

Output EXACTLY in one line:
<PAST><VERB>verb_here</VERB><NOUN>noun_here_or_none</NOUN><DESC>short_concise_sentence</DESC><CONF>0.00</CONF></PAST>

Rules:
- VERB must be a single base-form verb token (no tense).
- NOUN must be a short noun phrase or "none".
- DESC must be a short simple declarative sentence; avoid tense markers ("was", "were", "-ed", "-ing").
- CONF must be a decimal number between 0 and 1 (inclusive), preferably with two digits.
- Do NOT output anything outside the tags."""


MS_RTRV_CAND_TEMPLATE = """[multistep past retrieval | candidate | t={t_eval:.1f}s | lag={lookback_sec:.1f}s] Based on the egocentric video up to now, choose the ONE option (A/B/C/D) that best matches what happened about {lookback_sec:.1f}s ago (i.e., at time t-{lookback_sec:.1f}s).

Then output:
- the chosen option letter (ANS),
- the verb and noun copied EXACTLY from the chosen option,
- a short simple declarative sentence consistent with the chosen option,
- confidence.

Output EXACTLY in one line:
<PAST><ANS>A</ANS><VERB>verb_here</VERB><NOUN>noun_here_or_none</NOUN><DESC>short_concise_sentence</DESC><CONF>0.00</CONF></PAST>

Rules:
- You MUST choose exactly one of A/B/C/D.
- VERB and NOUN must be copied EXACTLY from the chosen option (same wording, no tense changes).
- DESC must be ONE short simple declarative sentence, preferably: "YOU <VERB> <NOUN>" (or "YOU do nothing" if NOUN is "none").
- Do NOT add tense changes (no -ing, no -ed, no "was/were/did") anywhere.
- CONF must be a decimal number in [0, 1], preferably with two digits.
- CONF semantics: probability that the chosen option is correct and supported by frames around t-{lookback_sec:.1f}s.
- Do NOT output anything outside the tags.

Options:
A. {A_OPT}
B. {B_OPT}
C. {C_OPT}
D. {D_OPT}"""


# ----------------------------
# ms_pred templates (NEW, per your spec)
# ----------------------------

MS_PRED_OPEN_TEMPLATE = """[multistep prediction | t={t_eval:.1f}s | horizon={lookback_sec:.1f}s] Based on the egocentric video up to now, predict what will happen about {lookback_sec:.1f}s later (i.e., at time t+{lookback_sec:.1f}s).

You should predict the main action/event around that future moment (not what is happening right now, and not a generic guess). If you cannot reliably predict a specific action/object around t+{lookback_sec:.1f}s, set VERB="none" and NOUN="none".

Answer FOUR fields:

1) Verb (canonical form, NO tense):
- Output ONE main verb in base form (e.g., open, take, cut, wipe, move, put, pick, pour, look, walk).
- Do NOT use tense changes (no -ing, no -ed, no "will/going to/was/were/did").

2) Noun (canonical phrase, NO tense):
- Output a short noun phrase describing the main object (e.g., "concrete", "brick mold", "knife", "manure").
- Keep it simple; avoid full sentences.
- If you cannot reliably predict a specific object/action around t+{lookback_sec:.1f}s, set NOUN to "none" and VERB to "none".

3) Short description (simple declarative sentence):
- Output ONE short sentence, preferably starting with "YOU".
- Keep it a simple statement without tense changes.
- If VERB="none" or NOUN="none", set DESC to "YOU do nothing".

4) Confidence (0 to 1):
- Output a number in [0, 1] indicating how confident you are that VERB+NOUN+DESC will be correct around t+{lookback_sec:.1f}s.

Output EXACTLY in one line:
<FUTURE><VERB>verb_here</VERB><NOUN>noun_here_or_none</NOUN><DESC>short_concise_sentence</DESC><CONF>0.00</CONF></FUTURE>

Rules:
- VERB must be a single base-form verb token (no tense).
- NOUN must be a short noun phrase or "none".
- DESC must be a short simple declarative sentence; avoid tense markers ("will", "going to", "was", "were", "-ed", "-ing").
- CONF must be a decimal number between 0 and 1 (inclusive), preferably with two digits.
- Do NOT output anything outside the tags."""


MS_PRED_CAND_TEMPLATE = """[multistep prediction | candidate | t={t_eval:.1f}s | horizon={lookback_sec:.1f}s] Based on the egocentric video up to now, choose the ONE option (A/B/C/D) that best matches what will happen about {lookback_sec:.1f}s later (i.e., at time t+{lookback_sec:.1f}s).

Then output:
- the chosen option letter (ANS),
- the verb and noun copied EXACTLY from the chosen option,
- a short simple declarative sentence consistent with the chosen option,
- confidence.

Output EXACTLY in one line:
<FUTURE><ANS>A</ANS><VERB>verb_here</VERB><NOUN>noun_here_or_none</NOUN><DESC>short_concise_sentence</DESC><CONF>0.00</CONF></FUTURE>

Rules:
- You MUST choose exactly one of A/B/C/D.
- VERB and NOUN must be copied EXACTLY from the chosen option (same wording, no tense changes).
- DESC must be ONE short simple declarative sentence, preferably: "YOU <VERB> <NOUN>" (or "YOU do nothing" if NOUN is "none").
- Do NOT add tense changes (no -ing, no -ed, no "will/going to/was/were/did") anywhere.
- CONF must be a decimal number in [0, 1], preferably with two digits.
- CONF semantics: probability that the chosen option will be correct around t+{lookback_sec:.1f}s.
- Do NOT output anything outside the tags.

Options:
A. {A_OPT}
B. {B_OPT}
C. {C_OPT}
D. {D_OPT}"""


# ----------------------------
# sh_pred templates (minimal changes from sh_rtrv)
# ----------------------------

SH_PRED_OPEN_TEMPLATE = """[short-horizon prediction | t={t_eval:.1f}s | horizon={lookback_sec:.1f}s] Based on the egocentric video up to now, predict what will MOST LIKELY happen NEXT within the next {lookback_sec:.1f}s (from t to t+{lookback_sec:.1f}s).

You should predict the next main action/event within this short future horizon (not what is happening right now, and not a generic guess). If you cannot reliably predict a specific object/action within the next {lookback_sec:.1f}s, set NOUN to "none" and VERB to "none".

Answer FOUR fields:

1) Verb (canonical form, NO tense):
- Output ONE main verb in base form (e.g., open, take, cut, wipe, move, put, pick, pour, look, walk).
- Do NOT use tense changes (no -ing, no -ed, no "will/going to/was/were/did").

2) Noun (canonical phrase, NO tense):
- Output a short noun phrase describing the main object (e.g., "concrete", "brick mold", "knife", "manure").
- Keep it simple; avoid full sentences.
- If you cannot reliably predict a specific object/action within the next {lookback_sec:.1f}s, set NOUN to "none" and VERB to "none".

3) Short description (simple declarative sentence):
- Output ONE short sentence, preferably starting with "YOU".
- Keep it a simple statement without tense changes.
- If VERB="none" or NOUN="none", set DESC to "YOU do nothing".

4) Confidence (0 to 1):
- Output a number in [0, 1] indicating how confident you are that VERB+NOUN+DESC will be correct within the next {lookback_sec:.1f}s.

Output EXACTLY in one line:
<FUTURE><VERB>verb_here</VERB><NOUN>noun_here_or_none</NOUN><DESC>short_concise_sentence</DESC><CONF>0.00</CONF></FUTURE>

Rules:
- VERB must be a single base-form verb token (no tense).
- NOUN must be a short noun phrase or "none".
- DESC must be a short simple declarative sentence; avoid tense markers ("will", "going to", "was", "were", "-ed", "-ing").
- CONF must be a decimal number between 0 and 1 (inclusive), preferably with two digits.
- Do NOT output anything outside the tags."""


SH_PRED_CAND_TEMPLATE = """[short-horizon prediction | candidate | t={t_eval:.1f}s | horizon={lookback_sec:.1f}s] Based on the egocentric video up to now, choose the ONE option (A/B/C/D) that best matches what will MOST LIKELY happen NEXT within the next {lookback_sec:.1f}s (from t to t+{lookback_sec:.1f}s).

Then output:
- the chosen option letter (ANS),
- the verb and noun copied EXACTLY from the chosen option,
- a short simple declarative sentence consistent with the chosen option,
- confidence.

Output EXACTLY in one line:
<FUTURE><ANS>A</ANS><VERB>verb_here</VERB><NOUN>noun_here_or_none</NOUN><DESC>short_concise_sentence</DESC><CONF>0.00</CONF></FUTURE>

Rules:
- You MUST choose exactly one of A/B/C/D.
- VERB and NOUN must be copied EXACTLY from the chosen option (same wording, no tense changes).
- DESC must be ONE short simple declarative sentence, preferably: "YOU <VERB> <NOUN>" (or "YOU do nothing" if NOUN is "none").
- Do NOT add tense changes (no -ing, no -ed, no "will/going to/was/were/did") anywhere.
- CONF must be a decimal number in [0, 1], preferably with two digits.
- CONF semantics: probability that the chosen option will be correct within the next {lookback_sec:.1f}s.
- Do NOT output anything outside the tags.

Options:
A. {A_OPT}
B. {B_OPT}
C. {C_OPT}
D. {D_OPT}"""


# ----------------------------
# Utilities
# ----------------------------

def _load_json_or_one_jsonl(path: str | Path) -> Dict[str, Any]:
    """
    Robust loader:
      - .json: dict
      - .jsonl:
          1) try json.loads(full_text) (works for pretty multi-line JSON object even if suffix is .jsonl)
          2) fallback: treat as JSONL where each non-empty line is a JSON object; require exactly 1 record.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"GT/MCP file not found: {p}")

    raw = p.read_text(encoding="utf-8")
    if not raw.strip():
        raw = p.read_text(encoding="utf-8-sig")
    raw = raw.strip()
    if not raw:
        raise ValueError(f"Empty file: {p}")

    # Try full JSON first
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
            return obj[0]
        raise ValueError(f"Top-level JSON must be dict: {p} (got {type(obj)})")
    except json.JSONDecodeError:
        pass

    # Fallback JSONL
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    recs = [json.loads(ln) for ln in lines]
    if len(recs) != 1 or not isinstance(recs[0], dict):
        raise ValueError(f"Expected exactly 1 JSON object in JSONL: {p}, got {len(recs)}")
    return recs[0]


def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _env_get(key: str, default: str) -> str:
    v = os.environ.get(key, "").strip()
    return v if v else default


def _normalize_task_name(task: str) -> str:
    t = (task or "").strip().lower()
    if not t:
        return "now_narration"
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


def _normalize_pred_flavor(x: str) -> str:
    t = (x or "").strip().lower()
    if t in {"cand", "candidate", "mcq"}:
        return "cand"
    return "open"


def _normalize_now_narration_flavor(x: str) -> str:
    """
    now_narration supports 3 flavors:
      - open
      - cand_state  (state-only)
      - cand_mcq    (action MCQ, segment-only)
    """
    t = (x or "").strip().lower()
    if not t or t == "open":
        return "open"
    if t in {"cand_state", "state", "state_only", "state-only", "candstate"}:
        return "cand_state"
    if t in {"cand_mcq", "mcq", "action", "action_mcq", "action-mcq", "candmcq", "cand"}:
        return "cand_mcq"
    return "open"


def _normalize_now_state_switch_flavor(x: str) -> str:
    """
    now_state_switch supports 2 flavors only:
      - open
      - cand_state
    Map any "cand"/"candidate"/"mcq" to cand_state for convenience.
    """
    t = (x or "").strip().lower()
    if not t or t == "open":
        return "open"
    if t in {"cand_state", "state", "state_only", "state-only", "candstate", "cand", "candidate", "mcq"}:
        return "cand_state"
    return "open"


def _default_mcp_root_for_task(task: str) -> str:
    task = (task or "").strip().lower()
    project_root = Path(__file__).resolve().parents[1]
    if task == "now_narration_action":
        return str(project_root / "data" / "mcq_shuffled" / "now_narration_action")
    if task == "ms_rtrv":
        return str(project_root / "data" / "mcq_shuffled" / "ms_rtrv")
    if task == "ms_pred":
        return str(project_root / "data" / "mcq_shuffled" / "ms_pred")
    if task == "sh_pred":
        return str(project_root / "data" / "mcq_shuffled" / "sh_pred")
    # default to sh_rtrv for backward compatibility
    return str(project_root / "data" / "mcq_shuffled" / "sh_rtrv")


def _resolve_mcp_path(gt_path: Path, split: str, task: str) -> Path:
    """
    If MCP_ROOT is set, use it.
    Else use default based on task.
    If MCP_SPLIT_SUBDIR=1 (default), use MCP_ROOT/<split>/.
    File names align by stem:
      <gt_stem>.json or <gt_stem>.jsonl
    """
    default_root = _default_mcp_root_for_task(task)
    mcp_root = Path(_env_get("MCP_ROOT", default_root)).expanduser()
    use_split_subdir = _env_get("MCP_SPLIT_SUBDIR", "1").strip() != "0"

    base = (mcp_root / split) if use_split_subdir else mcp_root
    p_json = base / f"{gt_path.stem}.json"
    if p_json.exists():
        return p_json
    p_jsonl = base / f"{gt_path.stem}.jsonl"
    if p_jsonl.exists():
        return p_jsonl

    raise FileNotFoundError(f"MCP not found for GT stem={gt_path.stem} under {base}")


def _index_samples_by_idx(samples: Any) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    if not isinstance(samples, list):
        return out
    for s in samples:
        if not isinstance(s, dict):
            continue
        if "idx" not in s:
            continue
        try:
            out[int(s["idx"])] = s
        except Exception:
            continue
    return out


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


# ----------------------------
# Uniform downsampling + global stop (NEW)
# ----------------------------

_TARGET_ENV_MAP = {
    "now_narration": "NOW_TARGET_SAMPLES",
    "now_narration_mcq": "NOW_TARGET_SAMPLES",   # NEW: independent stream for cand_mcq, but same env knobs
    "ms_rtrv": "MS_RTRV_TARGET_SAMPLES",
    "ms_pred": "MS_PRED_TARGET_SAMPLES",
}
_DEFAULT_TARGET = 800

_DS_ENV_EVERY_MAP = {
    "now_narration": "NOW_DS_EVERY",
    "now_narration_mcq": "NOW_DS_EVERY",         # NEW
    "ms_rtrv": "MS_RTRV_DS_EVERY",
    "ms_pred": "MS_PRED_DS_EVERY",
}
_DS_ENV_KEEP_MAP = {
    "now_narration": "NOW_DS_KEEP",
    "now_narration_mcq": "NOW_DS_KEEP",          # NEW
    "ms_rtrv": "MS_RTRV_DS_KEEP",
    "ms_pred": "MS_PRED_DS_KEEP",
}
_DS_DEFAULT_EVERY = 3
_DS_DEFAULT_KEEP = 2

_GLOBAL_DS_PHASE: Dict[str, int] = {"now_narration": 0, "now_narration_mcq": 0, "ms_rtrv": 0, "ms_pred": 0}
_GLOBAL_CUM_SAMPLES: Dict[str, int] = {"now_narration": 0, "now_narration_mcq": 0, "ms_rtrv": 0, "ms_pred": 0}


def _task_target_samples(task: str) -> int:
    envk = _TARGET_ENV_MAP.get(task, "")
    v = os.environ.get(envk, "").strip() if envk else ""
    if not v:
        v = os.environ.get("TASK_TARGET_SAMPLES", "").strip()
    if v:
        try:
            n = int(float(v))
            if n > 0:
                return n
        except Exception:
            pass
    return int(_DEFAULT_TARGET)


def _task_ds_params(task: str) -> Tuple[int, int]:
    env_every = _DS_ENV_EVERY_MAP.get(task, "")
    env_keep = _DS_ENV_KEEP_MAP.get(task, "")
    every_s = os.environ.get(env_every, "").strip() if env_every else ""
    keep_s = os.environ.get(env_keep, "").strip() if env_keep else ""

    every = _safe_int(every_s, _DS_DEFAULT_EVERY) if every_s else _DS_DEFAULT_EVERY
    keep = _safe_int(keep_s, _DS_DEFAULT_KEEP) if keep_s else _DS_DEFAULT_KEEP

    if every <= 0:
        every = _DS_DEFAULT_EVERY
    if keep < 0:
        keep = 0
    if keep > every:
        keep = every
    return int(every), int(keep)


def _ds_should_keep_unit(task: str) -> bool:
    every, keep = _task_ds_params(task)
    ph = int(_GLOBAL_DS_PHASE.get(task, 0))
    keep_flag = (ph % every) < keep
    _GLOBAL_DS_PHASE[task] = ph + 1
    return bool(keep_flag)


def _ds_update_cumulative(task: str, add_samples: int) -> Tuple[int, int, int, bool, bool]:
    target = _task_target_samples(task)
    before = int(_GLOBAL_CUM_SAMPLES.get(task, 0))
    already = before >= target
    after = before + int(add_samples)
    _GLOBAL_CUM_SAMPLES[task] = after
    reached = after >= target
    return before, after, target, reached, already


def _anchor_group_key(s: Dict[str, Any], fallback_unique: int) -> Tuple[str, int]:
    for k in ("anchor_idx", "anchor_id", "anchor", "anchorIndex"):
        if k in s and s.get(k) is not None:
            try:
                return ("a", int(s.get(k)))
            except Exception:
                pass
    return ("i", _safe_int(s.get("idx", fallback_unique), fallback_unique))


def _group_contiguous_by_anchor(samples_list: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    groups: List[List[Dict[str, Any]]] = []
    prev_key: Optional[Tuple[str, int]] = None
    cur: List[Dict[str, Any]] = []
    for i, s in enumerate(samples_list):
        if not isinstance(s, dict):
            continue
        k = _anchor_group_key(s, fallback_unique=i)
        if prev_key is None:
            prev_key = k
            cur = [s]
            continue
        if k == prev_key:
            cur.append(s)
        else:
            if cur:
                groups.append(cur)
            cur = [s]
            prev_key = k
    if cur:
        groups.append(cur)
    return groups


# ----------------------------
# NOW reject allowlist (NEW)
# ----------------------------

def _norm_reject_reason(x: Any) -> str:
    s = str(x or "").strip().lower()
    s = s.replace(" ", "")
    return s


_NOW_REJECT_ALLOWLIST = {
    _norm_reject_reason("no_human - object_interaction_from_camera_wearer_(e.g._c_looks_around, _c_pauses_work)"),
    _norm_reject_reason("no_human-object_interaction_from_camera_wearer_(e.g._c_looks_around,_c_pauses_work)"),
    _norm_reject_reason("object_of_change_is_not_visibles"),
}


def _should_reject_now_sample(sample: Dict[str, Any], filter_rejected: bool) -> bool:
    if not filter_rejected:
        return False
    if not bool(sample.get("is_rejected", False)):
        return False
    rr = sample.get("reject_reason", sample.get("rejection_reason", sample.get("rejectReason", "")))
    rrn = _norm_reject_reason(rr)
    if rrn in _NOW_REJECT_ALLOWLIST:
        return False
    return True


# ----------------------------
# now_state_switch helpers (NEW)
# ----------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _time_key(t: float, dt: float) -> int:
    if dt <= 0:
        return int(round(t))
    return int(round(float(t) / float(dt)))


def _parse_ss_pos(raw: str) -> Tuple[str, float, str, bool]:
    s = (raw or "").strip().lower()
    if not s:
        s = "r:0.0"
    used_heur = False

    if ":" in s:
        a, b = s.split(":", 1)
        a = a.strip()
        b = b.strip()
        if a in {"r", "ratio"}:
            return "ratio", float(_f(b, 0.0)), raw, used_heur
        if a in {"t", "time", "s", "sec", "secs"}:
            return "time", float(_f(b, 0.0)), raw, used_heur

    if s.endswith("r"):
        return "ratio", float(_f(s[:-1], 0.0)), raw, used_heur
    if s.endswith("s"):
        return "time", float(_f(s[:-1], 0.0)), raw, used_heur

    if s.startswith("r") and len(s) > 1:
        return "ratio", float(_f(s[1:], 0.0)), raw, used_heur
    if s.startswith("t") and len(s) > 1:
        return "time", float(_f(s[1:], 0.0)), raw, used_heur

    v = float(_f(s, 0.0))
    used_heur = True
    if v <= 1.0:
        return "ratio", v, raw, used_heur
    return "time", v, raw, used_heur


def _get_bounds(
    sample: Dict[str, Any],
    start_keys: Tuple[str, ...],
    end_keys: Tuple[str, ...],
    *,
    t_center: float,
    dt: float,
    interval_len_sec: float,
) -> Tuple[float, float]:
    start = None
    end = None
    for k in start_keys:
        if k in sample and sample.get(k) is not None:
            start = _f(sample.get(k), None)
            break
    for k in end_keys:
        if k in sample and sample.get(k) is not None:
            end = _f(sample.get(k), None)
            break

    if start is None or end is None:
        hw = float(dt) * 0.5 if dt > 0 else 4.0
        start = float(t_center - hw)
        end = float(t_center + hw)

    start = float(start)
    end = float(end)
    if end < start:
        start, end = end, start

    start = max(0.0, start)
    if interval_len_sec and interval_len_sec > 0:
        end = min(float(interval_len_sec), end)

    if end < start:
        end = start
    return start, end


def _pos_in_span(start: float, end: float, mode: str, value: float) -> float:
    start = float(start)
    end = float(end)
    if end < start:
        start, end = end, start
    if mode == "ratio":
        r = _clamp(float(value), 0.0, 1.0)
        return float(start + r * (end - start))
    return float(start + float(value))


def _pick_lag_sec_for_ms_rtrv(sample: Dict[str, Any], params: Dict[str, Any]) -> float:
    if isinstance(sample, dict):
        if "lookback_sec" in sample and sample.get("lookback_sec") is not None:
            v = _f(sample.get("lookback_sec"), 0.0)
            if v > 0:
                return float(v)
        if "lag_sec" in sample and sample.get("lag_sec") is not None:
            v = _f(sample.get("lag_sec"), 0.0)
            if v > 0:
                return float(v)

    if isinstance(params, dict):
        lags = params.get("lags_sec", None)
        step_idx = _safe_int(sample.get("step_idx", None), -1)
        if isinstance(lags, list) and step_idx >= 0 and step_idx < len(lags):
            v = _f(lags[step_idx], 0.0)
            if v > 0:
                return float(v)

    step_idx = _safe_int(sample.get("step_idx", None), -1)
    if step_idx in (0, 1, 2):
        return float(8.0 * (step_idx + 1))

    return 8.0


def _pick_horizon_sec_for_ms_pred(sample: Dict[str, Any], params: Dict[str, Any], t_eval_rel: float) -> float:
    if isinstance(sample, dict):
        for k in ("horizon_sec", "lookahead_sec", "pred_sec"):
            if k in sample and sample.get(k) is not None:
                v = _f(sample.get(k), 0.0)
                if v > 0:
                    return float(v)

    step_idx = _safe_int(sample.get("step_idx", None), -1)
    if isinstance(params, dict):
        hs = params.get("horizons_sec", None)
        if isinstance(hs, list) and step_idx >= 0 and step_idx < len(hs):
            v = _f(hs[step_idx], 0.0)
            if v > 0:
                return float(v)

        lags = params.get("lags_sec", None)
        if isinstance(lags, list) and step_idx >= 0 and step_idx < len(lags):
            v = _f(lags[step_idx], 0.0)
            if v > 0:
                return float(v)

    if isinstance(sample, dict):
        for k in ("anchor_t_eval", "anchor_sec", "anchor_time_sec", "target_t_eval", "target_sec"):
            if k in sample and sample.get(k) is not None:
                anchor = _f(sample.get(k), 0.0)
                if anchor > 0:
                    v = float(anchor - float(t_eval_rel))
                    if v > 0:
                        return v

    if step_idx in (0, 1, 2):
        return float(8.0 * (step_idx + 1))

    return 8.0


def _pick_horizon_sec_for_sh_pred(sample: Dict[str, Any]) -> float:
    env_v = os.environ.get("SH_PRED_HORIZON_SEC", "").strip()
    if env_v:
        v = _f(env_v, 8.0)
        if v > 0:
            return float(v)

    if isinstance(sample, dict):
        for k in ("horizon_sec", "lookahead_sec", "pred_sec"):
            if k in sample and sample.get(k) is not None:
                v = _f(sample.get(k), 0.0)
                if v > 0:
                    return float(v)

    return 8.0

'''
def _ms_window_from_evidence_segment(sample: Dict[str, Any], t_eval_sched: float, lag: float) -> Tuple[float, float, float]:
    t_target = float(t_eval_sched - float(lag))

    gs = sample.get("gt_segment_start", None)
    ge = sample.get("gt_segment_end", None)
    if gs is not None and ge is not None:
        ws = float(_f(gs, 0.0))
        we = float(_f(ge, ws))
        if we < ws:
            we = ws
        we = min(we, float(t_eval_sched))
        ws = max(0.0, min(ws, we))
        return ws, we, t_target

    pad = 2.0
    ws = max(0.0, t_target - pad)
    we = min(float(t_eval_sched), t_target + pad)
    if we < ws:
        we = ws
    return ws, we, t_target
'''

def _ms_window_from_evidence_segment(sample: Dict[str, Any], t_eval_sched: float, lag: float) -> Tuple[float, float, float]:
    """
    ms_rtrv strict-online window:
      - window_end MUST align to t_eval_sched (no look-ahead)
      - window_start SHOULD cover evidence start (gt_segment_start) if available
      - fallback: start around t_target, but still end at t_eval_sched
    Also robust to time_mode="full" by inferring offset = t_eval - t_eval_rel when available.
    """
    t_eval_sched = float(t_eval_sched)
    lag = float(lag)
    t_target = float(t_eval_sched - lag)

    # infer offset (clip->full) if possible: offset = t_eval - t_eval_rel
    # in your helper, ns always has both fields set before calling this function
    off = 0.0
    try:
        if isinstance(sample, dict) and ("t_eval" in sample) and ("t_eval_rel" in sample):
            off = float(_f(sample.get("t_eval", 0.0), 0.0) - _f(sample.get("t_eval_rel", 0.0), 0.0))
    except Exception:
        off = 0.0

    gs = sample.get("gt_segment_start", None)
    ge = sample.get("gt_segment_end", None)

    # STRICT: window always ends at current time (t_eval_sched)
    we = float(t_eval_sched)

    if gs is not None and ge is not None:
        # map segment bounds into the same time space as t_eval_sched if offset!=0
        gs_sched = float(_f(gs, 0.0)) + float(off)
        # ge is not used to end the window anymore, but we still sanity-clamp start
        # (keep for potential future debug)
        _ = float(_f(ge, gs_sched)) + float(off)

        ws = float(gs_sched)
        # clamp to [0, we]
        if ws < 0.0:
            ws = 0.0
        if ws > we:
            ws = we
        return ws, we, t_target

    # fallback when segment bounds are missing:
    # start around target time (evidence proxy), but STILL end at t_eval_sched
    pad_pre = 2.0
    ws = float(t_target - pad_pre)
    if ws < 0.0:
        ws = 0.0
    if ws > we:
        ws = we
    return ws, we, t_target

class _NowHelper:
    """
    Runner contract:
      - prepare_queryset(queryset_path, out_path, task) -> Path
    """

    @staticmethod
    def _build_prompt_now_xml_v1(t_rel: float) -> str:
        return NOW_XML_TEMPLATE_V1.format(t_eval=float(t_rel))

    @staticmethod
    def _build_prompt_now_state_only_cand(t_rel: float) -> str:
        return NOW_STATE_ONLY_CAND_TEMPLATE.format(t_eval=float(t_rel))

    @staticmethod
    def _build_prompt_now_action_mcq(t_rel: float, options: List[str]) -> str:
        if not isinstance(options, list) or len(options) != 4:
            raise ValueError("MCP sample.options must be list[str] length 4")
        return NOW_ACTION_MCQ_TEMPLATE.format(
            t_eval=float(t_rel),
            A_OPT=str(options[0]),
            B_OPT=str(options[1]),
            C_OPT=str(options[2]),
            D_OPT=str(options[3]),
        )

    @staticmethod
    def infer_time_offset(*, time_mode: str, interval_start_sec: float) -> float:
        tm = (time_mode or "clip").lower()
        if tm == "full":
            return float(interval_start_sec)
        return 0.0

    @staticmethod
    def _memory_mode_enabled() -> bool:
        return _env_get("MEMORY_MODE", "0").strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _expected_memory_keyframe_timestamps(start_sec: float, end_sec: float, fps: float, sampling: str) -> List[float]:
        start = float(max(0.0, start_sec))
        end = float(max(start, end_sec))
        fps = float(max(1e-6, fps))
        sampling = (sampling or "fixed_fps_floor_plus_one_v1").strip().lower()
        if sampling != "fixed_fps_floor_plus_one_v1":
            sampling = "fixed_fps_floor_plus_one_v1"
        duration = float(max(0.0, end - start))
        count = int(max(1, int(duration * fps + 1e-9) + 1))
        step = 1.0 / fps
        out: List[float] = []
        for i in range(count):
            ts = min(end, start + float(i) * step)
            if ts <= end + 1e-6:
                value = round(float(ts), 6)
                if not out or abs(out[-1] - value) > 1e-6:
                    out.append(value)
        if not out:
            out.append(round(end, 6))
        return out

    def _attach_memory_metadata(
        self,
        sample: Dict[str, Any],
        *,
        task: str,
        params: Dict[str, Any],
        gt_obj: Dict[str, Any],
        gt_path: Path,
    ) -> None:
        del params
        if not self._memory_mode_enabled():
            return

        t_eval_rel = _f(sample.get("t_eval_rel", sample.get("t_eval", 0.0)), 0.0)
        dense_sec = max(0.0, _f(_env_get("MEMORY_DENSE_SEC", "10"), 10.0))
        dense_fps = max(1e-6, _f(_env_get("MEMORY_DENSE_FPS", "1.5"), 1.5))
        dense_num_frames = max(1, _safe_int(_env_get("MEMORY_DENSE_NUM_FRAMES", "15"), 15))
        sparse_sec = max(0.0, _f(_env_get("MEMORY_SPARSE_SEC", "30"), 30.0))
        keyframe_fps = max(1e-6, _f(_env_get("MEMORY_KEYFRAME_FPS", "0.25"), 0.25))
        keyframe_sampling = _env_get("MEMORY_KEYFRAME_SAMPLING", "fixed_fps_floor_plus_one_v1")
        caption_source = _env_get("MEMORY_CAPTION_SOURCE", "openrouter_generated")
        caption_input_type = _env_get("MEMORY_CAPTION_INPUT_TYPE", "keyframes")
        caption_model = _env_get("MEMORY_CAPTION_MODEL", "google/gemini-3.1-pro-preview")
        caption_prompt_version = _env_get("MEMORY_CAPTION_PROMPT_VERSION", "memory_caption_v1")
        caption_cache_root = _env_get("MEMORY_CAPTION_CACHE_ROOT", "memory_cache/captions")
        setting_name = _env_get("MEMORY_SETTING_NAME", f"memory_d{int(dense_sec):g}_s{int(sparse_sec):g}")

        dense_end = float(max(0.0, t_eval_rel))
        dense_start = float(max(0.0, dense_end - dense_sec))
        sparse_end = float(dense_start)
        sparse_start = float(max(0.0, sparse_end - sparse_sec))
        keyframe_ts = self._expected_memory_keyframe_timestamps(sparse_start, sparse_end, keyframe_fps, keyframe_sampling)
        keyframe_ts = [float(ts) for ts in keyframe_ts if float(ts) <= sparse_end + 1e-6]

        vm = gt_obj.get("video_metadata", {}) if isinstance(gt_obj.get("video_metadata"), dict) else {}
        split = str(vm.get("split", "") or gt_obj.get("split", "") or "val").strip().lower() or "val"
        video_uid = gt_obj.get("video_uid", None)
        if video_uid in (None, ""):
            video_uid = vm.get("video_uid", None)

        sample_idx = sample.get("idx", sample.get("sample_index", None))
        sample_id = sample.get("sample_id", sample.get("id", sample_idx))
        interval_start = vm.get("interval_start_sec", None)
        interval_end = vm.get("interval_end_sec", None)

        coverage_note = None
        task_name = str(task or gt_obj.get("task_name") or gt_obj.get("task") or "").strip().lower()
        if task_name == "ms_rtrv":
            lag_sec = _f(sample.get("lag_sec", sample.get("lookback_sec", 0.0)), 0.0)
            if lag_sec > 0 and (dense_sec + sparse_sec) < lag_sec - 1e-6:
                coverage_note = "coverage_limited_long_lag"

        memory_meta: Dict[str, Any] = {
            "enabled": True,
            "setting_name": setting_name,
            "schema_version": "memory_proxy_v1",
            "time_space": "clip_relative",
            "dense_window": {
                "start_sec": float(dense_start),
                "end_sec": float(dense_end),
                "duration_sec": float(dense_sec),
                "fps": float(dense_fps),
                "num_frames": int(dense_num_frames),
            },
            "sparse_window": {
                "start_sec": float(sparse_start),
                "end_sec": float(sparse_end),
                "duration_sec": float(sparse_sec),
                "keyframe_fps": float(keyframe_fps),
                "keyframe_sampling": str(keyframe_sampling),
                "expected_keyframe_timestamps_sec": keyframe_ts,
            },
            "caption_source": str(caption_source),
            "caption_input_type": str(caption_input_type),
            "caption_model": str(caption_model),
            "caption_prompt_version": str(caption_prompt_version),
            "no_lookahead": bool(dense_end <= t_eval_rel + 1e-6 and sparse_end <= dense_start + 1e-6 and all(ts <= sparse_end + 1e-6 for ts in keyframe_ts)),
            "coverage_note": coverage_note,
            "cache_context": {
                "task": task_name or None,
                "split": split,
                "video_uid": video_uid,
                "clip_uid": vm.get("clip_uid", None),
                "clip_id": vm.get("clip_id", None),
                "gt_file_stem": gt_path.stem,
                "sample_idx": sample_idx,
                "sample_id": sample_id,
                "t_eval_rel": float(t_eval_rel),
                "interval_start_sec": interval_start,
                "interval_end_sec": interval_end,
            },
        }

        segment_key = build_segment_key(memory_meta)
        usage_key = build_usage_key(memory_meta)
        memory_meta["segment_key"] = segment_key
        memory_meta["usage_key"] = usage_key
        memory_meta["caption_cache_key"] = segment_key
        memory_meta["sparse_window"]["caption_cache_key"] = segment_key
        memory_meta["sparse_window"]["caption_cache_path"] = caption_record_path(caption_cache_root, memory_meta)

        sample["memory"] = memory_meta

    def _maybe_attach_memory_queryset(self, new_gt: Dict[str, Any], *, task: str, gt_obj: Dict[str, Any], gt_path: Path) -> None:
        if not self._memory_mode_enabled():
            return
        params = new_gt.get("params", {})
        if not isinstance(params, dict):
            params = {}
            new_gt["params"] = params
        params["memory_mode"] = True
        params["memory_setting_name"] = _env_get("MEMORY_SETTING_NAME", "")
        params["memory_dense_sec"] = float(_f(_env_get("MEMORY_DENSE_SEC", "10"), 10.0))
        params["memory_dense_fps"] = float(_f(_env_get("MEMORY_DENSE_FPS", "1.5"), 1.5))
        params["memory_dense_num_frames"] = int(_safe_int(_env_get("MEMORY_DENSE_NUM_FRAMES", "15"), 15))
        params["memory_sparse_sec"] = float(_f(_env_get("MEMORY_SPARSE_SEC", "30"), 30.0))
        params["memory_keyframe_fps"] = float(_f(_env_get("MEMORY_KEYFRAME_FPS", "0.25"), 0.25))
        params["memory_keyframe_sampling"] = _env_get("MEMORY_KEYFRAME_SAMPLING", "fixed_fps_floor_plus_one_v1")

        samples = new_gt.get("samples", [])
        if isinstance(samples, list):
            for sample in samples:
                if isinstance(sample, dict):
                    self._attach_memory_metadata(sample, task=task, params=params, gt_obj=gt_obj, gt_path=gt_path)

        helper_meta = new_gt.get("helper", {})
        if not isinstance(helper_meta, dict):
            helper_meta = {}
            new_gt["helper"] = helper_meta
        helper_meta["memory_proxy"] = {
            "enabled": True,
            "schema_version": "memory_proxy_v1",
            "setting_name": _env_get("MEMORY_SETTING_NAME", ""),
            "caption_cache_root": _env_get("MEMORY_CAPTION_CACHE_ROOT", "memory_cache/captions"),
        }

    @staticmethod
    def _pick_lookback_sec_for_sh_rtrv(sample: Dict[str, Any]) -> float:
        env_v = os.environ.get("SH_RTRV_LOOKBACK_SEC", "").strip()
        if env_v:
            return _f(env_v, 20.0)
        if isinstance(sample, dict) and "lookback_sec" in sample:
            return _f(sample.get("lookback_sec"), 20.0)
        return 20.0

    # ---- sh_rtrv prompt builders ----
    @staticmethod
    def _build_prompt_past_open(t_rel: float, lookback_sec: float) -> str:
        return PAST_RTRV_OPEN_TEMPLATE.format(t_eval=float(t_rel), lookback_sec=float(lookback_sec))

    @staticmethod
    def _build_prompt_past_cand(t_rel: float, lookback_sec: float, options: List[str]) -> str:
        if not isinstance(options, list) or len(options) != 4:
            raise ValueError("MCP sample.options must be list[str] length 4")
        return PAST_RTRV_CAND_TEMPLATE.format(
            t_eval=float(t_rel),
            lookback_sec=float(lookback_sec),
            A_OPT=str(options[0]),
            B_OPT=str(options[1]),
            C_OPT=str(options[2]),
            D_OPT=str(options[3]),
        )

    # ---- ms_rtrv prompt builders ----
    @staticmethod
    def _build_prompt_ms_open(t_rel: float, lag_sec: float, sample: Dict[str, Any]) -> str:
        step_idx = _safe_int(sample.get("step_idx", None), -1)
        group_size = _safe_int(sample.get("group_size", None), -1)
        anchor_idx = _safe_int(sample.get("anchor_idx", None), -1)

        base = MS_RTRV_OPEN_TEMPLATE.format(t_eval=float(t_rel), lookback_sec=float(lag_sec))
        if step_idx >= 0 or group_size > 0 or anchor_idx >= 0:
            extra = []
            if step_idx >= 0 and group_size > 0:
                extra.append(f"step={step_idx+1}/{group_size}")
            elif step_idx >= 0:
                extra.append(f"step={step_idx}")
            if anchor_idx >= 0:
                extra.append(f"anchor_idx={anchor_idx}")
            if extra:
                base = base.replace(
                    "\nBased on the egocentric video up to now, recall",
                    "\n[" + " | ".join(extra) + "]\nBased on the egocentric video up to now, recall",
                    1,
                )
        return base

    @staticmethod
    def _build_prompt_ms_cand(t_rel: float, lag_sec: float, options: List[str], sample: Dict[str, Any]) -> str:
        if not isinstance(options, list) or len(options) != 4:
            raise ValueError("MCP sample.options must be list[str] length 4")

        step_idx = _safe_int(sample.get("step_idx", None), -1)
        group_size = _safe_int(sample.get("group_size", None), -1)
        anchor_idx = _safe_int(sample.get("anchor_idx", None), -1)

        base = MS_RTRV_CAND_TEMPLATE.format(
            t_eval=float(t_rel),
            lookback_sec=float(lag_sec),
            A_OPT=str(options[0]),
            B_OPT=str(options[1]),
            C_OPT=str(options[2]),
            D_OPT=str(options[3]),
        )
        if step_idx >= 0 or group_size > 0 or anchor_idx >= 0:
            extra = []
            if step_idx >= 0 and group_size > 0:
                extra.append(f"step={step_idx+1}/{group_size}")
            elif step_idx >= 0:
                extra.append(f"step={step_idx}")
            if anchor_idx >= 0:
                extra.append(f"anchor_idx={anchor_idx}")
            if extra:
                base = base.replace(
                    "\nBased on the egocentric video up to now, choose",
                    "\n[" + " | ".join(extra) + "]\nBased on the egocentric video up to now, choose",
                    1,
                )
        return base

    # ---- ms_pred prompt builders ----
    @staticmethod
    def _build_prompt_pred_open(t_rel: float, horizon_sec: float, sample: Dict[str, Any]) -> str:
        step_idx = _safe_int(sample.get("step_idx", None), -1)
        group_size = _safe_int(sample.get("group_size", None), -1)
        anchor_idx = _safe_int(sample.get("anchor_idx", None), -1)

        base = MS_PRED_OPEN_TEMPLATE.format(t_eval=float(t_rel), lookback_sec=float(horizon_sec))
        if step_idx >= 0 or group_size > 0 or anchor_idx >= 0:
            extra = []
            if step_idx >= 0 and group_size > 0:
                extra.append(f"step={step_idx+1}/{group_size}")
            elif step_idx >= 0:
                extra.append(f"step={step_idx}")
            if anchor_idx >= 0:
                extra.append(f"anchor_idx={anchor_idx}")
            if extra:
                base = base.replace(
                    "\nBased on the egocentric video up to now, predict",
                    "\n[" + " | ".join(extra) + "]\nBased on the egocentric video up to now, predict",
                    1,
                )
        return base

    @staticmethod
    def _build_prompt_pred_cand(t_rel: float, horizon_sec: float, options: List[str], sample: Dict[str, Any]) -> str:
        if not isinstance(options, list) or len(options) != 4:
            raise ValueError("MCP sample.options must be list[str] length 4")

        step_idx = _safe_int(sample.get("step_idx", None), -1)
        group_size = _safe_int(sample.get("group_size", None), -1)
        anchor_idx = _safe_int(sample.get("anchor_idx", None), -1)

        base = MS_PRED_CAND_TEMPLATE.format(
            t_eval=float(t_rel),
            lookback_sec=float(horizon_sec),
            A_OPT=str(options[0]),
            B_OPT=str(options[1]),
            C_OPT=str(options[2]),
            D_OPT=str(options[3]),
        )
        if step_idx >= 0 or group_size > 0 or anchor_idx >= 0:
            extra = []
            if step_idx >= 0 and group_size > 0:
                extra.append(f"step={step_idx+1}/{group_size}")
            elif step_idx >= 0:
                extra.append(f"step={step_idx}")
            if anchor_idx >= 0:
                extra.append(f"anchor_idx={anchor_idx}")
            if extra:
                base = base.replace(
                    "\nBased on the egocentric video up to now, choose",
                    "\n[" + " | ".join(extra) + "]\nBased on the egocentric video up to now, choose",
                    1,
                )
        return base

    # ---- sh_pred prompt builders ----
    @staticmethod
    def _build_prompt_sh_pred_open(t_rel: float, horizon_sec: float) -> str:
        return SH_PRED_OPEN_TEMPLATE.format(t_eval=float(t_rel), lookback_sec=float(horizon_sec))

    @staticmethod
    def _build_prompt_sh_pred_cand(t_rel: float, horizon_sec: float, options: List[str]) -> str:
        if not isinstance(options, list) or len(options) != 4:
            raise ValueError("MCP sample.options must be list[str] length 4")
        return SH_PRED_CAND_TEMPLATE.format(
            t_eval=float(t_rel),
            lookback_sec=float(horizon_sec),
            A_OPT=str(options[0]),
            B_OPT=str(options[1]),
            C_OPT=str(options[2]),
            D_OPT=str(options[3]),
        )

    def prepare_queryset(self, queryset_path: Path, out_path: Path, task: str) -> Path:
        task = _normalize_task_name(task)

        src = Path(queryset_path).expanduser()
        dst = Path(out_path).expanduser()
        dst.parent.mkdir(parents=True, exist_ok=True)

        gt = _load_json_or_one_jsonl(src)

        vm = gt.get("video_metadata", {}) if isinstance(gt.get("video_metadata"), dict) else {}
        interval_start_sec = _f(vm.get("interval_start_sec", 0.0), 0.0)
        interval_len_sec = _f(vm.get("interval_len_sec", 0.0), 0.0)

        time_mode = _env_get("NOW_TIME_MODE", "clip")
        offset = self.infer_time_offset(time_mode=time_mode, interval_start_sec=interval_start_sec)

        samples = gt.get("samples", [])
        if not isinstance(samples, list):
            raise ValueError(f"GT 'samples' must be a list: {src}")

        # --------------------------
        # NOW tasks
        # --------------------------
        if task in {"now_narration", "now_state_switch"}:
            prompt_style = _env_get("NOW_PROMPT_STYLE", "xml_v1").lower()
            lookback_sec = _f(_env_get("NOW_LOOKBACK_SEC", "20.0"), 20.0)
            filter_rejected = _env_get("NOW_FILTER_REJECTED", "1") == "1"

            kept: List[Dict[str, Any]] = []
            for s in samples:
                if not isinstance(s, dict):
                    continue
                if bool(s.get("is_invalid_annotation", False)):
                    continue
                if _should_reject_now_sample(s, filter_rejected):
                    continue
                if "t_eval" not in s:
                    continue
                kept.append(s)

            if not kept:
                raise ValueError(f"No usable samples found in GT: {src}")

            kept_sorted = sorted(
                kept,
                key=lambda x: (int(x.get("idx", 10**9)), _f(x.get("t_eval", 0.0), 0.0)),
            )

            new_gt = copy.deepcopy(gt)
            new_gt["task"] = task
            new_gt["task_name"] = task

            params = new_gt.get("params", {})
            if not isinstance(params, dict):
                params = {}
            params["lookback_sec"] = float(lookback_sec)
            params["time_mode"] = str(time_mode)
            params["time_offset_sec"] = float(offset)
            params["interval_len_sec"] = float(interval_len_sec)
            new_gt["params"] = params

            new_gt["helper"] = {
                "prompt_style": prompt_style,
                "time_mode": str(time_mode),
                "time_offset_sec": float(offset),
                "note": "Designed for interval-clip runner: keep t_eval relative unless NOW_TIME_MODE=full.",
            }

            # ---- now_narration: flavors (open / cand_state / cand_mcq) ----
            if task == "now_narration":
                now_flavor = _normalize_now_narration_flavor(_env_get("PRED_FLAVOR", "open"))
                new_gt["helper"]["pred_flavor"] = now_flavor
                new_gt["helper"]["suggested_subdir"] = now_flavor
                new_gt["params"]["pred_flavor"] = now_flavor

                # common dedup by scheduled time
                used_times = set()

                # ----------------
                # open: keep old behavior (downsample + global stop + xml_v1 prompt)
                # ----------------
                if now_flavor == "open":
                    new_samples: List[Dict[str, Any]] = []

                    ds_task = "now_narration"
                    ds_every, ds_keep = _task_ds_params(ds_task)

                    for s in kept_sorted:
                        t_rel = _f(s.get("t_eval", 0.0), 0.0)
                        t_sched = float(offset + t_rel)

                        if t_sched in used_times:
                            continue

                        if not _ds_should_keep_unit(ds_task):
                            continue

                        used_times.add(t_sched)

                        ns = dict(s)
                        ns["t_eval_rel"] = float(t_rel)
                        ns["t_eval"] = float(t_sched)

                        if prompt_style == "xml_v1":
                            ns["prompt"] = self._build_prompt_now_xml_v1(t_rel)
                        else:
                            ns["prompt"] = self._build_prompt_now_xml_v1(t_rel)

                        new_samples.append(ns)

                    new_gt["samples"] = new_samples
                    new_gt["num_samples"] = int(len(new_samples))

                    before, after, target, reached, already = _ds_update_cumulative(ds_task, len(new_samples))
                    new_gt["helper"]["downsample"] = {
                        "enabled": True,
                        "task": ds_task,
                        "unit": "sample",
                        "policy": "keep_first_k_in_every_n (uniform, global stream across GT files)",
                        "every": int(ds_every),
                        "keep": int(ds_keep),
                        "phase_after": int(_GLOBAL_DS_PHASE.get(ds_task, 0)),
                    }
                    new_gt["helper"]["global_limit"] = {
                        "target_samples": int(target),
                        "cumulative_before": int(before),
                        "cumulative_after": int(after),
                        "already_reached_before_this_file": bool(already),
                        "stop_after_this_file": bool(reached),
                    }
                    new_gt["helper"]["stop_after_this_file"] = bool(reached)

                    self._maybe_attach_memory_queryset(new_gt, task=task, gt_obj=gt, gt_path=src)
                    with open(dst, "w", encoding="utf-8") as f:
                        json.dump(new_gt, f, ensure_ascii=False, indent=2)
                    return dst

                # ----------------
                # cand_state: state-only template, SAME downsample + global stop as open
                # ----------------
                if now_flavor == "cand_state":
                    new_samples: List[Dict[str, Any]] = []

                    ds_task = "now_narration"
                    ds_every, ds_keep = _task_ds_params(ds_task)

                    for s in kept_sorted:
                        t_rel = _f(s.get("t_eval", 0.0), 0.0)
                        t_sched = float(offset + t_rel)

                        if t_sched in used_times:
                            continue

                        if not _ds_should_keep_unit(ds_task):
                            continue

                        used_times.add(t_sched)

                        ns = dict(s)
                        ns["t_eval_rel"] = float(t_rel)
                        ns["t_eval"] = float(t_sched)

                        ns["prompt"] = self._build_prompt_now_state_only_cand(t_rel)

                        new_samples.append(ns)

                    new_gt["samples"] = new_samples
                    new_gt["num_samples"] = int(len(new_samples))

                    before, after, target, reached, already = _ds_update_cumulative(ds_task, len(new_samples))
                    new_gt["helper"]["downsample"] = {
                        "enabled": True,
                        "task": ds_task,
                        "unit": "sample",
                        "policy": "keep_first_k_in_every_n (uniform, global stream across GT files)",
                        "every": int(ds_every),
                        "keep": int(ds_keep),
                        "phase_after": int(_GLOBAL_DS_PHASE.get(ds_task, 0)),
                    }
                    new_gt["helper"]["global_limit"] = {
                        "target_samples": int(target),
                        "cumulative_before": int(before),
                        "cumulative_after": int(after),
                        "already_reached_before_this_file": bool(already),
                        "stop_after_this_file": bool(reached),
                    }
                    new_gt["helper"]["stop_after_this_file"] = bool(reached)

                    self._maybe_attach_memory_queryset(new_gt, task=task, gt_obj=gt, gt_path=src)
                    with open(dst, "w", encoding="utf-8") as f:
                        json.dump(new_gt, f, ensure_ascii=False, indent=2)
                    return dst

                # ----------------
                # cand_mcq: action MCQ, ONLY for region==segment; read MCP from now_narration_action
                # NOW: apply SAME downsample (2/3) + global stop (>=800), but in an INDEPENDENT stream.
                # ----------------
                split = str(vm.get("split", "")) or str(gt.get("split", "")) or "val"
                split = split.strip().lower() or "val"

                mcp_path = _resolve_mcp_path(src, split, task="now_narration_action")
                mcp_obj = _load_json_or_one_jsonl(mcp_path)
                mcp_by_idx = _index_samples_by_idx(mcp_obj.get("samples", []))

                new_gt["params"]["mcp_path"] = str(mcp_path)
                new_gt["helper"]["mcp_root"] = _env_get("MCP_ROOT", _default_mcp_root_for_task("now_narration_action"))
                new_gt["helper"]["mcp_task"] = "now_narration_action"
                new_gt["helper"]["cand_only_region"] = "segment"

                # NEW: independent DS stream for cand_mcq
                ds_task = "now_narration_mcq"
                ds_every, ds_keep = _task_ds_params(ds_task)

                def _region_of(x: Dict[str, Any]) -> str:
                    return str(x.get("region", "")).strip().lower()

                new_samples: List[Dict[str, Any]] = []

                # common dedup by scheduled time (same GT-file scope; fine)
                used_times = set()

                for s in kept_sorted:
                    if _region_of(s) != "segment":
                        continue

                    t_rel = _f(s.get("t_eval", 0.0), 0.0)
                    t_sched = float(offset + t_rel)

                    if t_sched in used_times:
                        continue

                    if not _ds_should_keep_unit(ds_task):
                        continue

                    used_times.add(t_sched)

                    idx = _safe_int(s.get("idx", -1), -1)
                    if idx < 0:
                        raise ValueError(f"GT sample missing idx: {src}")
                    mc = mcp_by_idx.get(idx)
                    if mc is None:
                        raise KeyError(f"MCP missing idx={idx} for {src.name}")

                    options = mc.get("options", None)
                    if not isinstance(options, list) or len(options) != 4:
                        raise ValueError(f"MCP idx={idx} options must be list len 4: {mcp_path}")

                    ns = dict(s)
                    ns["t_eval_rel"] = float(t_rel)
                    ns["t_eval"] = float(t_sched)

                    ns["prompt"] = self._build_prompt_now_action_mcq(t_rel, [str(x) for x in options])

                    ns["mcq"] = {
                        "mcp_file": str(mcp_path),
                        "answer": mc.get("answer", None),
                        "answer_idx": mc.get("answer_idx", None),
                        "option_sources": mc.get("option_sources", None),
                        "options": options,
                    }

                    new_samples.append(ns)

                if not new_samples:
                    raise ValueError(f"No usable samples found in GT: {src}")

                new_gt["samples"] = new_samples
                new_gt["num_samples"] = int(len(new_samples))

                before, after, target, reached, already = _ds_update_cumulative(ds_task, len(new_samples))
                new_gt["helper"]["downsample"] = {
                    "enabled": True,
                    "task": ds_task,
                    "unit": "sample",
                    "policy": "keep_first_k_in_every_n (uniform, global stream across GT files)",
                    "every": int(ds_every),
                    "keep": int(ds_keep),
                    "phase_after": int(_GLOBAL_DS_PHASE.get(ds_task, 0)),
                }
                new_gt["helper"]["global_limit"] = {
                    "target_samples": int(target),
                    "cumulative_before": int(before),
                    "cumulative_after": int(after),
                    "already_reached_before_this_file": bool(already),
                    "stop_after_this_file": bool(reached),
                }
                new_gt["helper"]["stop_after_this_file"] = bool(reached)

                self._maybe_attach_memory_queryset(new_gt, task=task, gt_obj=gt, gt_path=src)
                with open(dst, "w", encoding="utf-8") as f:
                    json.dump(new_gt, f, ensure_ascii=False, indent=2)
                return dst

            # ---- now_state_switch: add cand_state prompt option (no other logic changes) ----
            params_in = gt.get("params", {})
            if not isinstance(params_in, dict):
                params_in = {}
            query_freq = float(_f(params_in.get("query_freq", 8.0), 8.0))
            if query_freq <= 0:
                query_freq = 8.0

            ss_flavor = _normalize_now_state_switch_flavor(_env_get("PRED_FLAVOR", "open"))
            new_gt["helper"]["pred_flavor"] = ss_flavor
            new_gt["helper"]["suggested_subdir"] = ss_flavor
            new_gt["params"]["pred_flavor"] = ss_flavor

            # choose prompt builder (shared template with now_narration cand_state)
            prompt_builder = self._build_prompt_now_state_only_cand if ss_flavor == "cand_state" else self._build_prompt_now_xml_v1

            ss_pos_raw = os.environ.get("SS_POS", "").strip()
            if not ss_pos_raw:
                ss_pos_raw = os.environ.get("NOW_SS_POS", "").strip()
            if not ss_pos_raw:
                ss_pos_raw = "r:0.0"
            ss_mode, ss_value, ss_raw_echo, ss_used_heur = _parse_ss_pos(ss_pos_raw)

            by_key: Dict[int, List[Dict[str, Any]]] = {}
            for s in kept_sorted:
                t0 = float(_f(s.get("t_eval", 0.0), 0.0))
                k = _time_key(t0, query_freq)
                by_key.setdefault(k, []).append(s)

            def _region_of(x: Dict[str, Any]) -> str:
                return str(x.get("region", "")).strip().lower()

            def _has_region(samples_list: List[Dict[str, Any]], region: str) -> bool:
                r = (region or "").strip().lower()
                for xx in samples_list:
                    if _region_of(xx) == r:
                        return True
                return False

            filtered: List[Dict[str, Any]] = []
            filtered_by_key: Dict[int, List[Dict[str, Any]]] = {}
            neighbor_kept = 0
            neighbor_dropped = 0

            for s in kept_sorted:
                r = _region_of(s)
                if r not in {"segment", "gap"}:
                    neighbor_dropped += 1
                    continue

                t0 = float(_f(s.get("t_eval", 0.0), 0.0))
                k = _time_key(t0, query_freq)
                prev_list = by_key.get(k - 1, [])
                next_list = by_key.get(k + 1, [])

                if r == "segment":
                    keep = _has_region(prev_list, "gap") or _has_region(next_list, "gap")
                else:
                    keep = _has_region(prev_list, "segment") or _has_region(next_list, "segment")

                if keep:
                    filtered.append(s)
                    filtered_by_key.setdefault(k, []).append(s)
                    neighbor_kept += 1
                else:
                    neighbor_dropped += 1

            def _sorted_by_idx_then_t(lst: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                return sorted(
                    lst,
                    key=lambda x: (_safe_int(x.get("idx", 10**9), 10**9), float(_f(x.get("t_eval", 0.0), 0.0))),
                )

            keys_sorted = sorted(filtered_by_key.keys())
            pairs: List[Tuple[str, int, Dict[str, Any], Dict[str, Any]]] = []

            for k in keys_sorted:
                curr = filtered_by_key.get(k, [])
                nxt = filtered_by_key.get(k + 1, [])
                if not curr or not nxt:
                    continue

                curr_sorted = _sorted_by_idx_then_t(curr)
                nxt_sorted = _sorted_by_idx_then_t(nxt)

                gaps_curr = [x for x in curr_sorted if _region_of(x) == "gap"]
                segs_curr = [x for x in curr_sorted if _region_of(x) == "segment"]
                gaps_next = [x for x in nxt_sorted if _region_of(x) == "gap"]
                segs_next = [x for x in nxt_sorted if _region_of(x) == "segment"]

                for g in gaps_curr:
                    for s in segs_next:
                        pairs.append(("gap_to_segment", k, g, s))

                for s in segs_curr:
                    for g in gaps_next:
                        pairs.append(("segment_to_gap", k, s, g))

            new_samples: List[Dict[str, Any]] = []

            def _get_gap_bounds(g: Dict[str, Any], t_center: float) -> Tuple[float, float]:
                return _get_bounds(
                    g,
                    start_keys=("gap_block_start", "gap_block_start_sec", "gap_start_sec", "gap_start", "gap_start_time"),
                    end_keys=("gap_block_end", "gap_block_end_sec", "gap_end_sec", "gap_end", "gap_end_time"),
                    t_center=t_center,
                    dt=query_freq,
                    interval_len_sec=interval_len_sec,
                )

            def _get_seg_bounds(s: Dict[str, Any], t_center: float) -> Tuple[float, float]:
                return _get_bounds(
                    s,
                    start_keys=("gt_segment_start", "segment_start_sec", "segment_start", "seg_start_sec", "seg_start"),
                    end_keys=("gt_segment_end", "segment_end_sec", "segment_end", "seg_end_sec", "seg_end"),
                    t_center=t_center,
                    dt=query_freq,
                    interval_len_sec=interval_len_sec,
                )

            def _mk_pair_id(pair_type: str, a: Dict[str, Any], b: Dict[str, Any]) -> str:
                a_idx = _safe_int(a.get("idx", -1), -1)
                b_idx = _safe_int(b.get("idx", -1), -1)
                a_t = float(_f(a.get("t_eval", 0.0), 0.0))
                b_t = float(_f(b.get("t_eval", 0.0), 0.0))
                return f"{pair_type}|a_idx={a_idx}|b_idx={b_idx}|a_t={a_t:.3f}|b_t={b_t:.3f}|dt={query_freq:.3f}"

            def _final_clamp(t: float) -> float:
                t = float(t)
                if interval_len_sec and interval_len_sec > 0:
                    return float(_clamp(t, 0.0, float(interval_len_sec)))
                return max(0.0, t)

            pairs_sorted = sorted(
                pairs,
                key=lambda x: (
                    x[1],
                    0 if x[0] == "gap_to_segment" else 1,
                    _safe_int(x[2].get("idx", 10**9), 10**9),
                    _safe_int(x[3].get("idx", 10**9), 10**9),
                ),
            )

            for pair_type, k, a, b in pairs_sorted:
                pair_id = _mk_pair_id(pair_type, a, b)

                if pair_type == "gap_to_segment":
                    g = a
                    s = b
                    g_t0 = float(_f(g.get("t_eval", 0.0), 0.0))
                    s_t0 = float(_f(s.get("t_eval", 0.0), 0.0))

                    g_start, g_end = _get_gap_bounds(g, g_t0)
                    s_start, s_end = _get_seg_bounds(s, s_t0)

                    t_gap_prime = _final_clamp(0.5 * (g_start + g_end))
                    t_seg_prime = _pos_in_span(s_start, s_end, ss_mode, ss_value)
                    t_seg_prime = _final_clamp(_clamp(t_seg_prime, s_start, s_end))

                    ns_g = dict(g)
                    ns_g["t_eval_orig"] = float(g_t0)
                    ns_g["t_eval_rel_orig"] = float(g_t0)
                    ns_g["t_eval_rel"] = float(t_gap_prime)
                    ns_g["t_eval"] = float(offset + t_gap_prime)
                    ns_g["prompt"] = prompt_builder(t_gap_prime)

                    ns_g["state_switch_pair_type"] = "gap_to_segment"
                    ns_g["state_switch_role"] = "fixed_gap_mid"
                    ns_g["state_switch_param"] = {"raw": ss_raw_echo, "mode": ss_mode, "value": float(ss_value), "numeric_heuristic": bool(ss_used_heur)}
                    ns_g["state_switch_neighbor_dt"] = float(query_freq)
                    ns_g["state_switch_pair_id"] = pair_id

                    ns_g["state_switch_trace"] = {
                        "gap": {
                            "idx": _safe_int(g.get("idx", -1), -1),
                            "t_eval_orig": float(g_t0),
                            "gap_start": float(g_start),
                            "gap_end": float(g_end),
                        },
                        "segment": {
                            "idx": _safe_int(s.get("idx", -1), -1),
                            "t_eval_orig": float(s_t0),
                            "segment_start": float(s_start),
                            "segment_end": float(s_end),
                        },
                        "rewritten": {"t_gap_prime": float(t_gap_prime), "t_seg_prime": float(t_seg_prime)},
                    }
                    new_samples.append(ns_g)

                    ns_s = dict(s)
                    ns_s["t_eval_orig"] = float(s_t0)
                    ns_s["t_eval_rel_orig"] = float(s_t0)
                    ns_s["t_eval_rel"] = float(t_seg_prime)
                    ns_s["t_eval"] = float(offset + t_seg_prime)
                    ns_s["prompt"] = prompt_builder(t_seg_prime)

                    ns_s["state_switch_pair_type"] = "gap_to_segment"
                    ns_s["state_switch_role"] = "scan_segment"
                    ns_s["state_switch_param"] = {"raw": ss_raw_echo, "mode": ss_mode, "value": float(ss_value), "numeric_heuristic": bool(ss_used_heur)}
                    ns_s["state_switch_neighbor_dt"] = float(query_freq)
                    ns_s["state_switch_pair_id"] = pair_id

                    ns_s["state_switch_trace"] = {
                        "gap": {
                            "idx": _safe_int(g.get("idx", -1), -1),
                            "t_eval_orig": float(g_t0),
                            "gap_start": float(g_start),
                            "gap_end": float(g_end),
                        },
                        "segment": {
                            "idx": _safe_int(s.get("idx", -1), -1),
                            "t_eval_orig": float(s_t0),
                            "segment_start": float(s_start),
                            "segment_end": float(s_end),
                        },
                        "rewritten": {"t_gap_prime": float(t_gap_prime), "t_seg_prime": float(t_seg_prime)},
                    }
                    new_samples.append(ns_s)

                else:
                    s = a
                    g = b
                    s_t0 = float(_f(s.get("t_eval", 0.0), 0.0))
                    g_t0 = float(_f(g.get("t_eval", 0.0), 0.0))

                    s_start, s_end = _get_seg_bounds(s, s_t0)
                    g_start, g_end = _get_gap_bounds(g, g_t0)

                    t_seg_prime = _final_clamp(0.5 * (s_start + s_end))
                    t_gap_prime = _pos_in_span(g_start, g_end, ss_mode, ss_value)
                    t_gap_prime = _final_clamp(_clamp(t_gap_prime, g_start, g_end))

                    ns_s = dict(s)
                    ns_s["t_eval_orig"] = float(s_t0)
                    ns_s["t_eval_rel_orig"] = float(s_t0)
                    ns_s["t_eval_rel"] = float(t_seg_prime)
                    ns_s["t_eval"] = float(offset + t_seg_prime)
                    ns_s["prompt"] = prompt_builder(t_seg_prime)

                    ns_s["state_switch_pair_type"] = "segment_to_gap"
                    ns_s["state_switch_role"] = "fixed_segment_mid"
                    ns_s["state_switch_param"] = {"raw": ss_raw_echo, "mode": ss_mode, "value": float(ss_value), "numeric_heuristic": bool(ss_used_heur)}
                    ns_s["state_switch_neighbor_dt"] = float(query_freq)
                    ns_s["state_switch_pair_id"] = pair_id

                    ns_s["state_switch_trace"] = {
                        "segment": {
                            "idx": _safe_int(s.get("idx", -1), -1),
                            "t_eval_orig": float(s_t0),
                            "segment_start": float(s_start),
                            "segment_end": float(s_end),
                        },
                        "gap": {
                            "idx": _safe_int(g.get("idx", -1), -1),
                            "t_eval_orig": float(g_t0),
                            "gap_start": float(g_start),
                            "gap_end": float(g_end),
                        },
                        "rewritten": {"t_seg_prime": float(t_seg_prime), "t_gap_prime": float(t_gap_prime)},
                    }
                    new_samples.append(ns_s)

                    ns_g = dict(g)
                    ns_g["t_eval_orig"] = float(g_t0)
                    ns_g["t_eval_rel_orig"] = float(g_t0)
                    ns_g["t_eval_rel"] = float(t_gap_prime)
                    ns_g["t_eval"] = float(offset + t_gap_prime)
                    ns_g["prompt"] = prompt_builder(t_gap_prime)

                    ns_g["state_switch_pair_type"] = "segment_to_gap"
                    ns_g["state_switch_role"] = "scan_gap"
                    ns_g["state_switch_param"] = {"raw": ss_raw_echo, "mode": ss_mode, "value": float(ss_value), "numeric_heuristic": bool(ss_used_heur)}
                    ns_g["state_switch_neighbor_dt"] = float(query_freq)
                    ns_g["state_switch_pair_id"] = pair_id

                    ns_g["state_switch_trace"] = {
                        "segment": {
                            "idx": _safe_int(s.get("idx", -1), -1),
                            "t_eval_orig": float(s_t0),
                            "segment_start": float(s_start),
                            "segment_end": float(s_end),
                        },
                        "gap": {
                            "idx": _safe_int(g.get("idx", -1), -1),
                            "t_eval_orig": float(g_t0),
                            "gap_start": float(g_start),
                            "gap_end": float(g_end),
                        },
                        "rewritten": {"t_seg_prime": float(t_seg_prime), "t_gap_prime": float(t_gap_prime)},
                    }
                    new_samples.append(ns_g)

            for i, ns in enumerate(new_samples):
                ns["idx"] = int(i)

            new_gt["helper"].update(
                {
                    "state_switch_query_freq": float(query_freq),
                    "state_switch_ss_pos_raw": str(ss_raw_echo),
                    "state_switch_ss_pos_parsed": {"mode": ss_mode, "value": float(ss_value), "numeric_heuristic": bool(ss_used_heur)},
                    "state_switch_neighbor_filter_kept": int(neighbor_kept),
                    "state_switch_neighbor_filter_dropped": int(neighbor_dropped),
                    "state_switch_pairs_total": int(len(pairs_sorted)),
                    "note_state_switch": "now_state_switch uses neighbor filtering (dt=query_freq), enumerates adjacent gap/segment pairs, then emits two probes per pair with rewritten t_eval.",
                }
            )

            new_gt["samples"] = new_samples
            new_gt["num_samples"] = int(len(new_samples))

            self._maybe_attach_memory_queryset(new_gt, task=task, gt_obj=gt, gt_path=src)
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(new_gt, f, ensure_ascii=False, indent=2)

            return dst

        # shared bits for retrieval/pred tasks
        pred_flavor = _normalize_pred_flavor(_env_get("PRED_FLAVOR", "open"))
        filter_rejected = _env_get("NOW_FILTER_REJECTED", "1") == "1"

        kept: List[Dict[str, Any]] = []
        for s in samples:
            if not isinstance(s, dict):
                continue
            if filter_rejected and bool(s.get("is_rejected", False)):
                continue
            if bool(s.get("is_invalid_annotation", False)):
                continue
            if "t_eval" not in s:
                continue
            kept.append(s)
        if not kept:
            raise ValueError(f"No usable samples found in GT: {src}")

        kept_sorted = sorted(
            kept,
            key=lambda x: (int(x.get("idx", 10**9)), _f(x.get("t_eval", 0.0), 0.0)),
        )

        params_in = gt.get("params", {})
        if not isinstance(params_in, dict):
            params_in = {}

        # --------------------------
        # sh_rtrv (unchanged)
        # --------------------------
        if task == "sh_rtrv":
            mcp_path: Optional[Path] = None
            mcp_by_idx: Dict[int, Dict[str, Any]] = {}
            if pred_flavor == "cand":
                split = str(vm.get("split", "")) or str(gt.get("split", "")) or "val"
                split = split.strip().lower() or "val"
                mcp_path = _resolve_mcp_path(src, split, task="sh_rtrv")
                mcp_obj = _load_json_or_one_jsonl(mcp_path)
                mcp_by_idx = _index_samples_by_idx(mcp_obj.get("samples", []))

            new_gt = copy.deepcopy(gt)
            new_gt["task"] = "sh_rtrv"
            new_gt["task_name"] = "sh_rtrv"

            params = new_gt.get("params", {})
            if not isinstance(params, dict):
                params = {}
            params["pred_flavor"] = pred_flavor
            params["time_mode"] = str(time_mode)
            params["time_offset_sec"] = float(offset)
            params["interval_len_sec"] = float(interval_len_sec)
            params["sh_rtrv_lookback_default_sec"] = float(_f(_env_get("SH_RTRV_LOOKBACK_SEC", "20"), 20.0))
            if mcp_path is not None:
                params["mcp_path"] = str(mcp_path)
            new_gt["params"] = params

            new_gt["helper"] = {
                "pred_flavor": pred_flavor,
                "time_mode": str(time_mode),
                "time_offset_sec": float(offset),
                "note": "sh_rtrv prompts generated here. Candidate mode reads shuffled MCP (no reshuffle).",
                "mcp_root": _env_get("MCP_ROOT", _default_mcp_root_for_task("sh_rtrv")),
            }

            new_samples = []
            used_times = set()

            for s in kept_sorted:
                t_rel = _f(s.get("t_eval", 0.0), 0.0)
                t_sched = float(offset + t_rel)

                if t_sched in used_times:
                    continue
                used_times.add(t_sched)

                lookback_sec = float(self._pick_lookback_sec_for_sh_rtrv(s))

                ns = dict(s)
                ns["t_eval_rel"] = float(t_rel)
                ns["t_eval"] = float(t_sched)

                ns["window_start_sec"] = max(0.0, float(t_sched - lookback_sec))
                ns["window_end_sec"] = float(t_sched)

                if pred_flavor == "open":
                    ns["prompt"] = self._build_prompt_past_open(t_rel, lookback_sec)
                else:
                    idx = _safe_int(ns.get("idx", -1), -1)
                    if idx < 0:
                        raise ValueError(f"GT sample missing idx: {src}")
                    mc = mcp_by_idx.get(idx)
                    if mc is None:
                        raise KeyError(f"MCP missing idx={idx} for {src.name}")
                    options = mc.get("options", None)
                    if not isinstance(options, list) or len(options) != 4:
                        raise ValueError(f"MCP idx={idx} options must be list len 4: {mcp_path}")
                    ns["prompt"] = self._build_prompt_past_cand(t_rel, lookback_sec, [str(x) for x in options])

                    ns["mcq"] = {
                        "mcp_file": str(mcp_path),
                        "answer": mc.get("answer", None),
                        "answer_idx": mc.get("answer_idx", None),
                        "option_sources": mc.get("option_sources", None),
                        "options": options,
                    }

                new_samples.append(ns)

            new_gt["samples"] = new_samples
            new_gt["num_samples"] = int(len(new_samples))

            self._maybe_attach_memory_queryset(new_gt, task=task, gt_obj=gt, gt_path=src)
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(new_gt, f, ensure_ascii=False, indent=2)

            return dst

        # --------------------------
        # ms_rtrv (unchanged below)
        # --------------------------
        if task == "ms_rtrv":
            mcp_path: Optional[Path] = None
            mcp_by_idx: Dict[int, Dict[str, Any]] = {}
            if pred_flavor == "cand":
                split = str(vm.get("split", "")) or str(gt.get("split", "")) or "val"
                split = split.strip().lower() or "val"
                mcp_path = _resolve_mcp_path(src, split, task="ms_rtrv")
                mcp_obj = _load_json_or_one_jsonl(mcp_path)
                mcp_by_idx = _index_samples_by_idx(mcp_obj.get("samples", []))

            new_gt = copy.deepcopy(gt)
            new_gt["task"] = "ms_rtrv"
            new_gt["task_name"] = "ms_rtrv"

            params = new_gt.get("params", {})
            if not isinstance(params, dict):
                params = {}
            params["pred_flavor"] = pred_flavor
            params["time_mode"] = str(time_mode)
            params["time_offset_sec"] = float(offset)
            params["interval_len_sec"] = float(interval_len_sec)
            if "lags_sec" in params_in and "lags_sec" not in params:
                params["lags_sec"] = params_in.get("lags_sec")
            if mcp_path is not None:
                params["mcp_path"] = str(mcp_path)
            new_gt["params"] = params

            new_gt["helper"] = {
                "pred_flavor": pred_flavor,
                "time_mode": str(time_mode),
                "time_offset_sec": float(offset),
                "note": "ms_rtrv prompts generated here. Window uses evidence segment (gt_segment_start/end) when available.",
                "mcp_root": _env_get("MCP_ROOT", _default_mcp_root_for_task("ms_rtrv")),
            }

            ds_task = "ms_rtrv"
            ds_every, ds_keep = _task_ds_params(ds_task)

            groups = _group_contiguous_by_anchor(kept_sorted)

            new_samples: List[Dict[str, Any]] = []
            used_times = set()

            for grp in groups:
                if not grp:
                    continue

                rep_t_rel = _f(grp[0].get("t_eval", 0.0), 0.0)
                rep_t_sched = float(offset + rep_t_rel)
                if rep_t_sched in used_times:
                    continue

                if not _ds_should_keep_unit(ds_task):
                    continue

                used_times.add(rep_t_sched)

                for s in grp:
                    t_rel = _f(s.get("t_eval", 0.0), 0.0)
                    t_sched = float(offset + t_rel)

                    lag_sec = float(_pick_lag_sec_for_ms_rtrv(s, params_in))

                    ns = dict(s)
                    ns["t_eval_rel"] = float(t_rel)
                    ns["t_eval"] = float(t_sched)

                    ws, we, t_target = _ms_window_from_evidence_segment(ns, t_eval_sched=t_sched, lag=lag_sec)
                    ns["t_target_sec"] = float(t_target)
                    ns["window_start_sec"] = float(ws)
                    ns["window_end_sec"] = float(we)

                    ns["lag_sec"] = float(lag_sec)
                    ns["lookback_sec"] = float(lag_sec)

                    if pred_flavor == "open":
                        ns["prompt"] = self._build_prompt_ms_open(t_rel, lag_sec, ns)
                    else:
                        idx = _safe_int(ns.get("idx", -1), -1)
                        if idx < 0:
                            raise ValueError(f"GT sample missing idx: {src}")
                        mc = mcp_by_idx.get(idx)
                        if mc is None:
                            raise KeyError(f"MCP missing idx={idx} for {src.name}")
                        options = mc.get("options", None)
                        if not isinstance(options, list) or len(options) != 4:
                            raise ValueError(f"MCP idx={idx} options must be list len 4: {mcp_path}")
                        ns["prompt"] = self._build_prompt_ms_cand(t_rel, lag_sec, [str(x) for x in options], ns)

                        ns["mcq"] = {
                            "mcp_file": str(mcp_path),
                            "answer": mc.get("answer", None),
                            "answer_idx": mc.get("answer_idx", None),
                            "option_sources": mc.get("option_sources", None),
                            "options": options,
                        }

                    new_samples.append(ns)

            new_gt["samples"] = new_samples
            new_gt["num_samples"] = int(len(new_samples))

            before, after, target, reached, already = _ds_update_cumulative(ds_task, len(new_samples))
            new_gt["helper"]["downsample"] = {
                "enabled": True,
                "task": ds_task,
                "unit": "anchor_group",
                "policy": "keep_first_k_in_every_n_groups (uniform, global stream across GT files)",
                "every": int(ds_every),
                "keep": int(ds_keep),
                "phase_after": int(_GLOBAL_DS_PHASE.get(ds_task, 0)),
            }
            new_gt["helper"]["global_limit"] = {
                "target_samples": int(target),
                "cumulative_before": int(before),
                "cumulative_after": int(after),
                "already_reached_before_this_file": bool(already),
                "stop_after_this_file": bool(reached),
            }
            new_gt["helper"]["stop_after_this_file"] = bool(reached)

            self._maybe_attach_memory_queryset(new_gt, task=task, gt_obj=gt, gt_path=src)
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(new_gt, f, ensure_ascii=False, indent=2)

            return dst

        # --------------------------
        # ms_pred (unchanged below)
        # --------------------------
        if task == "ms_pred":
            mcp_path: Optional[Path] = None
            mcp_by_idx: Dict[int, Dict[str, Any]] = {}
            if pred_flavor == "cand":
                split = str(vm.get("split", "")) or str(gt.get("split", "")) or "val"
                split = split.strip().lower() or "val"
                mcp_path = _resolve_mcp_path(src, split, task="ms_pred")
                mcp_obj = _load_json_or_one_jsonl(mcp_path)
                mcp_by_idx = _index_samples_by_idx(mcp_obj.get("samples", []))

            new_gt = copy.deepcopy(gt)
            new_gt["task"] = "ms_pred"
            new_gt["task_name"] = "ms_pred"

            params = new_gt.get("params", {})
            if not isinstance(params, dict):
                params = {}
            params["pred_flavor"] = pred_flavor
            params["time_mode"] = str(time_mode)
            params["time_offset_sec"] = float(offset)
            params["interval_len_sec"] = float(interval_len_sec)
            if "horizons_sec" in params_in and "horizons_sec" not in params:
                params["horizons_sec"] = params_in.get("horizons_sec")
            if "lags_sec" in params_in and "lags_sec" not in params:
                params["lags_sec"] = params_in.get("lags_sec")
            params["ms_pred_context_default_sec"] = float(_f(_env_get("MS_PRED_CONTEXT_SEC", "20"), 20.0))
            if mcp_path is not None:
                params["mcp_path"] = str(mcp_path)
            new_gt["params"] = params

            new_gt["helper"] = {
                "pred_flavor": pred_flavor,
                "time_mode": str(time_mode),
                "time_offset_sec": float(offset),
                "note": "ms_pred prompts generated here. Window is CONTEXT only: [t_eval-context, t_eval]. Candidate mode reads shuffled MCP (no reshuffle).",
                "mcp_root": _env_get("MCP_ROOT", _default_mcp_root_for_task("ms_pred")),
                "ms_pred_context_sec": float(_f(_env_get("MS_PRED_CONTEXT_SEC", "20"), 20.0)),
            }

            ds_task = "ms_pred"
            ds_every, ds_keep = _task_ds_params(ds_task)

            groups = _group_contiguous_by_anchor(kept_sorted)

            new_samples: List[Dict[str, Any]] = []
            used_times = set()

            for grp in groups:
                if not grp:
                    continue

                rep_t_rel = _f(grp[0].get("t_eval", 0.0), 0.0)
                rep_t_sched = float(offset + rep_t_rel)
                if rep_t_sched in used_times:
                    continue

                if not _ds_should_keep_unit(ds_task):
                    continue

                used_times.add(rep_t_sched)

                for s in grp:
                    t_rel = _f(s.get("t_eval", 0.0), 0.0)
                    t_sched = float(offset + t_rel)

                    horizon_sec = float(_pick_horizon_sec_for_ms_pred(s, params_in, t_eval_rel=float(t_rel)))

                    default_ctx = float(_f(_env_get("MS_PRED_CONTEXT_SEC", "20"), 20.0))
                    # NEW: control whether context depends on horizon
                    context_mode = os.environ.get("MS_PRED_CONTEXT_MODE", "fixed").strip().lower() or "fixed"
                    if context_mode in {"max_with_horizon", "max", "horizon_max"}:
                        context_sec = float(max(default_ctx, float(horizon_sec)))
                    else:
                        # fixed context: isolate horizon difficulty
                        context_sec = float(default_ctx)
                    ns = dict(s)

                    # (optional but strongly recommended) record for debugging
                    ns["context_sec"] = float(context_sec)
                    ns["context_mode"] = str(context_mode)

                    ns["t_eval_rel"] = float(t_rel)
                    ns["t_eval"] = float(t_sched)

                    ns["horizon_sec"] = float(horizon_sec)
                    ns["t_target_sec"] = float(t_sched + horizon_sec)

                    ns["window_end_sec"] = float(t_sched)
                    ns["window_start_sec"] = float(max(0.0, t_sched - context_sec))

                    ns["lookback_sec"] = float(context_sec)

                    if pred_flavor == "open":
                        ns["prompt"] = self._build_prompt_pred_open(t_rel, horizon_sec, ns)
                    else:
                        idx = _safe_int(ns.get("idx", -1), -1)
                        if idx < 0:
                            raise ValueError(f"GT sample missing idx: {src}")
                        mc = mcp_by_idx.get(idx)
                        if mc is None:
                            raise KeyError(f"MCP missing idx={idx} for {src.name}")
                        options = mc.get("options", None)
                        if not isinstance(options, list) or len(options) != 4:
                            raise ValueError(f"MCP idx={idx} options must be list len 4: {mcp_path}")
                        ns["prompt"] = self._build_prompt_pred_cand(t_rel, horizon_sec, [str(x) for x in options], ns)

                        ns["mcq"] = {
                            "mcp_file": str(mcp_path),
                            "answer": mc.get("answer", None),
                            "answer_idx": mc.get("answer_idx", None),
                            "option_sources": mc.get("option_sources", None),
                            "options": options,
                        }

                    new_samples.append(ns)

            new_gt["samples"] = new_samples
            new_gt["num_samples"] = int(len(new_samples))

            before, after, target, reached, already = _ds_update_cumulative(ds_task, len(new_samples))
            new_gt["helper"]["downsample"] = {
                "enabled": True,
                "task": ds_task,
                "unit": "anchor_group",
                "policy": "keep_first_k_in_every_n_groups (uniform, global stream across GT files)",
                "every": int(ds_every),
                "keep": int(ds_keep),
                "phase_after": int(_GLOBAL_DS_PHASE.get(ds_task, 0)),
            }
            new_gt["helper"]["global_limit"] = {
                "target_samples": int(target),
                "cumulative_before": int(before),
                "cumulative_after": int(after),
                "already_reached_before_this_file": bool(already),
                "stop_after_this_file": bool(reached),
            }
            new_gt["helper"]["stop_after_this_file"] = bool(reached)

            self._maybe_attach_memory_queryset(new_gt, task=task, gt_obj=gt, gt_path=src)
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(new_gt, f, ensure_ascii=False, indent=2)

            return dst

        # --------------------------
        # sh_pred (unchanged)
        # --------------------------
        if task == "sh_pred":
            mcp_path: Optional[Path] = None
            mcp_by_idx: Dict[int, Dict[str, Any]] = {}
            if pred_flavor == "cand":
                split = str(vm.get("split", "")) or str(gt.get("split", "")) or "val"
                split = split.strip().lower() or "val"
                mcp_path = _resolve_mcp_path(src, split, task="sh_pred")
                mcp_obj = _load_json_or_one_jsonl(mcp_path)
                mcp_by_idx = _index_samples_by_idx(mcp_obj.get("samples", []))

            _all_raw = _env_get("SH_PRED_ALL_STATES", "0").strip().lower()
            sh_pred_all_states = _all_raw not in {"", "0", "false", "no", "off"}
            enforce_nn_filter = (pred_flavor == "cand") and (not sh_pred_all_states)
            out_task_name = "sh_pred_full" if ((pred_flavor == "cand") and sh_pred_all_states) else "sh_pred"

            new_gt = copy.deepcopy(gt)
            new_gt["task"] = out_task_name
            new_gt["task_name"] = out_task_name

            params = new_gt.get("params", {})
            if not isinstance(params, dict):
                params = {}
            params["pred_flavor"] = pred_flavor
            params["time_mode"] = str(time_mode)
            params["time_offset_sec"] = float(offset)
            params["interval_len_sec"] = float(interval_len_sec)
            params["sh_pred_horizon_default_sec"] = float(_f(_env_get("SH_PRED_HORIZON_SEC", "8"), 8.0))
            params["sh_pred_context_default_sec"] = float(_f(_env_get("SH_PRED_CONTEXT_SEC", "20"), 20.0))
            if mcp_path is not None:
                params["mcp_path"] = str(mcp_path)
            params["sh_pred_all_states"] = bool(sh_pred_all_states)
            new_gt["params"] = params

            cand_only_state = "ALL" if not enforce_nn_filter else "NN"
            cand_filtered_out = 0
            cand_kept = 0

            new_gt["helper"] = {
                "pred_flavor": pred_flavor,
                "time_mode": str(time_mode),
                "time_offset_sec": float(offset),
                "note": "sh_pred prompts generated here. Window is CONTEXT only: [t_eval-context, t_eval]. Candidate mode reads shuffled MCP (no reshuffle).",
                "mcp_root": _env_get("MCP_ROOT", _default_mcp_root_for_task("sh_pred")),
                "cand_only_state": cand_only_state,
                "sh_pred_all_states":bool(sh_pred_all_states),
            }

            new_samples: List[Dict[str, Any]] = []
            used_times = set()

            for s in kept_sorted:
                if enforce_nn_filter:
                    st = str(s.get("state", "")).upper().strip()
                    if st != "NN":
                        cand_filtered_out += 1
                        continue

                t_rel = _f(s.get("t_eval", 0.0), 0.0)
                t_sched = float(offset + t_rel)

                if t_sched in used_times:
                    continue

                horizon_sec = float(_pick_horizon_sec_for_sh_pred(s))

                default_ctx = float(_f(_env_get("SH_PRED_CONTEXT_SEC", "20"), 20.0))
                context_sec = float(max(default_ctx, float(horizon_sec)))

                ns = dict(s)
                ns["t_eval_rel"] = float(t_rel)
                ns["t_eval"] = float(t_sched)

                ns["horizon_sec"] = float(horizon_sec)
                ns["t_target_sec"] = float(t_sched + horizon_sec)

                ns["window_end_sec"] = float(t_sched)
                ns["window_start_sec"] = float(max(0.0, t_sched - context_sec))

                ns["lookback_sec"] = float(context_sec)

                if pred_flavor == "open":
                    ns["prompt"] = self._build_prompt_sh_pred_open(t_rel, horizon_sec)
                else:
                    idx = _safe_int(ns.get("idx", -1), -1)
                    if idx < 0:
                        raise ValueError(f"GT sample missing idx: {src}")
                    mc = mcp_by_idx.get(idx)
                    if mc is None:
                        raise KeyError(f"MCP missing idx={idx} for {src.name}")
                    options = mc.get("options", None)
                    if not isinstance(options, list) or len(options) != 4:
                        raise ValueError(f"MCP idx={idx} options must be list len 4: {mcp_path}")
                    ns["prompt"] = self._build_prompt_sh_pred_cand(t_rel, horizon_sec, [str(x) for x in options])

                    ns["mcq"] = {
                        "mcp_file": str(mcp_path),
                        "answer": mc.get("answer", None),
                        "answer_idx": mc.get("answer_idx", None),
                        "option_sources": mc.get("option_sources", None),
                        "options": options,
                    }

                new_samples.append(ns)
                used_times.add(t_sched)
                if pred_flavor == "cand":
                    cand_kept += 1

            if pred_flavor == "cand":
                new_gt["helper"]["cand_filtered_out_count"] = int(cand_filtered_out)
                new_gt["helper"]["cand_kept_count"] = int(cand_kept)

            new_gt["samples"] = new_samples
            new_gt["num_samples"] = int(len(new_samples))

            self._maybe_attach_memory_queryset(new_gt, task=task, gt_obj=gt, gt_path=src)
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(new_gt, f, ensure_ascii=False, indent=2)

            return dst

        raise ValueError(f"Unsupported task for this helper: {task}")


def create_helper():
    return _NowHelper()
