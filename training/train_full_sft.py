#!/usr/bin/env python3
"""Public EgoSAT full SFT entry point.

This wrapper exposes clean recipes for two independent SFT tasks
(`mixed5_cand`, `mixed7_stateheavy`) and two model variants (`timechat`,
`roi_timechat`). It validates public inputs, supports config+CLI overrides,
and delegates the heavy model work to cleaned legacy reference scripts.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = Path(__file__).resolve().parent
LEGACY_DIR = TRAINING_DIR / "legacy"

MODEL_VARIANTS = {"timechat", "roi_timechat"}
SFT_TASKS = {"mixed5_cand", "mixed7_stateheavy"}

DEFAULTS: Dict[str, Any] = {
    "training": {
        "epochs": 1,
        "batch_size": 1,
        "grad_accum": 8,
        "learning_rate": 2.0e-4,
        "weight_decay": 0.0,
        "seed": 1234,
        "max_frames": 16,
        "save_every_steps": 500,
        "log_every": 20,
        "max_steps": 0,
    },
    "lora": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    },
    "extra_trainable_patterns": [
        "mm_projector",
        "projector",
        "merger",
        "multi_modal_projector",
        "visual.merger",
    ],
}


class ConfigError(RuntimeError):
    """Raised for invalid public configuration."""


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(x.strip()) for x in inner.split(",")]
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", value):
            return float(value)
    except Exception:
        pass
    return value


def strip_yaml_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            if idx == 0 or line[idx - 1].isspace():
                return line[:idx].rstrip()
    return line.rstrip()


def simple_yaml_load(text: str) -> Dict[str, Any]:
    """Parse the small YAML subset used by training/configs/*.yaml.

    PyYAML is used when available. This fallback supports nested mappings and
    lists of scalar values, which keeps --config usable without adding a hard
    dependency just for dry-runs.
    """

    raw_lines: List[Tuple[int, str]] = []
    for raw in text.splitlines():
        line = strip_yaml_comment(raw)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        raw_lines.append((indent, line.strip()))

    def parse_block(index: int, indent: int) -> Tuple[Any, int]:
        result: Any = None
        while index < len(raw_lines):
            current_indent, content = raw_lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ConfigError(f"Invalid YAML indentation near: {content}")

            if content.startswith("- "):
                if result is None:
                    result = []
                if not isinstance(result, list):
                    raise ConfigError(f"Cannot mix YAML list and mapping near: {content}")
                item = content[2:].strip()
                if item:
                    result.append(parse_scalar(item))
                    index += 1
                else:
                    child, index = parse_block(index + 1, indent + 2)
                    result.append(child)
                continue

            if result is None:
                result = {}
            if not isinstance(result, dict):
                raise ConfigError(f"Cannot mix YAML mapping and list near: {content}")
            key, sep, value = content.partition(":")
            if not sep:
                raise ConfigError(f"Expected ':' in YAML line: {content}")
            key = key.strip()
            value = value.strip()
            if value:
                result[key] = parse_scalar(value)
                index += 1
            else:
                child, index = parse_block(index + 1, indent + 2)
                result[key] = child

        return ({} if result is None else result), index

    parsed, final_index = parse_block(0, 0)
    if final_index != len(raw_lines):
        raise ConfigError("Could not parse complete YAML file.")
    if not isinstance(parsed, dict):
        raise ConfigError("Top-level YAML config must be a mapping.")
    return parsed


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path).expanduser()
    if not cfg_path.exists():
        raise ConfigError(f"--config does not exist: {cfg_path}")
    text = cfg_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except Exception:
        return simple_yaml_load(text)
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ConfigError("Top-level YAML config must be a mapping.")
    return data


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run EgoSAT full SFT for one public recipe. "
            "mixed5_cand and mixed7_stateheavy are independent tasks, not stages."
        )
    )
    parser.add_argument("--config", help="Optional YAML config. CLI values override config values.")
    parser.add_argument("--model-variant", choices=sorted(MODEL_VARIANTS))
    parser.add_argument("--sft-task", choices=sorted(SFT_TASKS))
    parser.add_argument("--manifest")
    parser.add_argument("--ego4d-root")
    parser.add_argument("--roi-cache-root")
    parser.add_argument("--timechat-repo-root")
    parser.add_argument("--base-model")
    parser.add_argument("--output-dir")
    parser.add_argument("--lora-r", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--save-every-steps", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--max-steps", type=int, help="Optional cap for debugging; 0 means no cap.")
    parser.add_argument("--resume-from", help="Optional LoRA checkpoint directory to continue from.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and inspect manifest without training.")
    return parser


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = deep_merge(DEFAULTS, config)
    top_level = {
        "model_variant": args.model_variant,
        "sft_task": args.sft_task,
        "manifest": args.manifest,
        "ego4d_root": args.ego4d_root,
        "roi_cache_root": args.roi_cache_root,
        "timechat_repo_root": args.timechat_repo_root,
        "base_model": args.base_model,
        "output_dir": args.output_dir,
        "resume_from": args.resume_from,
    }
    for key, value in top_level.items():
        if value is not None:
            cfg[key] = value

    training_map = {
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_frames": args.max_frames,
        "seed": args.seed,
        "save_every_steps": args.save_every_steps,
        "log_every": args.log_every,
        "weight_decay": args.weight_decay,
        "max_steps": args.max_steps,
    }
    cfg.setdefault("training", {})
    for key, value in training_map.items():
        if value is not None:
            cfg["training"][key] = value

    cfg.setdefault("lora", {})
    if args.lora_r is not None:
        cfg["lora"]["r"] = args.lora_r
    if args.lora_alpha is not None:
        cfg["lora"]["alpha"] = args.lora_alpha
    if args.lora_dropout is not None:
        cfg["lora"]["dropout"] = args.lora_dropout

    cfg["dry_run"] = bool(args.dry_run)
    return cfg


def require_value(plan: Dict[str, Any], key: str) -> str:
    value = plan.get(key)
    if value is None or str(value).strip() == "":
        raise ConfigError(f"Missing required argument/config value: {key}")
    return str(value)


def validate_plan(plan: Dict[str, Any], *, dry_run: bool) -> None:
    variant = require_value(plan, "model_variant")
    task = require_value(plan, "sft_task")
    if variant not in MODEL_VARIANTS:
        raise ConfigError(f"Unsupported model_variant={variant!r}; expected one of {sorted(MODEL_VARIANTS)}")
    if task not in SFT_TASKS:
        raise ConfigError(f"Unsupported sft_task={task!r}; expected one of {sorted(SFT_TASKS)}")

    manifest = Path(require_value(plan, "manifest")).expanduser()
    if not manifest.exists():
        raise ConfigError(f"Manifest does not exist: {manifest}")

    require_value(plan, "ego4d_root")
    require_value(plan, "timechat_repo_root")
    require_value(plan, "base_model")
    require_value(plan, "output_dir")

    if variant == "roi_timechat":
        require_value(plan, "roi_cache_root")

    for section, keys in {
        "training": ["epochs", "batch_size", "grad_accum", "learning_rate", "seed", "max_frames"],
        "lora": ["r", "alpha", "dropout"],
    }.items():
        obj = plan.get(section)
        if not isinstance(obj, dict):
            raise ConfigError(f"Config section {section!r} must be a mapping.")
        for key in keys:
            if key not in obj:
                raise ConfigError(f"Missing config value: {section}.{key}")

    if dry_run:
        return

    timechat_root = Path(str(plan["timechat_repo_root"])).expanduser()
    if not timechat_root.exists():
        raise ConfigError(f"timechat-repo-root does not exist: {timechat_root}")
    ego4d_root = Path(str(plan["ego4d_root"])).expanduser()
    if not ego4d_root.exists():
        raise ConfigError(f"ego4d-root does not exist: {ego4d_root}")
    if variant == "roi_timechat":
        roi_root = Path(str(plan["roi_cache_root"])).expanduser()
        if not roi_root.exists():
            raise ConfigError(f"roi-cache-root does not exist: {roi_root}")


def check_training_dependencies(variant: str) -> None:
    required = ["torch", "transformers", "peft", "decord", "PIL"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise ConfigError(
            "Missing required training dependencies: "
            + ", ".join(missing)
            + ". Install the TimeChat/EgoSAT training environment before running real SFT."
        )
    if variant == "roi_timechat" and importlib.util.find_spec("peft") is None:
        raise ConfigError("roi_timechat training requires peft.")


def read_jsonl_samples(path: Path, limit: int = 2) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ConfigError(f"JSONL row must be an object at {path}:{line_no}")
            samples.append(obj)
            if len(samples) >= limit:
                break
    if not samples:
        raise ConfigError(f"Manifest has no usable JSON rows: {path}")
    return samples


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ConfigError(f"JSONL row must be an object at {path}:{line_no}")
            yield obj


def safe_stem(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "sample"


def interval_clip_path_for_record(rec: Dict[str, Any], output_dir: Path) -> str:
    vm = rec.get("video_metadata") if isinstance(rec.get("video_metadata"), dict) else {}
    task = str(rec.get("task", "task"))
    video_uid = str(rec.get("video_uid", "video"))
    clip_id = str(vm.get("clip_id", "clip"))
    clip_uid = str(vm.get("clip_uid", "uid"))
    start = float(vm.get("interval_start_sec", 0.0) or 0.0)
    end = float(vm.get("interval_end_sec", start) or start)
    name = safe_stem(f"{video_uid}__{clip_id}__{clip_uid}_{start:.3f}-{end:.3f}") + ".mp4"
    return str(output_dir / "_runtime" / "video_cache" / safe_stem(task) / name)


def roi_cache_path_for_record(rec: Dict[str, Any], roi_cache_root: Path) -> str:
    vm = rec.get("video_metadata") if isinstance(rec.get("video_metadata"), dict) else {}
    video_uid = str(rec.get("video_uid", "")).strip()
    clip_id = str(vm.get("clip_id", "")).strip()
    clip_uid = str(vm.get("clip_uid", "")).strip()

    candidates: List[Path] = []
    if video_uid and clip_id and clip_uid:
        base = f"{video_uid}__{clip_id}__{clip_uid}"
        candidates.extend(
            [
                roi_cache_root / f"{base}.roi_cache_merged_fps1.jsonl",
                roi_cache_root / f"{base}.roi_cache_merged_fps1.json",
                roi_cache_root / f"{base}.roi_cache.jsonl",
            ]
        )
    if video_uid:
        candidates.extend(
            [
                roi_cache_root / f"{video_uid}.roi_cache_merged_fps1.jsonl",
                roi_cache_root / f"{video_uid}.roi_cache.jsonl",
            ]
        )
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return str(candidates[0] if candidates else roi_cache_root / "UNKNOWN.roi_cache_merged_fps1.jsonl")


def prepare_runtime_manifest(plan: Dict[str, Any]) -> Path:
    manifest = Path(str(plan["manifest"])).expanduser()
    output_dir = Path(str(plan["output_dir"])).expanduser()
    runtime_dir = output_dir / "_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_manifest = runtime_dir / f"{plan['model_variant']}__{plan['sft_task']}__runtime_manifest.jsonl"
    roi_root = Path(str(plan["roi_cache_root"])).expanduser() if plan["model_variant"] == "roi_timechat" else None

    count = 0
    with runtime_manifest.open("w", encoding="utf-8") as fout:
        for rec in iter_jsonl(manifest):
            rec = copy.deepcopy(rec)
            if "sample" not in rec or "target_text" not in rec:
                raise ConfigError("Every manifest row must include sample and target_text for SFT.")
            if "interval_clip_path" not in rec:
                rec["interval_clip_path"] = interval_clip_path_for_record(rec, output_dir)
            if roi_root is not None and "roi_cache_path" not in rec:
                rec["roi_cache_path"] = roi_cache_path_for_record(rec, roi_root)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    if count == 0:
        raise ConfigError("Runtime manifest would be empty.")
    return runtime_manifest


def select_legacy_script(plan: Dict[str, Any]) -> Path:
    variant = str(plan["model_variant"])
    resume_from = str(plan.get("resume_from") or "").strip()
    if resume_from:
        return LEGACY_DIR / ("lora_projector_sft_roi_new.py" if variant == "roi_timechat" else "lora_projector_sft_new.py")
    return LEGACY_DIR / ("lora_projector_sft_roi.py" if variant == "roi_timechat" else "lora_projector_sft.py")


def env_list(values: Iterable[Any]) -> str:
    return ",".join(str(v).strip() for v in values if str(v).strip())


def build_legacy_env(plan: Dict[str, Any], runtime_manifest: Path) -> Dict[str, str]:
    training = plan["training"]
    lora = plan["lora"]
    output_dir = Path(str(plan["output_dir"])).expanduser()
    env = os.environ.copy()
    env.update(
        {
            "MANIFEST_PATH": str(runtime_manifest),
            "OUT_DIR": str(output_dir),
            "EGO4D_ROOT": str(Path(str(plan["ego4d_root"])).expanduser()),
            "VIDEO_ROOT": str(Path(str(plan["ego4d_root"])).expanduser()),
            "TIMECHAT_REPO_ROOT": str(Path(str(plan["timechat_repo_root"])).expanduser()),
            "TIMECHAT_HF_MODEL_ID": str(plan["base_model"]),
            "MODEL_ID": str(plan["base_model"]),
            "FPS1_CACHE_DIR": str(output_dir / "_runtime" / "fps1_cache"),
            "SEED": str(training["seed"]),
            "EPOCHS": str(training["epochs"]),
            "BATCH_SIZE": str(training["batch_size"]),
            "GRAD_ACCUM": str(training["grad_accum"]),
            "LR": str(training["learning_rate"]),
            "WEIGHT_DECAY": str(training.get("weight_decay", 0.0)),
            "MAX_FRAMES": str(training["max_frames"]),
            "SAVE_EVERY_STEPS": str(training.get("save_every_steps", 500)),
            "LOG_EVERY": str(training.get("log_every", 20)),
            "MAX_STEPS": str(training.get("max_steps", 0)),
            "LORA_R": str(lora["r"]),
            "LORA_ALPHA": str(lora["alpha"]),
            "LORA_DROPOUT": str(lora["dropout"]),
            "LORA_TARGETS": env_list(lora.get("target_modules", DEFAULTS["lora"]["target_modules"])),
            "UNFREEZE_PATTERNS": env_list(plan.get("extra_trainable_patterns", DEFAULTS["extra_trainable_patterns"])),
            "USE_DTD_IN_TRAIN": "1",
            "TIMECHAT_USE_ROI_MODEL": "1" if plan["model_variant"] == "roi_timechat" else "0",
            "TIMECHAT_DROP_METHOD": "roi_feature" if plan["model_variant"] == "roi_timechat" else "feature",
        }
    )
    if plan["model_variant"] == "roi_timechat":
        env["ROI_CACHE_ROOT"] = str(Path(str(plan["roi_cache_root"])).expanduser())
    resume_from = str(plan.get("resume_from") or "").strip()
    if resume_from:
        env["RESUME_DIR"] = str(Path(resume_from).expanduser())
        env["PREV_TRAIN_OUT_DIR"] = str(Path(resume_from).expanduser())
    return env


def dry_run(plan: Dict[str, Any]) -> None:
    manifest = Path(str(plan["manifest"])).expanduser()
    samples = read_jsonl_samples(manifest, limit=2)
    warnings: List[str] = []
    for key in ["timechat_repo_root", "ego4d_root"]:
        path = Path(str(plan[key])).expanduser()
        if not path.exists():
            warnings.append(f"{key} does not exist in dry-run environment: {path}")
    if plan["model_variant"] == "roi_timechat":
        roi_path = Path(str(plan["roi_cache_root"])).expanduser()
        if not roi_path.exists():
            warnings.append(f"roi_cache_root does not exist in dry-run environment: {roi_path}")

    summary = {
        "model_variant": plan["model_variant"],
        "sft_task": plan["sft_task"],
        "manifest": str(manifest),
        "ego4d_root": str(plan["ego4d_root"]),
        "roi_cache_root": plan.get("roi_cache_root"),
        "timechat_repo_root": str(plan["timechat_repo_root"]),
        "base_model": str(plan["base_model"]),
        "output_dir": str(plan["output_dir"]),
        "resume_from": plan.get("resume_from"),
        "training": plan["training"],
        "lora": plan["lora"],
        "extra_trainable_patterns": plan.get("extra_trainable_patterns", []),
        "selected_legacy_script": str(select_legacy_script(plan).relative_to(ROOT)),
        "notes": [
            "Dry-run does not initialize TimeChat, load weights, cut video, or touch GPU.",
            "mixed5_cand and mixed7_stateheavy are independent public SFT tasks.",
        ],
        "warnings": warnings,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nManifest sample preview:")
    for idx, sample in enumerate(samples, start=1):
        row = {
            "row": idx,
            "task": sample.get("task"),
            "video_uid": sample.get("video_uid"),
            "target_text": sample.get("target_text"),
            "prompt_prefix": str((sample.get("sample") or {}).get("prompt", ""))[:160],
        }
        print(json.dumps(row, ensure_ascii=False, indent=2))


def run_training(plan: Dict[str, Any]) -> int:
    check_training_dependencies(str(plan["model_variant"]))
    runtime_manifest = prepare_runtime_manifest(plan)
    legacy_script = select_legacy_script(plan)
    if not legacy_script.exists():
        raise ConfigError(f"Legacy delegate script is missing: {legacy_script}")
    env = build_legacy_env(plan, runtime_manifest)
    cmd = [sys.executable, str(legacy_script)]
    print(f"[EgoSAT SFT] runtime manifest: {runtime_manifest}")
    print(f"[EgoSAT SFT] delegate: {legacy_script.relative_to(ROOT)}")
    print("[EgoSAT SFT] starting real training; this will load the TimeChat model.")
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        plan = apply_cli_overrides(cfg, args)
        validate_plan(plan, dry_run=bool(args.dry_run))
        if args.dry_run:
            dry_run(plan)
            return 0
        return run_training(plan)
    except ConfigError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
