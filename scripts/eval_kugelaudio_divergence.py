#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
"""Plan or execute a reproducible canonical-vs-ggml KugelAudio evaluation run.

The harness is intentionally config-driven so the exact input/settings tuple can
be checked in, reviewed, and replayed later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Path to evaluation JSON config")
    p.add_argument(
        "--execute",
        choices=["none", "canonical", "ggml", "both"],
        default="none",
        help="Actually run canonical and/or ggml commands (default: none / plan only)",
    )
    p.add_argument(
        "--allow-missing-artifacts",
        action="store_true",
        help="Permit plan generation even if referenced model/binary/artifact paths do not exist",
    )
    p.add_argument(
        "--write-plan",
        help="Optional path to write normalized plan JSON. Defaults to stdout only unless executing.",
    )
    return p.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("config root must be a JSON object")
    return data


def require(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"missing required config key: {key}")
    return data[key]


def resolve_path(base: Path, value: str | None) -> str | None:
    if value is None:
        return None
    p = Path(value)
    if not p.is_absolute():
        p = (base / p).resolve()
    return str(p)


def validate_exists(label: str, path_value: str | None, allow_missing: bool) -> None:
    if path_value is None:
        return
    if allow_missing:
        return
    if not Path(path_value).exists():
        raise FileNotFoundError(f"{label} does not exist: {path_value}")


def normalize_config(cfg: dict[str, Any], config_path: Path) -> dict[str, Any]:
    base = config_path.parent.resolve()
    out_dir = resolve_path(base, str(require(cfg, "output_dir")))
    norm = {
        "canonical_repo": resolve_path(base, str(require(cfg, "canonical_repo"))),
        "canonical_model": str(require(cfg, "canonical_model")),
        "ggml_cli": resolve_path(base, str(require(cfg, "ggml_cli"))),
        "ggml_model": resolve_path(base, str(require(cfg, "ggml_model"))),
        "ggml_tokenizer": resolve_path(base, str(require(cfg, "ggml_tokenizer"))),
        "reference_audio": resolve_path(base, str(require(cfg, "reference_audio"))),
        "text": str(require(cfg, "text")),
        "output_dir": out_dir,
        "seed": int(cfg.get("seed", 12345)),
        "cfg_scale": float(cfg.get("cfg_scale", 1.0)),
        "steps": int(cfg.get("steps", 8)),
        "max_frames": int(cfg.get("max_frames", 32)),
        "max_new_tokens": int(cfg.get("max_new_tokens", 4096)),
    }
    if not norm["text"].strip():
        raise ValueError("text must not be empty")
    if norm["steps"] <= 0:
        raise ValueError("steps must be > 0")
    if norm["max_frames"] <= 0:
        raise ValueError("max_frames must be > 0")
    if norm["max_new_tokens"] <= 0:
        raise ValueError("max_new_tokens must be > 0")
    return norm


def maybe_sha256(path_value: str) -> str | None:
    p = Path(path_value)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_canonical_inline_code(plan: dict[str, Any]) -> str:
    # Keep the inline script compact but explicit so the generated command is
    # self-contained and replayable from the saved plan.
    text = json.dumps(plan["shared_run"]["text"])
    ref = json.dumps(plan["shared_run"]["reference_audio"]["path"])
    model = json.dumps(plan["canonical"]["model"])
    out = json.dumps(plan["canonical"]["output_wav"])
    cfg = repr(plan["shared_run"]["generation"]["cfg_scale"])
    max_new = repr(plan["shared_run"]["generation"]["max_new_tokens"])
    seed = repr(plan["shared_run"]["generation"]["seed"])
    return (
        "import torch; "
        "from kugelaudio_open.utils.generation import load_model_and_processor, generate_speech; "
        "model, processor = load_model_and_processor(" + model + ", device='cpu', use_flash_attention=False); "
        "torch.manual_seed(" + seed + "); "
        "audio = generate_speech(model=model, processor=processor, text=" + text + ", voice_prompt=" + ref + ", cfg_scale=" + cfg + ", max_new_tokens=" + max_new + ", device='cpu', apply_watermark=False); "
        "processor.save_audio(audio, " + out + ")"
    )


def build_plan(cfg: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(cfg["output_dir"])
    canonical_out = str((out_dir / "canonical.wav").resolve())
    ggml_out = str((out_dir / "ggml.wav").resolve())
    canonical_log = str((out_dir / "canonical.log").resolve())
    ggml_log = str((out_dir / "ggml.log").resolve())

    plan = {
        "shared_run": {
            "text": cfg["text"],
            "reference_audio": {
                "path": cfg["reference_audio"],
                "sha256": maybe_sha256(cfg["reference_audio"]),
            },
            "generation": {
                "seed": cfg["seed"],
                "cfg_scale": cfg["cfg_scale"],
                "steps": cfg["steps"],
                "max_frames": cfg["max_frames"],
                "max_new_tokens": cfg["max_new_tokens"],
            },
        },
        "canonical": {
            "repo": cfg["canonical_repo"],
            "model": cfg["canonical_model"],
            "output_wav": canonical_out,
            "log_path": canonical_log,
        },
        "ggml": {
            "cli": cfg["ggml_cli"],
            "model": cfg["ggml_model"],
            "tokenizer": cfg["ggml_tokenizer"],
            "output_wav": ggml_out,
            "log_path": ggml_log,
        },
    }
    plan["canonical"]["command"] = [
        "uv", "run", "python", "-c", build_canonical_inline_code(plan),
    ]
    plan["ggml"]["command"] = [
        cfg["ggml_cli"],
        "tts",
        "--model", cfg["ggml_model"],
        "--tokenizer", cfg["ggml_tokenizer"],
        "--ref-audio", plan["shared_run"]["reference_audio"]["path"],
        "--text", plan["shared_run"]["text"],
        "--out", ggml_out,
        "--max-frames", str(plan["shared_run"]["generation"]["max_frames"]),
        "--steps", str(plan["shared_run"]["generation"]["steps"]),
        "--cfg", str(plan["shared_run"]["generation"]["cfg_scale"]),
        "--seed", str(plan["shared_run"]["generation"]["seed"]),
    ]
    return plan


def run_one(command: list[str], cwd: str, log_path: str, extra_env: dict[str, str] | None = None) -> int:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(command, cwd=cwd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = normalize_config(load_json(config_path), config_path)

    validate_exists("canonical_repo", cfg["canonical_repo"], args.allow_missing_artifacts)
    validate_exists("ggml_cli", cfg["ggml_cli"], args.allow_missing_artifacts)
    validate_exists("ggml_model", cfg["ggml_model"], args.allow_missing_artifacts)
    validate_exists("ggml_tokenizer", cfg["ggml_tokenizer"], args.allow_missing_artifacts)
    validate_exists("reference_audio", cfg["reference_audio"], args.allow_missing_artifacts)

    plan = build_plan(cfg)
    serialized = json.dumps(plan, indent=2, sort_keys=True)

    if args.write_plan:
        plan_path = Path(args.write_plan)
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(serialized + "\n", encoding="utf-8")

    if args.execute == "none":
        sys.stdout.write(serialized + "\n")
        return 0

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plan.json").write_text(serialized + "\n", encoding="utf-8")

    failures: list[str] = []
    if args.execute in {"canonical", "both"}:
        rc = run_one(plan["canonical"]["command"], plan["canonical"]["repo"], plan["canonical"]["log_path"])
        if rc != 0:
            failures.append(f"canonical rc={rc}")
    if args.execute in {"ggml", "both"}:
        rc = run_one(plan["ggml"]["command"], str(Path(cfg["ggml_cli"]).resolve().parent.parent.parent), plan["ggml"]["log_path"], {"VIBEVOICE_BACKEND": "cpu"})
        if rc != 0:
            failures.append(f"ggml rc={rc}")

    if failures:
        sys.stderr.write("eval harness failures: " + ", ".join(failures) + "\n")
        return 1
    sys.stdout.write(serialized + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
