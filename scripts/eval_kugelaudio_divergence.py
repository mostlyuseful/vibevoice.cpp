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
    p.add_argument(
        "--write-result-template",
        help="Optional path to write a normalized results-template JSON even in plan mode.",
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
        "ggml_model_q8_0": resolve_path(base, str(cfg.get("ggml_model_q8_0", ""))),
        "asr_model": resolve_path(base, str(cfg.get("asr_model", ""))),
        "asr_tokenizer": resolve_path(base, str(cfg.get("asr_tokenizer", ""))),
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
    norm["voice_cloned_sample"] = cfg.get("voice_cloned_sample", {})
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


def build_asr_command(cfg: dict[str, Any], audio_path: str) -> list[str]:
    return [
        cfg["ggml_cli"],
        "asr",
        "--model", str(cfg.get("asr_model", "")),
        "--tokenizer", str(cfg.get("asr_tokenizer", "")),
        "--audio", audio_path,
    ]


def build_plan(cfg: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(cfg["output_dir"])
    canonical_out = str((out_dir / "canonical.wav").resolve())
    ggml_out = str((out_dir / "ggml.wav").resolve())
    canonical_log = str((out_dir / "canonical.log").resolve())
    ggml_log = str((out_dir / "ggml.log").resolve())
    canonical_asr_log = str((out_dir / "canonical_asr.log").resolve())
    ggml_asr_log = str((out_dir / "ggml_asr.log").resolve())

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
            "asr_command": build_asr_command(cfg, canonical_out),
            "asr_log_path": canonical_asr_log,
        },
        "ggml": {
            "cli": cfg["ggml_cli"],
            "model": cfg["ggml_model"],
            "tokenizer": cfg["ggml_tokenizer"],
            "output_wav": ggml_out,
            "log_path": ggml_log,
            "asr_command": build_asr_command(cfg, ggml_out),
            "asr_log_path": ggml_asr_log,
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
    plan["voice_cloned_sample"] = {
        "name": cfg.get("voice_cloned_sample", {}).get("name", ""),
        "description": cfg.get("voice_cloned_sample", {}).get("description", ""),
        "ground_truth": canonical_out,
    }
    return plan


def build_results(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "shared_run": plan["shared_run"],
        "canonical": {
            "command": plan["canonical"]["command"],
            "log_path": plan["canonical"]["log_path"],
            "output_wav": plan["canonical"]["output_wav"],
            "asr_command": plan["canonical"]["asr_command"],
            "asr_log_path": plan["canonical"]["asr_log_path"],
            "status": "planned",
            "return_code": None,
            "output_sha256": None,
            "asr_return_code": None,
            "asr_transcript": None,
            "recall": None,
        },
        "ggml": {
            "command": plan["ggml"]["command"],
            "log_path": plan["ggml"]["log_path"],
            "output_wav": plan["ggml"]["output_wav"],
            "asr_command": plan["ggml"]["asr_command"],
            "asr_log_path": plan["ggml"]["asr_log_path"],
            "status": "planned",
            "return_code": None,
            "output_sha256": None,
            "asr_return_code": None,
            "asr_transcript": None,
            "recall": None,
        },
        "threshold_check": None,
    }


def write_json(path_value: str | Path, data: dict[str, Any]) -> None:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def extract_content(raw: str) -> str:
    """Extract the Content field(s) from the ASR JSON-like output."""
    key = '"Content":"'
    parts: list[str] = []
    pos = 0
    while True:
        idx = raw.find(key, pos)
        if idx == -1:
            break
        idx += len(key)
        end = idx
        while end < len(raw) and raw[end] != '"':
            end += 2 if (raw[end] == '\\' and end + 1 < len(raw)) else 1
        parts.append(raw[idx:end])
        pos = end
    return " ".join(parts)


def word_set(s: str) -> set[str]:
    import re
    return set(re.sub(r"[^A-Za-z0-9]+", " ", s).lower().split())


def compute_recall(source_text: str, transcript: str) -> float:
    src = word_set(source_text)
    out = word_set(extract_content(transcript))
    if not src:
        return 0.0
    hits = len(src & out)
    return hits / len(src)


def run_one(command: list[str], cwd: str, log_path: str, extra_env: dict[str, str] | None = None) -> tuple[int, str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(command, cwd=cwd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    return int(proc.returncode), Path(log_path).read_text(encoding="utf-8") if Path(log_path).exists() else ""


def check_threshold(results: dict[str, Any]) -> tuple[bool, str]:
    """Return (pass, message) for the v1 acceptance threshold."""
    canonical = results.get("canonical", {})
    ggml = results.get("ggml", {})
    c_recall = canonical.get("recall")
    g_recall = ggml.get("recall")
    if c_recall is None or g_recall is None:
        return False, "cannot check threshold: missing recall values"
    if c_recall == 0.0:
        return False, f"cannot check threshold: canonical recall is zero (ggml={g_recall:.4f})"
    ratio = g_recall / c_recall if c_recall > 0 else 0.0
    floor_ok = g_recall >= 0.80
    ratio_ok = ratio >= 0.95
    if floor_ok and ratio_ok:
        return True, f"PASS  ggml recall={g_recall:.4f}  canonical={c_recall:.4f}  ratio={ratio:.2%}  floor=0.80"
    reasons: list[str] = []
    if not ratio_ok:
        reasons.append(f"ratio {ratio:.2%} < 95% of canonical")
    if not floor_ok:
        reasons.append(f"recall {g_recall:.4f} < 0.80 floor")
    return False, "FAIL  " + "; ".join(reasons) + f"  (ggml={g_recall:.4f} canonical={c_recall:.4f})"


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = normalize_config(load_json(config_path), config_path)

    validate_exists("canonical_repo", cfg["canonical_repo"], args.allow_missing_artifacts)
    validate_exists("ggml_cli", cfg["ggml_cli"], args.allow_missing_artifacts)
    validate_exists("ggml_model", cfg["ggml_model"], args.allow_missing_artifacts)
    validate_exists("ggml_tokenizer", cfg["ggml_tokenizer"], args.allow_missing_artifacts)
    validate_exists("asr_model", cfg.get("asr_model"), args.allow_missing_artifacts)
    validate_exists("asr_tokenizer", cfg.get("asr_tokenizer"), args.allow_missing_artifacts)
    validate_exists("reference_audio", cfg["reference_audio"], args.allow_missing_artifacts)

    plan = build_plan(cfg)
    serialized = json.dumps(plan, indent=2, sort_keys=True)
    results = build_results(plan)

    if args.write_plan:
        write_json(args.write_plan, plan)
    if args.write_result_template:
        write_json(args.write_result_template, results)

    if args.execute == "none":
        sys.stdout.write(serialized + "\n")
        return 0

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "plan.json", plan)
    results_path = out_dir / "results.json"

    sys.stderr.write(
        f"eval: config={config_path} execute={args.execute}\n"
        f"eval: text={json.dumps(plan['shared_run']['text'])} ref={plan['shared_run']['reference_audio']['path']}\n"
        f"eval: generation seed={plan['shared_run']['generation']['seed']} "
        f"cfg={plan['shared_run']['generation']['cfg_scale']} "
        f"steps={plan['shared_run']['generation']['steps']} "
        f"max_frames={plan['shared_run']['generation']['max_frames']}\n"
        f"eval: canonical_model={plan['canonical']['model']} ggml_model={plan['ggml']['model']}\n"
    )

    failures: list[str] = []
    source_text = plan["shared_run"]["text"]
    if args.execute in {"canonical", "both"}:
        rc, _ = run_one(plan["canonical"]["command"], plan["canonical"]["repo"], plan["canonical"]["log_path"])
        results["canonical"]["return_code"] = rc
        results["canonical"]["status"] = "ok" if rc == 0 else "failed"
        results["canonical"]["output_sha256"] = maybe_sha256(plan["canonical"]["output_wav"])
        if rc != 0:
            failures.append(f"canonical rc={rc}")
        else:
            asr_rc, asr_log = run_one(plan["canonical"]["asr_command"], str(Path(cfg["ggml_cli"]).resolve().parent.parent.parent), plan["canonical"]["asr_log_path"])
            results["canonical"]["asr_return_code"] = asr_rc
            results["canonical"]["asr_transcript"] = asr_log.strip() if asr_log else None
            if asr_rc == 0:
                results["canonical"]["recall"] = compute_recall(source_text, asr_log)
            else:
                failures.append(f"canonical asr rc={asr_rc}")
    if args.execute in {"ggml", "both"}:
        rc, _ = run_one(plan["ggml"]["command"], str(Path(cfg["ggml_cli"]).resolve().parent.parent.parent), plan["ggml"]["log_path"], {"VIBEVOICE_BACKEND": "cpu"})
        results["ggml"]["return_code"] = rc
        results["ggml"]["status"] = "ok" if rc == 0 else "failed"
        results["ggml"]["output_sha256"] = maybe_sha256(plan["ggml"]["output_wav"])
        if rc != 0:
            failures.append(f"ggml rc={rc}")
        else:
            asr_rc, asr_log = run_one(plan["ggml"]["asr_command"], str(Path(cfg["ggml_cli"]).resolve().parent.parent.parent), plan["ggml"]["asr_log_path"])
            results["ggml"]["asr_return_code"] = asr_rc
            results["ggml"]["asr_transcript"] = asr_log.strip() if asr_log else None
            if asr_rc == 0:
                results["ggml"]["recall"] = compute_recall(source_text, asr_log)
            else:
                failures.append(f"ggml asr rc={asr_rc}")

    # Threshold check is meaningful only when both sides produced recall values
    if args.execute == "both" and results["canonical"].get("recall") is not None and results["ggml"].get("recall") is not None:
        passed, msg = check_threshold(results)
        results["threshold_check"] = {"passed": passed, "message": msg}
        if not passed:
            failures.append(msg)

    write_json(results_path, results)

    if failures:
        sys.stderr.write("eval harness failures: " + ", ".join(failures) + "\n")
        return 1
    sys.stdout.write(serialized + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
