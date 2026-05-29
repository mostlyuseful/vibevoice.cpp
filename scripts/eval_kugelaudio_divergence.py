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
import functools
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_EVALUATORS = {
    "transcription": {
        "backend": "faster-whisper",
        "model": "small.en",
        "language": "en",
        "compute_type": "int8",
        "beam_size": 5,
    },
    "speaker_similarity": {
        "backend": "speechbrain-ecapa",
        "model": "speechbrain/spkrec-ecapa-voxceleb",
    },
}

DEFAULT_METRIC_THRESHOLDS = {
    "transcript_recall": {"ratio": 0.95, "floor": 0.80},
    "speaker_similarity": {"ratio": 0.95, "floor": 0.60},
}


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
    p.add_argument(
        "--ggml-backend",
        choices=["cpu", "cuda", "vulkan"],
        help="Override ggml backend selection for the ggml run (defaults to config or cpu).",
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


def resolve_optional_path(base: Path, value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return resolve_path(base, text)


def validate_exists(label: str, path_value: str | None, allow_missing: bool) -> None:
    if path_value is None:
        return
    if allow_missing:
        return
    if not Path(path_value).exists():
        raise FileNotFoundError(f"{label} does not exist: {path_value}")


def validate_threshold_metric(name: str, metric: dict[str, Any]) -> dict[str, float]:
    ratio = float(metric.get("ratio", DEFAULT_METRIC_THRESHOLDS[name]["ratio"]))
    floor = float(metric.get("floor", DEFAULT_METRIC_THRESHOLDS[name]["floor"]))
    if not (0.0 <= ratio <= 1.0):
        raise ValueError(f"metric_thresholds.{name}.ratio must be within [0, 1]")
    if not (0.0 <= floor <= 1.0):
        raise ValueError(f"metric_thresholds.{name}.floor must be within [0, 1]")
    return {"ratio": ratio, "floor": floor}


def normalize_config(cfg: dict[str, Any], config_path: Path) -> dict[str, Any]:
    base = config_path.parent.resolve()
    out_dir = resolve_path(base, str(require(cfg, "output_dir")))

    evaluators = cfg.get("evaluators", {})
    if evaluators is None:
        evaluators = {}
    if not isinstance(evaluators, dict):
        raise ValueError("evaluators must be a JSON object")

    transcription_cfg = dict(DEFAULT_EVALUATORS["transcription"])
    transcription_cfg.update(evaluators.get("transcription", {}) or {})
    speaker_cfg = dict(DEFAULT_EVALUATORS["speaker_similarity"])
    speaker_cfg.update(evaluators.get("speaker_similarity", {}) or {})

    thresholds_cfg = cfg.get("metric_thresholds", {})
    if thresholds_cfg is None:
        thresholds_cfg = {}
    if not isinstance(thresholds_cfg, dict):
        raise ValueError("metric_thresholds must be a JSON object")

    norm = {
        "canonical_repo": resolve_path(base, str(require(cfg, "canonical_repo"))),
        "canonical_model": str(require(cfg, "canonical_model")),
        "ggml_cli": resolve_path(base, str(require(cfg, "ggml_cli"))),
        "ggml_model": resolve_path(base, str(require(cfg, "ggml_model"))),
        "ggml_tokenizer": resolve_path(base, str(require(cfg, "ggml_tokenizer"))),
        "ggml_model_q8_0": resolve_optional_path(base, cfg.get("ggml_model_q8_0")),
        "ggml_backend": str(cfg.get("ggml_backend", "cpu")).strip().lower() or "cpu",
        "reference_audio": resolve_path(base, str(require(cfg, "reference_audio"))),
        "text": str(require(cfg, "text")),
        "output_dir": out_dir,
        "seed": int(cfg.get("seed", 12345)),
        "cfg_scale": float(cfg.get("cfg_scale", 1.0)),
        "steps": int(cfg.get("steps", 8)),
        "max_frames": int(cfg.get("max_frames", 32)),
        "max_new_tokens": int(cfg.get("max_new_tokens", 4096)),
        "voice_cloned_sample": cfg.get("voice_cloned_sample", {}),
        "evaluators": {
            "transcription": {
                "backend": str(transcription_cfg.get("backend", "faster-whisper")).strip().lower(),
                "model": str(transcription_cfg.get("model", DEFAULT_EVALUATORS["transcription"]["model"])),
                "language": str(transcription_cfg.get("language", "en")).strip() or "en",
                "compute_type": str(transcription_cfg.get("compute_type", "int8")).strip() or "int8",
                "beam_size": int(transcription_cfg.get("beam_size", 5)),
            },
            "speaker_similarity": {
                "backend": str(speaker_cfg.get("backend", "speechbrain-ecapa")).strip().lower(),
                "model": str(speaker_cfg.get("model", DEFAULT_EVALUATORS["speaker_similarity"]["model"])),
            },
        },
        "metric_thresholds": {
            "transcript_recall": validate_threshold_metric(
                "transcript_recall", thresholds_cfg.get("transcript_recall", {}) or {}
            ),
            "speaker_similarity": validate_threshold_metric(
                "speaker_similarity", thresholds_cfg.get("speaker_similarity", {}) or {}
            ),
        },
    }
    if not norm["text"].strip():
        raise ValueError("text must not be empty")
    if norm["steps"] <= 0:
        raise ValueError("steps must be > 0")
    if norm["max_frames"] <= 0:
        raise ValueError("max_frames must be > 0")
    if norm["max_new_tokens"] <= 0:
        raise ValueError("max_new_tokens must be > 0")
    if norm["ggml_backend"] not in {"cpu", "cuda", "vulkan"}:
        raise ValueError("ggml_backend must be one of: cpu, cuda, vulkan")
    if norm["evaluators"]["transcription"]["backend"] != "faster-whisper":
        raise ValueError("evaluators.transcription.backend must currently be 'faster-whisper'")
    if norm["evaluators"]["transcription"]["beam_size"] <= 0:
        raise ValueError("evaluators.transcription.beam_size must be > 0")
    if norm["evaluators"]["speaker_similarity"]["backend"] != "speechbrain-ecapa":
        raise ValueError("evaluators.speaker_similarity.backend must currently be 'speechbrain-ecapa'")
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


def summarize_text_for_log(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"chars={len(text)} sha256={digest}"


def summarize_reference_for_log(path_value: str, sha256_value: str | None) -> str:
    if sha256_value:
        return f"present=yes sha256={sha256_value[:16]}"
    return f"present={Path(path_value).exists()} sha256=<missing>"


def build_canonical_inline_code(plan: dict[str, Any]) -> str:
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
    canonical_transcription_log = str((out_dir / "canonical_transcription.json").resolve())
    ggml_transcription_log = str((out_dir / "ggml_transcription.json").resolve())
    canonical_speaker_log = str((out_dir / "canonical_speaker_similarity.json").resolve())
    ggml_speaker_log = str((out_dir / "ggml_speaker_similarity.json").resolve())

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
            "evaluators": cfg["evaluators"],
            "metric_thresholds": cfg["metric_thresholds"],
        },
        "canonical": {
            "repo": cfg["canonical_repo"],
            "model": cfg["canonical_model"],
            "output_wav": canonical_out,
            "log_path": canonical_log,
            "transcription_log_path": canonical_transcription_log,
            "speaker_similarity_log_path": canonical_speaker_log,
        },
        "ggml": {
            "cli": cfg["ggml_cli"],
            "model": cfg["ggml_model"],
            "tokenizer": cfg["ggml_tokenizer"],
            "backend": cfg["ggml_backend"],
            "output_wav": ggml_out,
            "log_path": ggml_log,
            "transcription_log_path": ggml_transcription_log,
            "speaker_similarity_log_path": ggml_speaker_log,
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
        "schema_version": 2,
        "shared_run": plan["shared_run"],
        "canonical": {
            "command": plan["canonical"]["command"],
            "log_path": plan["canonical"]["log_path"],
            "output_wav": plan["canonical"]["output_wav"],
            "transcription_log_path": plan["canonical"]["transcription_log_path"],
            "speaker_similarity_log_path": plan["canonical"]["speaker_similarity_log_path"],
            "status": "planned",
            "return_code": None,
            "output_sha256": None,
            "transcript": None,
            "transcript_recall": None,
            "speaker_similarity": None,
        },
        "ggml": {
            "command": plan["ggml"]["command"],
            "log_path": plan["ggml"]["log_path"],
            "output_wav": plan["ggml"]["output_wav"],
            "backend": plan["ggml"]["backend"],
            "transcription_log_path": plan["ggml"]["transcription_log_path"],
            "speaker_similarity_log_path": plan["ggml"]["speaker_similarity_log_path"],
            "status": "planned",
            "return_code": None,
            "output_sha256": None,
            "transcript": None,
            "transcript_recall": None,
            "speaker_similarity": None,
        },
        "metric_thresholds": plan["shared_run"]["metric_thresholds"],
        "threshold_check": None,
    }


def write_json(path_value: str | Path, data: dict[str, Any]) -> None:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def extract_content(raw: str) -> str:
    """Extract Content fields from legacy JSON-like transcript output."""
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


def normalize_transcript_text(raw: str) -> str:
    raw = raw.strip()
    if '"Content":"' in raw:
        extracted = extract_content(raw).strip()
        if extracted:
            return extracted
    return raw


def word_set(s: str) -> set[str]:
    import re
    return set(re.sub(r"[^A-Za-z0-9]+", " ", s).lower().split())


def compute_recall(source_text: str, transcript: str) -> float:
    src = word_set(source_text)
    out = word_set(normalize_transcript_text(transcript))
    if not src:
        return 0.0
    hits = len(src & out)
    return hits / len(src)


@functools.lru_cache(maxsize=4)
def load_faster_whisper_model(model_name: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError(
            "faster-whisper is required for eval execution; install it in the active Python environment"
        ) from e
    return WhisperModel(model_name, device="cpu", compute_type=compute_type)


@functools.lru_cache(maxsize=4)
def load_speaker_recognizer(model_name: str):
    try:
        from speechbrain.inference.speaker import SpeakerRecognition
    except ImportError as e:
        raise RuntimeError(
            "speechbrain is required for speaker-similarity eval; install it in the active Python environment"
        ) from e
    return SpeakerRecognition.from_hparams(source=model_name, run_opts={"device": "cpu"})


def coerce_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return coerce_float(value[0])
    return float(value)


def run_transcription(evaluator_cfg: dict[str, Any], audio_path: str) -> tuple[str, dict[str, Any]]:
    model = load_faster_whisper_model(evaluator_cfg["model"], evaluator_cfg["compute_type"])
    segments, info = model.transcribe(
        audio_path,
        beam_size=int(evaluator_cfg["beam_size"]),
        language=evaluator_cfg.get("language") or None,
        condition_on_previous_text=False,
    )
    pieces: list[str] = []
    for segment in segments:
        text = getattr(segment, "text", "")
        if text:
            stripped = text.strip()
            if stripped:
                pieces.append(stripped)
    transcript = " ".join(pieces).strip()
    meta = {
        "backend": evaluator_cfg["backend"],
        "model": evaluator_cfg["model"],
        "language": evaluator_cfg.get("language"),
        "compute_type": evaluator_cfg["compute_type"],
        "beam_size": int(evaluator_cfg["beam_size"]),
        "detected_language": getattr(info, "language", None),
        "language_probability": coerce_float(getattr(info, "language_probability", 0.0)) if getattr(info, "language_probability", None) is not None else None,
    }
    return transcript, meta


def run_speaker_similarity(evaluator_cfg: dict[str, Any], reference_audio: str, audio_path: str) -> tuple[float, dict[str, Any]]:
    recognizer = load_speaker_recognizer(evaluator_cfg["model"])
    score, prediction = recognizer.verify_files(reference_audio, audio_path)
    similarity = coerce_float(score)
    meta = {
        "backend": evaluator_cfg["backend"],
        "model": evaluator_cfg["model"],
        "reference_audio": reference_audio,
        "generated_audio": audio_path,
        "prediction": bool(coerce_float(prediction)),
        "score": similarity,
    }
    return similarity, meta


def run_one(command: list[str], cwd: str, log_path: str, extra_env: dict[str, str] | None = None) -> tuple[int, str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(command, cwd=cwd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    return int(proc.returncode), Path(log_path).read_text(encoding="utf-8") if Path(log_path).exists() else ""


def evaluate_metric_threshold(name: str, canonical_value: Any, ggml_value: Any, ratio_req: float, floor_req: float) -> tuple[bool, str]:
    if canonical_value is None or ggml_value is None:
        return False, f"{name}: missing values"
    canonical_f = float(canonical_value)
    ggml_f = float(ggml_value)
    if canonical_f == 0.0:
        return False, f"{name}: canonical value is zero (ggml={ggml_f:.4f})"
    ratio = ggml_f / canonical_f if canonical_f > 0 else 0.0
    floor_ok = ggml_f >= floor_req
    ratio_ok = ratio >= ratio_req
    if floor_ok and ratio_ok:
        return True, (
            f"{name}={ggml_f:.4f} canonical={canonical_f:.4f} "
            f"ratio={ratio:.2%} floor={floor_req:.2f}"
        )
    reasons: list[str] = []
    if not ratio_ok:
        reasons.append(f"ratio {ratio:.2%} < {ratio_req:.0%} of canonical")
    if not floor_ok:
        reasons.append(f"floor {ggml_f:.4f} < {floor_req:.2f}")
    return False, f"{name}: " + "; ".join(reasons) + f" (ggml={ggml_f:.4f} canonical={canonical_f:.4f})"


def check_threshold(results: dict[str, Any]) -> tuple[bool, str]:
    thresholds = results.get("metric_thresholds") or DEFAULT_METRIC_THRESHOLDS
    checks = [
        (
            "transcript_recall",
            results.get("canonical", {}).get("transcript_recall"),
            results.get("ggml", {}).get("transcript_recall"),
            float(thresholds.get("transcript_recall", {}).get("ratio", DEFAULT_METRIC_THRESHOLDS["transcript_recall"]["ratio"])),
            float(thresholds.get("transcript_recall", {}).get("floor", DEFAULT_METRIC_THRESHOLDS["transcript_recall"]["floor"])),
        ),
        (
            "speaker_similarity",
            results.get("canonical", {}).get("speaker_similarity"),
            results.get("ggml", {}).get("speaker_similarity"),
            float(thresholds.get("speaker_similarity", {}).get("ratio", DEFAULT_METRIC_THRESHOLDS["speaker_similarity"]["ratio"])),
            float(thresholds.get("speaker_similarity", {}).get("floor", DEFAULT_METRIC_THRESHOLDS["speaker_similarity"]["floor"])),
        ),
    ]

    failures: list[str] = []
    passes: list[str] = []
    for name, canonical_value, ggml_value, ratio_req, floor_req in checks:
        passed, message = evaluate_metric_threshold(name, canonical_value, ggml_value, ratio_req, floor_req)
        if passed:
            passes.append(message)
        else:
            failures.append(message)
    if failures:
        return False, "FAIL  " + " | ".join(failures)
    return True, "PASS  " + " | ".join(passes)


def evaluate_outputs_for_side(
    side_name: str,
    plan: dict[str, Any],
    results: dict[str, Any],
    source_text: str,
) -> list[str]:
    failures: list[str] = []
    side_results = results[side_name]
    side_plan = plan[side_name]
    transcription_cfg = plan["shared_run"]["evaluators"]["transcription"]
    speaker_cfg = plan["shared_run"]["evaluators"]["speaker_similarity"]
    reference_audio = plan["shared_run"]["reference_audio"]["path"]

    try:
        transcript, transcript_meta = run_transcription(transcription_cfg, side_plan["output_wav"])
        transcript_meta["transcript"] = transcript
        transcript_meta["transcript_recall"] = compute_recall(source_text, transcript)
        write_json(side_plan["transcription_log_path"], transcript_meta)
        side_results["transcript"] = transcript
        side_results["transcript_recall"] = transcript_meta["transcript_recall"]
    except Exception as e:
        failures.append(f"{side_name} transcription: {e}")

    try:
        similarity, similarity_meta = run_speaker_similarity(speaker_cfg, reference_audio, side_plan["output_wav"])
        write_json(side_plan["speaker_similarity_log_path"], similarity_meta)
        side_results["speaker_similarity"] = similarity
    except Exception as e:
        failures.append(f"{side_name} speaker_similarity: {e}")

    return failures


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = normalize_config(load_json(config_path), config_path)
    if args.ggml_backend:
        cfg["ggml_backend"] = args.ggml_backend

    validate_exists("canonical_repo", cfg["canonical_repo"], args.allow_missing_artifacts)
    validate_exists("ggml_cli", cfg["ggml_cli"], args.allow_missing_artifacts)
    validate_exists("ggml_model", cfg["ggml_model"], args.allow_missing_artifacts)
    validate_exists("ggml_tokenizer", cfg["ggml_tokenizer"], args.allow_missing_artifacts)
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

    transcription_cfg = plan["shared_run"]["evaluators"]["transcription"]
    speaker_cfg = plan["shared_run"]["evaluators"]["speaker_similarity"]
    sys.stderr.write(
        f"eval: config={config_path} execute={args.execute}\n"
        f"eval: text_summary={summarize_text_for_log(plan['shared_run']['text'])} "
        f"ref_summary={summarize_reference_for_log(plan['shared_run']['reference_audio']['path'], plan['shared_run']['reference_audio']['sha256'])}\n"
        f"eval: generation seed={plan['shared_run']['generation']['seed']} "
        f"cfg={plan['shared_run']['generation']['cfg_scale']} "
        f"steps={plan['shared_run']['generation']['steps']} "
        f"max_frames={plan['shared_run']['generation']['max_frames']}\n"
        f"eval: canonical_model={plan['canonical']['model']} ggml_model={plan['ggml']['model']} ggml_backend={plan['ggml']['backend']}\n"
        f"eval: transcription_backend={transcription_cfg['backend']} transcription_model={transcription_cfg['model']} "
        f"speaker_backend={speaker_cfg['backend']} speaker_model={speaker_cfg['model']}\n"
    )

    failures: list[str] = []
    source_text = plan["shared_run"]["text"]
    ggml_cwd = str(Path(cfg["ggml_cli"]).resolve().parent.parent.parent)

    if args.execute in {"canonical", "both"}:
        rc, _ = run_one(plan["canonical"]["command"], plan["canonical"]["repo"], plan["canonical"]["log_path"])
        results["canonical"]["return_code"] = rc
        results["canonical"]["status"] = "ok" if rc == 0 else "failed"
        results["canonical"]["output_sha256"] = maybe_sha256(plan["canonical"]["output_wav"])
        if rc != 0:
            failures.append(f"canonical rc={rc}")
        else:
            failures.extend(evaluate_outputs_for_side("canonical", plan, results, source_text))

    if args.execute in {"ggml", "both"}:
        rc, _ = run_one(
            plan["ggml"]["command"],
            ggml_cwd,
            plan["ggml"]["log_path"],
            {"KUGELAUDIO_BACKEND": plan["ggml"]["backend"]},
        )
        results["ggml"]["return_code"] = rc
        results["ggml"]["status"] = "ok" if rc == 0 else "failed"
        results["ggml"]["output_sha256"] = maybe_sha256(plan["ggml"]["output_wav"])
        if rc != 0:
            failures.append(f"ggml rc={rc}")
        else:
            failures.extend(evaluate_outputs_for_side("ggml", plan, results, source_text))

    if args.execute == "both":
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
