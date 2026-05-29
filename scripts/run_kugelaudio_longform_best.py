#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
"""Run the current best KugelAudio long-form profile plus optional ASR QA.

This is an orchestration helper for listening/eval runs. It intentionally shells
out to the product CLI and the optional ASR verifier instead of becoming runtime
logic.

Current default profile reflects the best real-reference long-form findings so
far:
  - f16 model
  - tail-reference continuity with 1200 ms tail
  - cfg=2.0, steps=20, max_frames=320
  - no overlap sentences
  - punctuation pauses, 60 ms crossfade
  - final text-end ellipsis padding to avoid the confirmed final-cut fixture

`tail-reference` is still env-gated in the CLI, so this script sets
KUGELAUDIO_ENABLE_RETIRED_CONTINUITY=1 for that explicit diagnostic profile.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_cli(root: Path) -> Path:
    for candidate in (
        root / "build-cuda" / "bin" / "kugelaudio-cli",
        root / "build" / "bin" / "kugelaudio-cli",
    ):
        if candidate.exists():
            return candidate
    return root / "build-cuda" / "bin" / "kugelaudio-cli"


def parse_args() -> argparse.Namespace:
    root = repo_root()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Text to synthesize")
    src.add_argument("--text-file", type=Path, help="Path to text file")

    p.add_argument("--ref-audio", type=Path, required=True, help="Real spoken reference WAV/audio")
    p.add_argument("--out", type=Path, required=True, help="Output WAV path")
    p.add_argument("--model", type=Path, default=root / "models" / "kugelaudio-f16.gguf", help="KugelAudio GGUF model")
    p.add_argument("--tokenizer", type=Path, default=root / "models" / "tokenizer.gguf", help="Tokenizer GGUF")
    p.add_argument("--cli", type=Path, default=default_cli(root), help="kugelaudio-cli path")

    p.add_argument("--seed", type=int, default=12345, help="Generation seed")
    p.add_argument("--max-words-per-chunk", type=int, default=60, help="Chunk word budget; 40/60 are current listening favorites")
    p.add_argument("--max-frames", type=int, default=320, help="Per-chunk frame budget")
    p.add_argument("--steps", type=int, default=20, help="DPM-Solver steps")
    p.add_argument("--cfg", type=float, default=2.0, help="CFG scale")
    p.add_argument("--overlap-sentences", type=int, default=0, help="Chunk overlap sentence count")
    p.add_argument("--chunk-continuity", choices=("tail-reference", "none"), default="tail-reference", help="Continuity profile")
    p.add_argument("--continuity-tail-ms", type=int, default=1200, help="Tail-reference duration")
    p.add_argument("--pause-mode", choices=("none", "punctuation", "speaker-aware"), default="punctuation")
    p.add_argument("--crossfade-ms", type=int, default=60)
    p.add_argument("--chunking-strategy", choices=("heuristic", "syntax-aware"), default="heuristic")
    p.add_argument("--final-decoder-backend", choices=("auto", "cpu", "stream", "active"), default="auto")
    p.add_argument("--text-end-padding", choices=("none", "ellipsis"), default="ellipsis", help="Final generated text padding")
    p.add_argument("--chunk-boundary-cleanup", choices=("none", "trim-fade"), default="none", help="Optional seam trim/fade cleanup")
    p.add_argument("--chunk-boundary-leading-silence-ms", type=int, default=200)
    p.add_argument("--chunk-boundary-trailing-silence-ms", type=int, default=300)
    p.add_argument("--chunk-boundary-fade-ms", type=int, default=15)
    p.add_argument("--backend", default="cuda", help="KUGELAUDIO_BACKEND value; use cpu for CPU runs")
    p.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True, help="Pass --verbose to CLI")

    p.add_argument("--no-asr", action="store_true", help="Skip ASR verification")
    p.add_argument("--language", help="WhisperX language hint, e.g. de/en")
    p.add_argument("--asr-model", default="small", help="WhisperX model")
    p.add_argument("--asr-device", default="cpu", help="WhisperX device")
    p.add_argument("--asr-compute-type", default="int8", help="WhisperX compute type")
    p.add_argument("--min-ordered-coverage", type=float, default=0.90)
    p.add_argument("--min-tail-coverage", type=float, default=0.90)
    p.add_argument("--tail-words", type=int, default=24)
    p.add_argument("--transcript-file", type=Path, help="Existing transcript text; skip WhisperX")
    p.add_argument("--uvx-bin", default="uvx")
    p.add_argument("--whisperx-package", default="whisperx")

    p.add_argument("--log", type=Path, help="Generation log path; default: <out>.log")
    p.add_argument("--asr-json", type=Path, help="ASR JSON path; default: <out>.asr.json")
    p.add_argument("--metadata-out", type=Path, help="Run metadata JSON path; default: <out>.run.json")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return p.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"missing {label}: {path}")


def q(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str], dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(q(cmd))
    if dry_run:
        return 0
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + q(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        log.write(f"\nreturn_code={proc.returncode}\n")
        return proc.returncode


def source_arg(args: argparse.Namespace) -> list[str]:
    if args.text_file:
        return ["--text-file", str(args.text_file)]
    return ["--text", args.text or ""]


def source_text_for_asr(args: argparse.Namespace) -> list[str]:
    if args.text_file:
        return ["--text-file", str(args.text_file)]
    return ["--text", args.text or ""]


def main() -> int:
    args = parse_args()
    root = repo_root()
    require_file(args.cli, "kugelaudio-cli")
    require_file(args.model, "model")
    require_file(args.tokenizer, "tokenizer")
    require_file(args.ref_audio, "reference audio")
    if args.text_file:
        require_file(args.text_file, "text file")
    if args.transcript_file:
        require_file(args.transcript_file, "transcript file")

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.log or out.with_suffix(".log")
    asr_json = args.asr_json or out.with_suffix(".asr.json")
    metadata_out = args.metadata_out or out.with_suffix(".run.json")
    asr_work_dir = out.with_suffix("")
    asr_work_dir = asr_work_dir.parent / (asr_work_dir.name + ".asr")

    cmd = [
        str(args.cli),
        "--model", str(args.model),
        "--tokenizer", str(args.tokenizer),
        "--ref-audio", str(args.ref_audio),
        *source_arg(args),
        "--out", str(out),
        "--max-words-per-chunk", str(args.max_words_per_chunk),
        "--overlap-sentences", str(args.overlap_sentences),
        "--chunking-strategy", args.chunking_strategy,
        "--pause-mode", args.pause_mode,
        "--crossfade-ms", str(args.crossfade_ms),
        "--chunk-continuity", args.chunk_continuity,
        "--continuity-tail-ms", str(args.continuity_tail_ms),
        "--text-end-padding", args.text_end_padding,
        "--max-frames", str(args.max_frames),
        "--steps", str(args.steps),
        "--cfg", str(args.cfg),
        "--seed", str(args.seed),
        "--final-decoder-backend", args.final_decoder_backend,
    ]
    if args.chunk_boundary_cleanup != "none":
        cmd += [
            "--chunk-boundary-cleanup", args.chunk_boundary_cleanup,
            "--chunk-boundary-leading-silence-ms", str(args.chunk_boundary_leading_silence_ms),
            "--chunk-boundary-trailing-silence-ms", str(args.chunk_boundary_trailing_silence_ms),
            "--chunk-boundary-fade-ms", str(args.chunk_boundary_fade_ms),
        ]
    if args.verbose:
        cmd.append("--verbose")

    env = os.environ.copy()
    env["KUGELAUDIO_BACKEND"] = args.backend
    if args.chunk_continuity == "tail-reference":
        env["KUGELAUDIO_ENABLE_RETIRED_CONTINUITY"] = "1"

    started = time.time()
    gen_rc = run_logged(cmd, log_path, env, args.dry_run)
    if gen_rc != 0:
        print(f"generation failed, see {log_path}", file=sys.stderr)
        return gen_rc

    asr_rc: int | None = None
    asr_cmd: list[str] | None = None
    if not args.no_asr:
        verifier = root / "scripts" / "verify_longform_asr.py"
        require_file(verifier, "ASR verifier")
        asr_cmd = [
            sys.executable,
            str(verifier),
            *source_text_for_asr(args),
            "--audio", str(out),
            "--out", str(asr_json),
            "--work-dir", str(asr_work_dir),
            "--uvx-bin", args.uvx_bin,
            "--whisperx-package", args.whisperx_package,
            "--model", args.asr_model,
            "--device", args.asr_device,
            "--compute-type", args.asr_compute_type,
            "--min-ordered-coverage", str(args.min_ordered_coverage),
            "--min-tail-coverage", str(args.min_tail_coverage),
            "--tail-words", str(args.tail_words),
        ]
        if args.language:
            asr_cmd += ["--language", args.language]
        if args.transcript_file:
            asr_cmd += ["--transcript-file", str(args.transcript_file)]
        print(q(asr_cmd))
        if not args.dry_run:
            asr_rc = subprocess.run(asr_cmd).returncode
        else:
            asr_rc = 0

    metadata: dict[str, Any] = {
        "schema": "kugelaudio_longform_best_run.v1",
        "output": str(out),
        "generation_log": str(log_path),
        "asr_json": None if args.no_asr else str(asr_json),
        "generation_return_code": gen_rc,
        "asr_return_code": asr_rc,
        "elapsed_seconds": round(time.time() - started, 3),
        "profile": {
            "model": str(args.model),
            "tokenizer": str(args.tokenizer),
            "ref_audio": str(args.ref_audio),
            "seed": args.seed,
            "max_words_per_chunk": args.max_words_per_chunk,
            "max_frames": args.max_frames,
            "steps": args.steps,
            "cfg": args.cfg,
            "chunk_continuity": args.chunk_continuity,
            "continuity_tail_ms": args.continuity_tail_ms,
            "overlap_sentences": args.overlap_sentences,
            "pause_mode": args.pause_mode,
            "crossfade_ms": args.crossfade_ms,
            "text_end_padding": args.text_end_padding,
            "chunk_boundary_cleanup": args.chunk_boundary_cleanup,
            "backend": args.backend,
        },
        "generation_command": cmd,
        "asr_command": asr_cmd,
    }
    if not args.dry_run:
        metadata_out.parent.mkdir(parents=True, exist_ok=True)
        metadata_out.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"metadata: {metadata_out}")

    return 0 if asr_rc in (None, 0) else asr_rc


if __name__ == "__main__":
    raise SystemExit(main())
