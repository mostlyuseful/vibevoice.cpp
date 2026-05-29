#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
"""Sweep KugelAudio long-form chunking settings and collect artifacts.

This is meant for manual listening/evaluation on a fast machine (for example
CUDA). It runs the published CLI across a parameter grid, stores WAV/log files,
and writes JSON/CSV summaries with basic duration/rate metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any


def parse_csv_ints(text: str) -> list[int]:
    out: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return out


def parse_csv_floats(text: str) -> list[float]:
    out: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(float(item))
    return out


def parse_csv_strings(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text", help="Inline text to synthesize")
    p.add_argument("--text-file", help="Read synthesis text from a file")
    p.add_argument(
        "--output-dir",
        default=str((Path("runs") / "longform-sweep").resolve()),
        help="Directory for generated WAVs/logs/summary files",
    )
    p.add_argument(
        "--backend",
        default=os.environ.get("KUGELAUDIO_BACKEND", "cuda"),
        help="Backend passed to the CLI via KUGELAUDIO_BACKEND (default: cuda)",
    )
    p.add_argument(
        "--cli",
        default=os.environ.get("KUGELAUDIO_CLI", ""),
        help="Path to kugelaudio-cli (default: KUGELAUDIO_CLI env)",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("KUGELAUDIO_MODEL", ""),
        help="Path to KugelAudio GGUF (default: KUGELAUDIO_MODEL env)",
    )
    p.add_argument(
        "--tokenizer",
        default=os.environ.get("KUGELAUDIO_TOKENIZER", ""),
        help="Path to tokenizer GGUF (default: KUGELAUDIO_TOKENIZER env)",
    )
    p.add_argument(
        "--ref-audio",
        default=os.environ.get("KUGELAUDIO_REF_WAV", ""),
        help="Path to reference WAV (default: KUGELAUDIO_REF_WAV env)",
    )
    p.add_argument("--max-frames", type=int, default=256)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument(
        "--final-decoder-backend",
        default="auto",
        choices=["auto", "cpu", "stream"],
        help="Forwarded to kugelaudio-cli; use cpu as a hybrid workaround if CUDA final decode is unstable",
    )
    p.add_argument(
        "--cfg-scales",
        default="1.0,1.2,1.5",
        help="Comma-separated cfg_scale sweep (important long-form pacing knob)",
    )
    p.add_argument(
        "--max-words-per-chunk",
        default="0,320,256,224,192,160,128,96,80,64",
        help="Comma-separated max_words_per_chunk sweep. 0 disables chunking baseline.",
    )
    p.add_argument(
        "--overlap-sentences",
        default="0,1,2",
        help="Comma-separated overlap_sentences sweep",
    )
    p.add_argument(
        "--chunking-strategies",
        default="heuristic,syntax-aware",
        help="Comma-separated chunking strategies",
    )
    p.add_argument(
        "--pause-mode",
        default="punctuation",
        choices=["none", "punctuation", "speaker-aware"],
        help="Fixed pause mode for the sweep",
    )
    p.add_argument(
        "--crossfade-ms",
        type=int,
        default=30,
        help="Fixed crossfade for the sweep; use 0 to isolate pacing without seam smoothing",
    )
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Per-run timeout in seconds",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the plan without executing the CLI",
    )
    return p.parse_args()


def load_text(args: argparse.Namespace) -> str:
    if args.text and args.text_file:
        raise SystemExit("Choose either --text or --text-file, not both")
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8")
    if args.text:
        return args.text
    raise SystemExit("Provide --text or --text-file")


def require_file(label: str, value: str) -> str:
    if not value:
        raise SystemExit(f"Missing required {label}. Pass --{label.replace('_', '-')} or set the matching env var.")
    path = Path(value)
    if not path.exists():
        raise SystemExit(f"{label} does not exist: {path}")
    return str(path.resolve())


def read_wav_stats(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
    duration_s = frames / sample_rate if sample_rate > 0 else 0.0
    return {
        "frames": frames,
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width": sampwidth,
        "duration_s": duration_s,
    }


def sanitize_id(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("-")
    return "".join(out).strip("-") or "run"


def build_plan(args: argparse.Namespace, text: str) -> list[dict[str, Any]]:
    cfg_scales = parse_csv_floats(args.cfg_scales)
    chunk_sizes = parse_csv_ints(args.max_words_per_chunk)
    overlaps = parse_csv_ints(args.overlap_sentences)
    strategies = parse_csv_strings(args.chunking_strategies)
    if not cfg_scales:
        raise SystemExit("cfg_scales sweep is empty")
    if not chunk_sizes:
        raise SystemExit("max_words_per_chunk sweep is empty")
    if not overlaps:
        raise SystemExit("overlap_sentences sweep is empty")
    if not strategies:
        raise SystemExit("chunking_strategies sweep is empty")

    plan: list[dict[str, Any]] = []
    run_index = 0
    for cfg_scale in cfg_scales:
        for max_words in chunk_sizes:
            if max_words == 0:
                plan.append(
                    {
                        "run_index": run_index,
                        "cfg_scale": cfg_scale,
                        "max_words_per_chunk": 0,
                        "overlap_sentences": 0,
                        "chunking_strategy": "heuristic",
                        "pause_mode": args.pause_mode,
                        "crossfade_ms": args.crossfade_ms,
                        "notes": "baseline_no_chunking",
                    }
                )
                run_index += 1
                continue
            for overlap in overlaps:
                for strategy in strategies:
                    plan.append(
                        {
                            "run_index": run_index,
                            "cfg_scale": cfg_scale,
                            "max_words_per_chunk": max_words,
                            "overlap_sentences": overlap,
                            "chunking_strategy": strategy,
                            "pause_mode": args.pause_mode,
                            "crossfade_ms": args.crossfade_ms,
                            "notes": "chunked",
                        }
                    )
                    run_index += 1
    return plan


def run_one(
    entry: dict[str, Any],
    *,
    args: argparse.Namespace,
    text: str,
    cli: str,
    model: str,
    tokenizer: str,
    ref_audio: str,
    out_dir: Path,
    word_count: int,
) -> dict[str, Any]:
    run_id = sanitize_id(
        f"{entry['run_index']:03d}-cfg{entry['cfg_scale']}-mw{entry['max_words_per_chunk']}-ov{entry['overlap_sentences']}-strat-{entry['chunking_strategy']}"
    )
    wav_path = out_dir / f"{run_id}.wav"
    log_path = out_dir / f"{run_id}.log"

    argv = [
        cli,
        "--model",
        model,
        "--tokenizer",
        tokenizer,
        "--ref-audio",
        ref_audio,
        "--text",
        text,
        "--out",
        str(wav_path),
        "--max-frames",
        str(args.max_frames),
        "--steps",
        str(args.steps),
        "--cfg",
        str(entry["cfg_scale"]),
        "--seed",
        str(args.seed),
        "--verbose",
        "--final-decoder-backend",
        args.final_decoder_backend,
    ]
    if entry["max_words_per_chunk"] > 0:
        argv.extend(
            [
                "--max-words-per-chunk",
                str(entry["max_words_per_chunk"]),
                "--overlap-sentences",
                str(entry["overlap_sentences"]),
                "--chunking-strategy",
                entry["chunking_strategy"],
                "--pause-mode",
                entry["pause_mode"],
                "--crossfade-ms",
                str(entry["crossfade_ms"]),
            ]
        )

    env = os.environ.copy()
    env["KUGELAUDIO_BACKEND"] = args.backend

    command_str = shlex.join(argv)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"backend={args.backend}\n")
        logf.write(f"command={command_str}\n\n")
        logf.flush()
        proc = subprocess.run(
            argv,
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
            check=False,
            text=True,
        )
    elapsed_s = time.time() - started

    result: dict[str, Any] = dict(entry)
    result.update(
        {
            "run_id": run_id,
            "command": command_str,
            "backend": args.backend,
            "rc": proc.returncode,
            "elapsed_s": elapsed_s,
            "wav_path": str(wav_path),
            "log_path": str(log_path),
            "word_count": word_count,
        }
    )

    if wav_path.exists() and proc.returncode == 0:
        wav_stats = read_wav_stats(wav_path)
        duration_s = float(wav_stats["duration_s"])
        result.update(wav_stats)
        result["words_per_second"] = (word_count / duration_s) if duration_s > 0 else None
    else:
        result["duration_s"] = None
        result["words_per_second"] = None

    return result


def write_summary(results: list[dict[str, Any]], out_dir: Path) -> None:
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    csv_path = out_dir / "summary.csv"
    fieldnames = [
        "run_index",
        "run_id",
        "rc",
        "elapsed_s",
        "cfg_scale",
        "max_words_per_chunk",
        "overlap_sentences",
        "chunking_strategy",
        "pause_mode",
        "crossfade_ms",
        "duration_s",
        "words_per_second",
        "wav_path",
        "log_path",
        "notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in fieldnames})


def main() -> int:
    args = parse_args()
    text = load_text(args).strip()
    if not text:
        raise SystemExit("Input text must not be empty")

    cli = require_file("cli", args.cli)
    model = require_file("model", args.model)
    tokenizer = require_file("tokenizer", args.tokenizer)
    ref_audio = require_file("ref_audio", args.ref_audio)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    word_count = len(text.split())
    plan = build_plan(args, text)

    plan_path = out_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    text_path = out_dir / "input.txt"
    text_path.write_text(text, encoding="utf-8")

    print(f"[sweep] text_words={word_count} runs={len(plan)} backend={args.backend}")
    print(f"[sweep] output_dir={out_dir}")
    print("[sweep] important knobs in this sweep:")
    print("  - max_words_per_chunk  (likely cutoff threshold for pacing drift)")
    print("  - chunking_strategy    (heuristic vs syntax-aware)")
    print("  - overlap_sentences    (how much previous text stays in prompt context)")
    print("  - cfg_scale            (important long-form narration stability knob)")
    print("[sweep] fixed knobs:")
    print(f"  - pause_mode={args.pause_mode}")
    print(f"  - crossfade_ms={args.crossfade_ms}")
    print(f"  - steps={args.steps}")
    print(f"  - max_frames={args.max_frames}")
    print(f"  - final_decoder_backend={args.final_decoder_backend}")

    if args.dry_run:
        print(f"[sweep] dry-run only; wrote {plan_path}")
        return 0

    results: list[dict[str, Any]] = []
    for i, entry in enumerate(plan, start=1):
        print(
            f"[sweep] {i}/{len(plan)} cfg={entry['cfg_scale']} max_words={entry['max_words_per_chunk']} "
            f"overlap={entry['overlap_sentences']} strategy={entry['chunking_strategy']}"
        )
        try:
            result = run_one(
                entry,
                args=args,
                text=text,
                cli=cli,
                model=model,
                tokenizer=tokenizer,
                ref_audio=ref_audio,
                out_dir=out_dir,
                word_count=word_count,
            )
        except subprocess.TimeoutExpired:
            result = dict(entry)
            result.update(
                {
                    "run_id": sanitize_id(
                        f"{entry['run_index']:03d}-cfg{entry['cfg_scale']}-mw{entry['max_words_per_chunk']}-ov{entry['overlap_sentences']}-strat-{entry['chunking_strategy']}"
                    ),
                    "rc": -999,
                    "elapsed_s": args.timeout,
                    "duration_s": None,
                    "words_per_second": None,
                    "wav_path": str(out_dir / "<timed_out>.wav"),
                    "log_path": str(out_dir / "<timed_out>.log"),
                    "notes": "timeout",
                }
            )
        results.append(result)
        write_summary(results, out_dir)
        print(
            f"[sweep] rc={result['rc']} duration_s={result.get('duration_s')} "
            f"wps={result.get('words_per_second')} wav={Path(result['wav_path']).name}"
        )

    print(f"[sweep] wrote summaries: {out_dir / 'summary.json'} and {out_dir / 'summary.csv'}")
    print("[sweep] suggestion: first sort summary.csv by words_per_second, then listen around the chunk-size cutoff.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
