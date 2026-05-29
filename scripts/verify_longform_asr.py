#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
"""Optional long-form TTS transcript coverage verifier.

This is an eval/QA helper, not a runtime dependency. By default it shells out to
WhisperX through uvx, matching the smoke-test style:

  uvx whisperx --output_format txt --verbose True --print_progress True audio.wav

For tests or already-transcribed audio, pass --transcript-file to skip ASR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


@dataclass(frozen=True)
class WordSpan:
    word: str
    start: int
    end: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Expected source text")
    src.add_argument("--text-file", type=Path, help="Expected source text file")
    p.add_argument("--audio", type=Path, required=True, help="Generated WAV/audio to verify")
    p.add_argument("--out", type=Path, help="Write coverage JSON here")
    p.add_argument("--transcript-file", type=Path, help="Existing transcript text; skip WhisperX")
    p.add_argument("--work-dir", type=Path, help="Directory for WhisperX txt output")
    p.add_argument("--uvx-bin", default="uvx", help="uvx executable (default: uvx)")
    p.add_argument("--whisperx-package", default="whisperx", help="uvx package/tool name (default: whisperx)")
    p.add_argument("--language", help="Optional WhisperX language hint, e.g. en")
    p.add_argument("--model", help="Optional WhisperX model name")
    p.add_argument("--device", help="Optional WhisperX device, e.g. cpu/cuda")
    p.add_argument("--compute-type", help="Optional WhisperX compute type")
    p.add_argument("--max-missing-spans", type=int, default=24, help="Max missing spans to include in JSON/summary")
    p.add_argument("--min-ordered-coverage", type=float, default=0.95, help="Exit nonzero if ordered coverage is below this")
    p.add_argument("--min-tail-coverage", type=float, default=1.0, help="Exit nonzero if tail coverage is below this")
    p.add_argument("--tail-words", type=int, default=24, help="Number of final source words to treat as tail")
    p.add_argument("--print-transcript", action="store_true", help="Also print normalized transcript")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_text(args: argparse.Namespace) -> str:
    if args.text_file:
        return args.text_file.read_text(encoding="utf-8")
    return args.text or ""


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def word_spans(text: str) -> list[WordSpan]:
    return [WordSpan(m.group(0).lower(), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def run_whisperx(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    audio = args.audio.resolve()
    if not audio.exists():
        raise FileNotFoundError(f"audio does not exist: {audio}")
    if args.work_dir:
        work_dir = args.work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = None
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="kugelaudio-whisperx-")
        work_dir = Path(cleanup.name)

    cmd = [
        args.uvx_bin,
        args.whisperx_package,
        "--output_format",
        "txt",
        "--verbose",
        "True",
        "--print_progress",
        "True",
        "--output_dir",
        str(work_dir),
    ]
    if args.language:
        cmd += ["--language", args.language]
    if args.model:
        cmd += ["--model", args.model]
    if args.device:
        cmd += ["--device", args.device]
    if args.compute_type:
        cmd += ["--compute_type", args.compute_type]
    cmd.append(str(audio))

    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    txt_path = work_dir / f"{audio.stem}.txt"
    transcript = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""
    meta = {
        "backend": "whisperx-cli",
        "command": cmd,
        "return_code": proc.returncode,
        "output_dir": str(work_dir),
        "txt_path": str(txt_path),
        "log_tail": proc.stdout[-4000:],
    }
    if cleanup is not None:
        # Keep transcript in memory but remove temp files.
        cleanup.cleanup()
    if proc.returncode != 0:
        raise RuntimeError(f"WhisperX failed rc={proc.returncode}\n{proc.stdout[-4000:]}")
    if not transcript.strip():
        raise RuntimeError(f"WhisperX completed but transcript txt was empty/missing: {txt_path}")
    return transcript, meta


def load_transcript(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if args.transcript_file:
        path = args.transcript_file.resolve()
        return path.read_text(encoding="utf-8"), {"backend": "transcript-file", "path": str(path)}
    return run_whisperx(args)


def coverage(source: str, transcript: str, *, max_missing_spans: int, tail_words: int) -> dict[str, Any]:
    src = word_spans(source)
    hyp = word_spans(transcript)
    src_words = [w.word for w in src]
    hyp_words = [w.word for w in hyp]
    matcher = SequenceMatcher(a=src_words, b=hyp_words, autojunk=False)
    covered = [False] * len(src_words)
    for block in matcher.get_matching_blocks():
        for i in range(block.a, block.a + block.size):
            if 0 <= i < len(covered):
                covered[i] = True

    missing_ranges: list[tuple[int, int]] = []
    i = 0
    while i < len(covered):
        if covered[i]:
            i += 1
            continue
        j = i + 1
        while j < len(covered) and not covered[j]:
            j += 1
        missing_ranges.append((i, j))
        i = j

    missing_spans = []
    for begin, end in missing_ranges[:max_missing_spans]:
        char_start = src[begin].start
        char_end = src[end - 1].end
        missing_spans.append({
            "word_start": begin,
            "word_end_exclusive": end,
            "word_count": end - begin,
            "text": normalize_spaces(source[char_start:char_end]),
        })

    src_set = set(src_words)
    hyp_set = set(hyp_words)
    tail_n = min(max(tail_words, 0), len(src_words))
    tail_cov = 1.0
    tail_missing: list[str] = []
    if tail_n:
        tail_flags = covered[-tail_n:]
        tail_cov = sum(1 for v in tail_flags if v) / tail_n
        tail_missing = [src_words[len(src_words) - tail_n + k] for k, v in enumerate(tail_flags) if not v]

    trailing_missing_words = 0
    for v in reversed(covered):
        if v:
            break
        trailing_missing_words += 1

    return {
        "source_word_count": len(src_words),
        "transcript_word_count": len(hyp_words),
        "ordered_coverage": (sum(1 for v in covered if v) / len(covered)) if covered else 0.0,
        "unique_word_recall": (len(src_set & hyp_set) / len(src_set)) if src_set else 0.0,
        "missing_word_count": sum(1 for v in covered if not v),
        "missing_span_count": len(missing_ranges),
        "missing_spans_truncated": len(missing_ranges) > max_missing_spans,
        "missing_spans": missing_spans,
        "tail_words_checked": tail_n,
        "tail_coverage": tail_cov,
        "tail_missing_words": tail_missing,
        "trailing_missing_words": trailing_missing_words,
    }


def main() -> int:
    args = parse_args()
    text = source_text(args)
    transcript, asr_meta = load_transcript(args)
    cov = coverage(text, transcript, max_missing_spans=args.max_missing_spans, tail_words=args.tail_words)
    result = {
        "schema_version": 1,
        "audio": str(args.audio.resolve()),
        "audio_sha256": sha256_file(args.audio) if args.audio.exists() else None,
        "text_file": str(args.text_file.resolve()) if args.text_file else None,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "asr": asr_meta,
        "transcript": normalize_spaces(transcript),
        "coverage": cov,
        "thresholds": {
            "min_ordered_coverage": args.min_ordered_coverage,
            "min_tail_coverage": args.min_tail_coverage,
        },
        "ok": cov["ordered_coverage"] >= args.min_ordered_coverage and cov["tail_coverage"] >= args.min_tail_coverage,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"audio: {result['audio']}")
    print(f"ordered_coverage: {cov['ordered_coverage']:.3f}  unique_word_recall: {cov['unique_word_recall']:.3f}")
    print(f"missing_words: {cov['missing_word_count']} in {cov['missing_span_count']} span(s)")
    print(f"tail_coverage({cov['tail_words_checked']} words): {cov['tail_coverage']:.3f}  trailing_missing_words: {cov['trailing_missing_words']}")
    for span in cov["missing_spans"][:8]:
        print(f"missing[{span['word_start']}:{span['word_end_exclusive']}]: {span['text']}")
    if args.print_transcript:
        print("\ntranscript:")
        print(result["transcript"])
    if not result["ok"]:
        print("FAIL: coverage below threshold", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
