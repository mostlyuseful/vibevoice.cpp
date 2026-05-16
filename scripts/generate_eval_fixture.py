#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
"""Generate a reproducible synthetic reference WAV fixture for KugelAudio evaluations.

Properties designed to be stable for regression testing:
- 24 kHz sample rate, mono, 16-bit PCM
- 3.0 seconds duration
- 440 Hz sine tone, -18 dBFS peak amplitude
- Deterministic: same script version always produces the same byte-for-byte WAV
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path


def generate_fixture(output_path: Path, freq: float = 440.0, duration_sec: float = 3.0,
                     sr: int = 24000, peak_dbfs: float = -18.0) -> None:
    """Write a reproducible monophonic sine-wave WAV."""
    n_samples = int(round(sr * duration_sec))
    amplitude = 10.0 ** (peak_dbfs / 20.0)
    max_int16 = 32767.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for i in range(n_samples):
            sample = math.sin(2.0 * math.pi * freq * (i / sr)) * amplitude
            # clamp before cast to avoid wraparound
            val = int(round(max(-1.0, min(1.0, sample)) * max_int16))
            wf.writeframes(struct.pack("<h", val))


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    out = repo / "tests" / "fixtures" / "reference_sine.wav"
    generate_fixture(out)
    print(f"Wrote fixture: {out}")
    print(f"  {out.stat().st_size} bytes, 24 kHz mono 16-bit PCM, 3.0 s, 440 Hz @ -18 dBFS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
