# kugelaudio.cpp

Fork maintained by **Maurice Sonnemann**, based on
[`vibevoice.cpp`](https://github.com/mudler/vibevoice.cpp) by Ettore Di Giacinto / LocalAI.

**Brought to you by the [LocalAI](https://github.com/mudler/LocalAI) team** - the creators of LocalAI, the open-source AI engine that runs any model - LLMs, vision, voice, image, video - on any hardware. No GPU required.

[![Models on HF](https://img.shields.io/badge/HuggingFace-Models-yellow)](https://huggingface.co/mudler/kugelaudio.cpp-models)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![LocalAI](https://img.shields.io/badge/LocalAI-Run_Locally-orange)](https://github.com/mudler/LocalAI)

A ggml-based C++ inference engine for **KugelAudio TTS**, forked from
[`vibevoice.cpp`](https://github.com/mudler/vibevoice.cpp) and renamed for the
KugelAudio-only direction.

> Published direction: this branch is **KugelAudio-only TTS**.
> Any remaining VibeVoice compatibility code should be treated as internal or
> transitional, not as part of the published operator workflow.

> Status: this fork's **v1 acceptance path is KugelAudio-first**:
> `kugelaudio/kugelaudio-0-open`, single-speaker TTS, exactly one raw
> reference WAV, CLI-only, canonical-vs-ggml divergence evaluation.

## Rename / compatibility note

The primary binary is now `kugelaudio-cli`, and new automation should prefer
`KUGELAUDIO_*` environment variables and `scripts/convert_kugelaudio_to_gguf.py`.
Deprecated `vibevoice-cli`, `VIBEVOICE_*` env vars, and
`scripts/convert_vibevoice_to_gguf.py` remain as compatibility aliases while the
fork finishes shedding old internal names. Remaining `vibevoice.*` GGUF metadata
references are read/write compatibility details, not the published product name.

## KugelAudio v1 acceptance path

If you are here for the **currently supported** path in this fork, use this
workflow.

### Supported v1 scope
- checkpoint: `kugelaudio/kugelaudio-0-open`
- interface: CLI-only
- TTS shape: single-speaker only
- conditioning: exactly one raw reference WAV
- parity target: canonical-vs-ggml comparison against `../kugelaudio-open`
- quality target: `f16` transcript recall >= 95% of canonical recall with
  absolute floor 0.80, plus speaker similarity >= 95% of canonical with
  absolute floor 0.60
- execution target: `q8_0` must run end-to-end on the same acceptance path

### Acceptance workflow
```bash
# 1) build
cmake -B build -DKUGELAUDIO_BUILD_TESTS=ON && cmake --build build -j

# 2) convert tokenizer + KugelAudio checkpoint
python scripts/convert_tokenizer.py --src models/qwen2.5/tokenizer.json --out models/tokenizer.gguf
python scripts/convert_kugelaudio_to_gguf.py \
  --src models/kugelaudio-0-open \
  --out models/kugelaudio-f16.gguf

# 3) (optional) quantize execution artifact
python scripts/quantize_gguf.py \
  --src models/kugelaudio-f16.gguf \
  --out models/kugelaudio-q8_0.gguf \
  --type q8_0

# 4) run the canonical-vs-ggml evaluation harness
#    external evaluators: faster-whisper + SpeechBrain ECAPA-TDNN
uv run scripts/eval_kugelaudio_divergence.py \
  --config tests/fixtures/kugelaudio_eval_config.json \
  --execute both
```

### Vulkan build and smoke run

For Vulkan, use a dedicated build directory and prefer the smaller `q8_0`
artifact first.

```bash
# build ggml + kugelaudio.cpp with Vulkan enabled
cmake -B build-vulkan \
  -DKUGELAUDIO_GGML_VULKAN=ON \
  -DKUGELAUDIO_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j

# first execution target on Vulkan: q8_0
python scripts/quantize_gguf.py \
  --src models/kugelaudio-f16.gguf \
  --out models/kugelaudio-q8_0.gguf \
  --type q8_0

# smoke test through the CLI
KUGELAUDIO_BACKEND=vulkan ./build-vulkan/bin/kugelaudio-cli \
  --model models/kugelaudio-q8_0.gguf \
  --tokenizer models/tokenizer.gguf \
  --ref-audio tests/fixtures/reference_sine.wav \
  --text "Hello world." \
  --out /tmp/kugelaudio-vulkan-smoke.wav \
  --seed 12345 --cfg 1.0 --steps 4 --max-frames 24

# optional ctest smoke (requires env vars)
KUGELAUDIO_Q8_MODEL=$PWD/models/kugelaudio-q8_0.gguf \
KUGELAUDIO_TOKENIZER=$PWD/models/tokenizer.gguf \
KUGELAUDIO_REF_WAV=$PWD/tests/fixtures/reference_sine.wav \
ctest --test-dir build-vulkan -R test_kugelaudio_vulkan_smoke --output-on-failure

# explicit hybrid fallback for longer Vulkan runs:
# keep generation on Vulkan, decode the final waveform on CPU
KUGELAUDIO_BACKEND=vulkan ./build-vulkan/bin/kugelaudio-cli \
  --model models/kugelaudio-q8_0.gguf \
  --tokenizer models/tokenizer.gguf \
  --ref-audio tests/fixtures/reference_sine.wav \
  --text "Hello world." \
  --out /tmp/kugelaudio-vulkan-hybrid.wav \
  --seed 12345 --max-frames 64 \
  --final-decoder-backend cpu

# explicit streamed Vulkan decoder:
# keep the final decode on Vulkan, but chunk it instead of one-shot decode
KUGELAUDIO_BACKEND=vulkan KUGELAUDIO_STREAM_DECODER_FRAMES=8 \
./build-vulkan/bin/kugelaudio-cli \
  --model models/kugelaudio-q8_0.gguf \
  --tokenizer models/tokenizer.gguf \
  --ref-audio tests/fixtures/reference_sine.wav \
  --text "Hello world." \
  --out /tmp/kugelaudio-vulkan-stream.wav \
  --seed 12345 --max-frames 64 \
  --final-decoder-backend stream
```

Notes:
- `KUGELAUDIO_BACKEND=vulkan` is the supported runtime selector.
- `--threads` is currently a CPU-only knob; on Vulkan it is ignored.
- The current Vulkan path is correctness-first; use `q8_0` before trying the
  larger `f16` artifact.
- `--final-decoder-backend auto` uses one-shot final decode on CPU and streamed
  final decode on GPU backends. Streaming avoids large one-shot CUDA/Vulkan
  decoder allocations while keeping final decode on the active GPU.
- `--final-decoder-backend cpu` is an explicit hybrid fallback for longer
  GPU runs on memory-constrained systems: conditioning / LM / diffusion stay on
  the selected GPU backend, and only the final latent->waveform acoustic decoder
  runs on CPU.
- `--final-decoder-backend stream` explicitly keeps the final decoder on the
  active backend and chunks the latent->waveform decode to reduce peak memory.
  Tune chunk size with `KUGELAUDIO_STREAM_DECODER_FRAMES` (default
  `8`).
- `--final-decoder-backend active` forces one-shot decode on the active backend;
  this is mainly a CUDA/Vulkan diagnostic mode because longer outputs can need
  very large temporary im2col buffers.
- Vulkan still uses smaller internal encoder chunks than CPU/CUDA; that behavior is
  handled internally by the runtime.

### Coherent long-prompt single-sequence mode

For the best current speaker-identity consistency on long prompts, use the
f16 single-sequence path with generated step embeddings rounded to f16 and a
streamed final decoder. This keeps one LM/KV-cache sequence instead of restarting
independent chunks, so it avoids chunk-boundary speaker resets. It is a
quality/diagnostic path rather than full-coverage longform: generation stops at
the model's natural speech-end token and may not read the entire input text.

```bash
KUGELAUDIO_BACKEND=cpu \
./build/bin/kugelaudio-cli \
  --model models/kugelaudio-f16.gguf \
  --tokenizer models/tokenizer.gguf \
  --ref-audio tests/fixtures/reference_sine.wav \
  --text-file long_article.txt \
  --out /tmp/kugelaudio-single-sequence-natural.wav \
  --max-words-per-chunk 80 \
  --max-frames 64 \
  --chunk-continuity single-sequence \
  --overlap-sentences 1 \
  --chunking-strategy heuristic \
  --pause-mode punctuation \
  --crossfade-ms 60 \
  --steps 20 \
  --cfg 1.0 \
  --seed 12345 \
  --final-decoder-backend stream \
  --verbose
```

Important knobs:
- `KUGELAUDIO_BACKEND=cpu` avoids f16 long-prompt prefill OOM on 24 GB GPUs.
- f16 KugelAudio models now round generated step embeddings to f16 before LM
  feedback by default. Set `KUGELAUDIO_CAST_STEP_EMBED_F16=0` only
  for parity diagnostics against the older unstable path.
- `single-sequence` now stops naturally by default. Set
  `KUGELAUDIO_SINGLE_SEQUENCE_MIN_RATIO=0.90` only to reproduce the
  old forced-continuation diagnostic; it can produce silence after the natural
  stop.
- `--final-decoder-backend stream` avoids the large one-shot CPU final-decoder
  allocation for long latent sequences.

Use chunking below when full text coverage matters more than maximum speaker
identity consistency.

### Long-form all-in-one runner

For practical listening/eval runs, prefer the Python orchestration helper. It
wraps the current best long-form CLI profile, writes a generation log and run
metadata JSON, and can optionally run the ASR coverage checker afterward.

```bash
uv run scripts/run_kugelaudio_longform_best.py \
  --model models/kugelaudio-f16.gguf \
  --tokenizer models/tokenizer.gguf \
  --ref-audio /path/to/real_reference.wav \
  --text-file long_article.txt \
  --out runs/longform-best/output.wav \
  --language de
```

Defaults reflect the best current real-reference diagnostic profile:
`cfg=2.0`, `steps=20`, `max_frames=320`, `overlap_sentences=0`, final-only
ellipsis padding, and `tail-reference` continuity with a 1200 ms tail. Because
`tail-reference` is still a retired/diagnostic continuity mode, the script sets
`KUGELAUDIO_ENABLE_RETIRED_CONTINUITY=1` only for that explicit profile.

Useful variants:

```bash
# Generate only; skip WhisperX/ASR QA.
uv run scripts/run_kugelaudio_longform_best.py \
  --ref-audio /path/to/real_reference.wav \
  --text-file long_article.txt \
  --out runs/longform-best/output.wav \
  --no-asr

# Conservative default chunking, without waveform-tail feedback.
uv run scripts/run_kugelaudio_longform_best.py \
  --ref-audio /path/to/real_reference.wav \
  --text-file long_article.txt \
  --out runs/longform-best/output-none.wav \
  --chunk-continuity none \
  --no-asr

# Print the exact CLI/ASR commands without running them.
uv run scripts/run_kugelaudio_longform_best.py \
  --ref-audio /path/to/real_reference.wav \
  --text-file long_article.txt \
  --out runs/longform-best/output.wav \
  --dry-run
```

Outputs next to `--out` by default:
- `<out>.log` — full `kugelaudio-cli` stdout/stderr
- `<out>.run.json` — resolved profile, commands, return codes, elapsed time
- `<out>.asr.json` — optional ASR coverage report from
  `scripts/verify_longform_asr.py`

The ASR pass shells out through `uvx whisperx`; use `--no-asr` if WhisperX is
not installed or if you only need a listening sample. ASR coverage is useful for
omitted/cut-off text regressions, but it is not sufficient to detect every final
phoneme cutoff.

### Large natural blocks for fuller long-form coverage

The best current full-text compromise is to keep `--chunk-continuity none`, but
use fewer, larger chunks so speaker identity is re-sampled less often. This does
not carry generated audio, latents, or KV state across chunk boundaries; each
block still stops naturally within its frame budget and is stitched afterward.

```bash
KUGELAUDIO_BACKEND=cuda \
./build-cuda/bin/kugelaudio-cli \
  --model models/kugelaudio-f16.gguf \
  --tokenizer models/tokenizer.gguf \
  --ref-audio tests/fixtures/reference_sine.wav \
  --text-file long_article.txt \
  --out /tmp/kugelaudio-large-natural-blocks.wav \
  --max-words-per-chunk 224 \
  --max-frames 192 \
  --chunk-continuity none \
  --overlap-sentences 1 \
  --chunking-strategy heuristic \
  --pause-mode punctuation \
  --crossfade-ms 60 \
  --steps 20 \
  --cfg 1.0 \
  --seed 12345 \
  --final-decoder-backend auto \
  --verbose
```

Use this when you need more coverage than natural single-sequence provides. In
current listening/debug runs, `224` words / `192` frames produced three blocks
on the long-form fixture; larger two-block settings can stop too early or
compress coverage, so treat them as per-text tuning rather than defaults.

### Optional long-text chunking

Long-text chunking is available as an optional CLI path outside the strict v1
acceptance workflow. The runtime follows the canonical heuristic planner shape:

- sentence boundaries first
- clause fallback for oversized sentences
- hard-wrap by words as a last resort
- raw-reference conditioning cached/reused for normal chunked generation
- clean independent chunks by default (`--chunk-continuity none`)
- stitched output with configurable pause insertion and crossfade

```bash
./build/bin/kugelaudio-cli \
  --model models/kugelaudio-f16.gguf \
  --tokenizer models/tokenizer.gguf \
  --ref-audio tests/fixtures/reference_sine.wav \
  --text-file long_article.txt \
  --max-words-per-chunk 120 \
  --overlap-sentences 1 \
  --chunking-strategy syntax-aware \
  --pause-mode punctuation \
  --crossfade-ms 60 \
  --chunk-continuity none \
  --out /tmp/kugelaudio-long.wav
```

Notes:
- `--max-words-per-chunk <= 0` disables chunking.
- Short inputs bypass stitching automatically.
- `--overlap-sentences N` reuses the last `N` completed sentences from the
  previous chunk as prompt context for the next chunk.
- `--chunking-strategy heuristic|syntax-aware` selects between the canonical
  baseline planner and a lightweight syntax-aware variant that additionally
  prefers conjunction-aware phrase breaks for oversized sentences.
- `--chunk-continuity none` is the recommended quality mode: every chunk uses
  the same cached raw-reference conditioning and no generated audio/state is fed
  forward.
- `--chunk-continuity single-sequence` is an experimental true model-state
  continuity baseline: it ignores chunk stitching and generates the full text in
  one LM/KV-cache sequence with a total frame budget of
  `max_frames * planned_chunks`. This avoids independent chunk restarts and, by
  default, stops at the natural speech-end token rather than forcing full-budget
  coverage.
- Retired experiment: `clean-tail-reference` appended a short sanitized voiced
  island from generated audio to the original reference. It still caused noise
  and speaker drift, so generated waveform feedback remains unsafe.
- Retired experiment: `prompt-instruction` added text-only continuity guidance.
  It avoided noise but still changed speaker identity between chunks, so it is
  not recommended for quality-critical output.
- Retired experiment: `tail-reference` continuity conditioned chunk N+1 on the
  original raw reference plus a decoded voiced tail from chunk N. It reduced
  speaker-identity jumps, but listening showed decoded waveform noise was fed
  back and amplified from chunk 2 onward. Do not reintroduce waveform-tail
  reference conditioning without explicit denoising/gating evidence.
- Retired experiment: `latent-prefix` fed prior generated latent frames into the
  next chunk's LM state. Listening showed worse noise and intonation drift, so
  it is not exposed by the CLI.
- No heavy NLP dependency is added in this repo; sentence splitting still uses
  the existing punctuation heuristic, while `syntax-aware` mainly affects the
  oversized-sentence fallback path.

Primary docs for this path:
- `docs/conversion.md` — converter + GGUF contract
- `docs/kugelaudio-parity.md` — parity notes, acceptance fixture, eval/logging contract
- `AGENTS.md` — maintainer workflow / acceptance path orientation

### Canonical reference points into `../kugelaudio-open`
When behavior is ambiguous, these are the first files to check in the canonical
PyTorch implementation:
- prompt formatting + section semantics:
  `../kugelaudio-open/src/kugelaudio_open/processors/kugelaudio_processor.py`
- inference loop behavior (CFG, speech tokens, stop behavior):
  `../kugelaudio-open/src/kugelaudio_open/models/kugelaudio_inference.py`
- generation helpers used by the canonical path:
  `../kugelaudio-open/src/kugelaudio_open/utils/generation.py`
- reference-audio preprocessing / normalization:
  `../kugelaudio-open/src/kugelaudio_open/processors/audio_processor.py`
- published config/model shape assumptions:
  `../kugelaudio-open/src/kugelaudio_open/configs/kugelaudio_1.5b.json`
  and `../kugelaudio-open/src/kugelaudio_open/configs/model_config.py`
- CLI/reference workflow in the canonical repo:
  `../kugelaudio-open/src/kugelaudio_open/cli.py`

## Removed legacy operator workflows

The published CLI/docs surface for this branch is now **KugelAudio TTS only**.
The following older operator workflows have been removed from this README:

- pre-baked `voice.gguf` conditioning
- legacy raw-reference VibeVoice quickstarts
- ASR quickstarts and ASR benchmark guidance

Some compatibility code may still exist internally, but those paths are not the
supported publish surface for this branch. Fresh users should follow only the
KugelAudio acceptance workflow above.

## Tests

```bash
ctest --test-dir build --output-on-failure
```

For supported real-weight KugelAudio tests:

```bash
KUGELAUDIO_MODEL=models/kugelaudio-f16.gguf \
KUGELAUDIO_TOKENIZER=models/tokenizer.gguf \
KUGELAUDIO_REF_WAV=tests/fixtures/reference_sine.wav \
KUGELAUDIO_CLI=$PWD/build/bin/kugelaudio-cli \
ctest --test-dir build --output-on-failure -j 2
```

For the chunked CLI real-weight coverage specifically:

```bash
KUGELAUDIO_MODEL=models/kugelaudio-f16.gguf \
KUGELAUDIO_TOKENIZER=models/tokenizer.gguf \
KUGELAUDIO_REF_WAV=tests/fixtures/reference_sine.wav \
KUGELAUDIO_CLI=$PWD/build/bin/kugelaudio-cli \
ctest --test-dir build -R 'test_kugelaudio_cli_chunking_e2e|test_kugelaudio_cli_chunking_seed' --output-on-failure
```

## Embedding from Go (purego)

`kugelaudio.cpp` ships a flat C ABI in [`include/vibevoice_capi.h`](include/vibevoice_capi.h)
designed for `dlopen` / `purego.RegisterLibFunc` consumers.

> Note: for this branch's published surface, treat the C ABI as
> **compatibility / embedding infrastructure**, not the primary acceptance
> surface. The supported published workflow remains the KugelAudio TTS CLI.

The TTS entrypoints are:

```c
int  vv_capi_load(const char* tts_model, const char* asr_model,
                  const char* tokenizer, const char* voice, int n_threads);
int  vv_capi_tts(const char* text, const char* voice_path,
                 const char* const* ref_audio_paths, int n_ref_audio_paths,
                 const char* dst_wav, int steps, float cfg,
                 int max_speech_frames, uint32_t seed);
void vv_capi_unload(void);
```

Build the shared library and call it from Go:

```go
import "github.com/ebitengine/purego"

var (
    Load   func(tts, asr, tok, voice string, threads int) int
    // Model the TTS function with the exact pointer shape from
    // include/vibevoice_capi.h in your binding layer.
    TTS    any
    Unload func()
)

lib, _ := purego.Dlopen("./libkugelaudio.so", purego.RTLD_NOW|purego.RTLD_GLOBAL)
purego.RegisterLibFunc(&Load,   lib, "vv_capi_load")
purego.RegisterLibFunc(&TTS,    lib, "vv_capi_tts")
purego.RegisterLibFunc(&Unload, lib, "vv_capi_unload")
```

Build the shared library with `cmake -DKUGELAUDIO_SHARED=ON`.

## Why

KugelAudio's canonical open runtime is Python/PyTorch. `kugelaudio.cpp` provides:

- A native CPU runtime with no Python at inference time
- Free CUDA / Metal / Vulkan support via ggml backends
- A single `.gguf` weight file + a single binary
- A flat C ABI (`include/vibevoice_capi.h`) for embedding via dlopen / purego / cgo
- A KugelAudio-first published workflow centered on raw-reference TTS

## Build

```bash
git clone --recursive https://example.com/kugelaudio.cpp
cd kugelaudio.cpp
cmake -B build -DKUGELAUDIO_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Layout

See [`docs/`](docs/) and the project plan for the full architecture.

## Author

Ettore Di Giacinto ([@mudler](https://github.com/mudler)) - also the
maintainer of [LocalAI](https://github.com/mudler/LocalAI). PRs welcome.

## License

MIT - see [LICENSE](LICENSE). Copyright © 2026 Ettore Di Giacinto.
The model weights remain under their upstream license
([kugelaudio/kugelaudio-0-open](https://huggingface.co/kugelaudio/kugelaudio-0-open)).
