# Maintainer's guide — vibevoice.cpp / KugelAudio port

A pragmatic guide for whoever is poking at this repo next. Concise on
purpose; the README covers what users see.

## What this project is

This repo started as a C++/ggml port of Microsoft VibeVoice. Its current
purpose for forking is narrower:

- use `vibevoice.cpp` as the **ggml migration base**
- make **KugelAudio** the primary TTS target
- treat the canonical PyTorch implementation at **`../kugelaudio-open`** as
  the behavioral ground truth for TTS

Current v1 target:
- checkpoint: **`kugelaudio/kugelaudio-0-open`**
- mode: **single-speaker TTS only**
- conditioning: **raw reference audio only**
- interface: **CLI only** for acceptance purposes

Out of v1 scope:
- named / pre-encoded voices
- pre-encoded `voice_cache`
- language hints
- multi-speaker dialog
- long-text chunking
- watermarking

## Reference impls we trust

For KugelAudio TTS work, trust these in order:

1. **`../kugelaudio-open`** — single source of truth for prompt format,
   preprocessing, token behavior, CFG behavior, speech-end handling, and
   expected checkpoint/config shape.
2. **This repo's existing VibeVoice codepaths** — useful as implementation
   scaffolding only. Reuse aggressively, but do not let them override the
   KugelAudio reference behavior.
3. **Upstream Microsoft VibeVoice / mlx / transformers ports** — only for
   understanding inherited architecture pieces or debugging low-level math.

Rule of thumb:
- **behavioral disputes** -> `../kugelaudio-open` wins
- **implementation reuse choices** -> prefer the smallest safe diff from the
  current C++ code

## Current reality of this repo

What is real and relevant:
- `src/vibevoice_tts.cpp` contains the TTS orchestration logic we are adapting.
- `src/vibevoice_asr.cpp` is reused for closed-loop evaluation and shared
  speech/encoder helpers.
- `src/vibevoice_speech_helpers.hpp` already contains shared pieces reused by
  the 1.5B-style path.
- `scripts/eval_kugelaudio_divergence.py` is the current acceptance/eval
  orchestrator for canonical-vs-ggml comparison.
- `tests/fixtures/kugelaudio_eval_config.json` is the acceptance fixture/config
  entrypoint.
- `docs/kugelaudio-parity.md` is the maintainer-facing parity/eval note that
  reflects current Slice 3–5 reality.
- `include/vibevoice_capi.h` is the more real embedding surface than
  `include/vibevoice.h`, but **neither is a v1 acceptance target**.
- `src/vibevoice.cpp` / `include/vibevoice.h` are not the main product surface
  for this KugelAudio milestone.

What to optimize for:
- converter correctness
- loader/schema clarity
- prompt/inference parity with `../kugelaudio-open`
- deterministic eval runs
- closed-loop regression against canonical PyTorch

## Start here for the KugelAudio v1 acceptance path

If you are picking up this repo fresh, do **not** start from the old VibeVoice
quickstarts or the legacy voice-cache flow.

Read/use these first:
- `prd.md` — current execution plan and acceptance checklist
- `project-memory.md` — implementation decisions already taken during the
  KugelAudio migration
- `docs/conversion.md` — converter + GGUF contract, including KugelAudio-only
  schema expectations
- `docs/kugelaudio-parity.md` — parity notes, eval fixture shape, closed-loop
  ASR assumptions, logging contract, and acceptance-path caveats
- `scripts/eval_kugelaudio_divergence.py` — canonical-vs-ggml eval harness
- `tests/fixtures/kugelaudio_eval_config.json` — acceptance fixture/config

The current v1 acceptance workflow is:
1. convert `kugelaudio/kugelaudio-0-open`
2. optionally quantize to `q8_0`
3. run the canonical-vs-ggml divergence harness
4. inspect `results.json` + per-step logs
5. enforce closed-loop ASR thresholds on the supported fixture

If a doc/example conflicts with the above flow, treat it as legacy unless it
explicitly says it is part of the KugelAudio v1 acceptance path.

## Layout

```
include/
  vibevoice.h              # older public C API; not a v1 acceptance target
  vibevoice_capi.h         # flatter C ABI; more real than vibevoice.h
src/
  vibevoice.cpp            # older public-API impl / shim
  vibevoice_capi.cpp       # flat C ABI impl
  vibevoice_tts.{hpp,cpp}  # main KugelAudio TTS adaptation target
  vibevoice_asr.{hpp,cpp}  # ASR + closed-loop support + shared helpers
  vibevoice_speech_helpers.hpp
  qwen2.{hpp,cpp}
  acoustic_tokenizer.{hpp,cpp}
  diffusion_head.{hpp,cpp}
  dpm_solver.{hpp,cpp}
  conv1d.{hpp,cpp}
  model_loader.{hpp,cpp}
  tokenizer.{hpp,cpp}
  audio_io.{hpp,cpp}
scripts/
  convert_tokenizer.py
  convert_vibevoice_to_gguf.py
  convert_voice_to_gguf.py      # legacy VibeVoice path; not a KugelAudio v1 goal
  quantize_gguf.py
  eval_kugelaudio_divergence.py # acceptance/eval harness
tests/
  fixtures/kugelaudio_eval_config.json
  test_kugelaudio_*.{cpp,py}    # acceptance-path tests are concentrated here
docs/
  conversion.md
  kugelaudio-parity.md
third_party/ggml
../kugelaudio-open/
  src/kugelaudio_open/...       # canonical TTS behavior
```

## Build

```bash
git clone --recursive <repo> && cd vibevoice.cpp
cmake -B build -DVIBEVOICE_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Useful CMake options:
- `VIBEVOICE_BUILD_TESTS`
- `VIBEVOICE_TEST_LARGE`
- `VIBEVOICE_BUILD_EXAMPLES`
- `VIBEVOICE_GGML_CUDA`
- `VIBEVOICE_GGML_METAL`
- `VIBEVOICE_GGML_VULKAN`

## Acceptance focus

For the KugelAudio milestone, "works" is not enough.

The v1 acceptance shape is:
- convert `kugelaudio/kugelaudio-0-open`
- load it in the C++ runtime
- generate a reproducible single-speaker sample
- generate a reproducible raw-reference voice-cloned sample
- compare against `../kugelaudio-open` with the **same**:
  - checkpoint
  - prompt
  - reference audio
  - seed
  - generation settings
- require `f16` closed-loop ASR recall to reach at least:
  - **95% of canonical PyTorch recall**, and
  - **0.80 absolute recall floor**
- require `q8_0` to run end-to-end

## Tests at a glance

The legacy VibeVoice-oriented tests are still useful, but for KugelAudio work
prioritize tests that validate:
- converter schema detection
- prompt formatting parity
- single-speaker raw-reference generation
- closed-loop regression against canonical PyTorch
- deterministic seeded behavior on CPU

When adding new tests, prefer making the acceptance path obvious rather than
expanding generic coverage.

## Env vars / model-dependent testing

Existing tests use env vars such as:

```
VIBEVOICE_MODEL
VIBEVOICE_TTS_MODEL
VIBEVOICE_ASR_MODEL
VIBEVOICE_VOICE
VIBEVOICE_TOKENIZER
VIBEVOICE_CLI
VIBEVOICE_TTS_15B_MODEL
VIBEVOICE_REF_WAV
```

For KugelAudio-focused tests, prefer introducing clearly named variables if
needed instead of overloading old VibeVoice names even further.

## Naming + conventions

### GGUF schema

Do **not** assume the long-term KugelAudio GGUF schema must match legacy
`vibevoice.*` metadata names.

Current rule:
- it is acceptable to support both legacy and KugelAudio-specific metadata
  during migration
- the converter and loader must fail clearly for unsupported checkpoints
- the metadata contract must be explicit, not inferred by accident

### Prompt semantics

For v1 TTS behavior, the canonical prompt shape comes from
`../kugelaudio-open/src/kugelaudio_open/processors/kugelaudio_processor.py`.
Keep these sections aligned exactly where practical:
- system prompt
- `Voice input:`
- `Text input:`
- `Speech output:`

Do not quietly preserve old VibeVoice prompt quirks if they diverge from the
KugelAudio reference.

### Reference-audio preprocessing

Match canonical KugelAudio behavior:
- accept whatever the current repo loader supports
- resample internally to **24 kHz mono**
- apply the same RMS / loudness normalization convention as the canonical
  implementation
- support **single reference input only** in v1

## Gotchas / migration notes

1. **Current converter variant detection is VibeVoice-biased.**
   `scripts/convert_vibevoice_to_gguf.py` was built around VibeVoice family
   assumptions. Re-check every architecture/config heuristic against
   `../kugelaudio-open` before trusting it.

2. **Prompt parity matters more than it looks.**
   Small prompt-format drift can look like a "model quality" issue when it is
   really a processor mismatch.

3. **Do not drop semantic conditioning.**
   For this KugelAudio milestone, semantic conditioning is required if the
   canonical path uses it for acceptable voice cloning.

4. **Behavior beats reuse when they conflict.**
   Reuse as much of the existing C++ as possible, but if a reused path diverges
   from `../kugelaudio-open`, fix the path.

5. **Seed everything for comparisons.**
   TTS evals are noisy without pinned seeds. Divergence testing should never be
   run unseeded.

6. **Keep CPU determinism in mind.**
   The spec now requires deterministic CPU behavior for fixed seed/settings.
   Treat nondeterminism as a bug unless clearly documented.

7. **The old voice gguf path is legacy.**
   `convert_voice_to_gguf.py` and related voice-gguf flow are not part of the
   KugelAudio v1 target.

8. **q8_0 is an execution target, not a parity target.**
   For v1, `f16` carries the parity/quality burden. `q8_0` only needs to make
   the defined eval path run end-to-end.

## Adding a new test

```bash
cp tests/test_smoke.cpp tests/test_my_thing.cpp
# edit it, return 0 on pass, 77 to mean "skipped"
echo 'vv_add_test(test_my_thing)' >> tests/CMakeLists.txt
```

For KugelAudio work, prefer tests that:
- compare prompt/token behavior to `../kugelaudio-open`
- exercise the exact supported checkpoint
- shell out to the CLI or eval harness for end-to-end acceptance paths
- keep fixtures and seeds fixed
- make it obvious whether a test belongs to the KugelAudio v1 acceptance path
  versus legacy VibeVoice regression coverage

## Converter workflow

Treat the converter as a first-class part of the product, not a throwaway.

Typical flow:

```
HF checkpoint/config -> scripts/convert_vibevoice_to_gguf.py -> .gguf
                                                         ↓
                                               scripts/quantize_gguf.py
                                                         ↓
                                                f16 / q8_0 gguf
                                                         ↓
                                  scripts/eval_kugelaudio_divergence.py
                                                         ↓
                                         results.json + per-step logs
```

Guidelines:
- run strict mapping checks when possible
- document every schema assumption
- reject unsupported checkpoints clearly
- keep the output metadata contract explicit

## Useful third-party references

- `../kugelaudio-open` — canonical KugelAudio behavior
- `microsoft/VibeVoice` — architecture ancestry / legacy comparison
- `Blaizzy/mlx-audio` — extra non-PyTorch reference for inherited pieces
- `huggingface/transformers` VibeVoice-related code — secondary reference only

## Style

- C++17
- no exceptions in the public API surface
- one translation unit per logical component
- keep shims thin
- don't add comments for the *what*, only for non-obvious *why*
- when changing behavior for KugelAudio parity, leave a short note pointing to
  the canonical file in `../kugelaudio-open`
