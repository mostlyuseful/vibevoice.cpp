# KugelAudio parity notes

This note records the Slice 3 parity status for the supported v1 path
(`kugelaudio/kugelaudio-0-open`) and, in particular, where existing ggml
components are intentionally reused versus where small temporary divergences
still exist.

Canonical reference:
- `../kugelaudio-open/src/kugelaudio_open/models/kugelaudio_inference.py`

## Reused components that remain behaviorally valid

### DPM solver
- C++ path: `src/dpm_solver.{hpp,cpp}` via `dpm_solver_sample(...)`
- Canonical behavior match:
  - v-prediction sampler
  - `num_train_timesteps = 1000`
  - order-2 multistep solver
  - `lower_order_final = true`
- Rationale:
  - this is a scheduler/math reuse, not a KugelAudio-specific semantic layer
  - the existing ggml implementation already mirrors the canonical scheduler
    contract and is covered by `tests/test_dpm_solver.cpp`

### Acoustic connector for generated speech latents
- C++ path: `src/vibevoice_tts.cpp` -> `run_speech_connector(...)`
- Canonical behavior match:
  - after diffusion, the canonical path maps generated speech latents to the
    next LM-step embedding through the acoustic connector only
  - no semantic re-encoding is performed for generated speech latents
- Rationale:
  - reusing the existing acoustic connector is semantically aligned with the
    canonical generation loop, so no KugelAudio-only fork was introduced
  - covered by `tests/test_kugelaudio_reused_components.cpp`

### Final acoustic decoder pass
- C++ path: `src/vibevoice_tts.cpp` -> `decode_latent_sequence(...)`
- Canonical behavior match:
  - latent chunks are accumulated across generation
  - final waveform decode runs once on the full latent sequence so decoder tail
    context is preserved
- Rationale:
  - the existing non-streaming decoder path matches the canonical behavior more
    closely than frame-by-frame decode would
  - decoder parity is covered by `tests/test_acoustic.cpp`
  - supported-path smoke coverage lives in `tests/test_kugelaudio_decode_smoke.cpp`

## Temporary explicit divergences for the supported v1 path

These are intentional for the current milestone and should be treated as
tracked parity debt, not accidental behavior.

### Hardcoded canonical speech token IDs
- Current C++ behavior:
  - uses the supported checkpoint's known canonical IDs directly:
    - `speech_start = 151652`
    - `speech_end = 151653`
    - `speech_diffusion = 151654`
    - `eos = 151643`
- Why this is acceptable for now:
  - v1 supports exactly one published checkpoint
  - the current C++ load path does not yet source a dedicated EOS/token contract
    from tokenizer/model metadata
- Follow-up:
  - move these from code constants to explicit runtime metadata once the loader
    exposes the tokenizer-side contract cleanly

### Hardcoded `speech_end_penalty = 1.5`
- Current C++ behavior:
  - applies the canonical default penalty before constrained speech-token
    selection
- Why this is acceptable for now:
  - it matches the canonical default behavior for the supported checkpoint
  - no wider public API/CLI tuning surface is required for v1 acceptance
- Follow-up:
  - expose this as a documented runtime setting only if parity/eval needs it

### Control-token guard (`32` non-audio steps)
- Current C++ behavior:
  - fails clearly if the KugelAudio loop emits more than 32 consecutive control
    tokens without producing a speech frame
- Why this diverges:
  - the current public API is frame-budget based rather than total-token-budget
    based, so an explicit guard prevents a hung run on bad model state
- Why this is acceptable for now:
  - it is a safety guard, not a target behavior, and does not affect normal
    supported-path generation
- Follow-up:
  - revisit once the generation interface has a cleaner token-budget notion

## Determinism status for regression use

### Deterministic configuration we currently promise
For the supported v1 KugelAudio path, the regression target is:
- `KUGELAUDIO_BACKEND=cpu`
- fixed model + tokenizer + reference WAV + text
- fixed generation settings
- explicit non-zero seed

This is the configuration exercised by:
- `tests/test_kugelaudio_determinism.cpp`
- `tests/test_kugelaudio_cli_seed.cpp`

### Known non-guaranteed / potentially nondeterministic cases
These are not current regression promises and should be treated as such:

- **Omitted seed (`seed = 0` / no `--seed`)**
  - the runtime falls back to `std::random_device()` seeding
  - result: different runs are expected to diverge

- **Backend auto-selection or non-CPU backends**
  - backend selection is process-wide and lazy; without `KUGELAUDIO_BACKEND=cpu`, the runtime may pick a GPU-class backend first
  - even with the same seed, cross-backend equality is not currently a regression guarantee

- **GPU / accelerator execution details**
  - CUDA / Metal / Vulkan / hipBLAS execution may differ from CPU due to backend-specific kernels, scheduling, and transfer/fallback behavior
  - these paths are correctness targets, not current determinism targets

- **Changing low-level execution knobs between runs**
  - changing backend choice, flash-attention enablement, build flags, quantization, or model artifact naturally falls outside the deterministic regression contract

In short: for acceptance/regression comparisons today, force CPU first and keep the full input + settings tuple pinned.

## Evaluation fixtures and harness setup

### Voice-cloned acceptance sample
- Name: `kugelaudio-0-open-hello-sine` (defined in `tests/fixtures/kugelaudio_eval_config.json` under `voice_cloned_sample`)
- Input tuple:
  - Text: `Hello world.`
  - Reference audio: `tests/fixtures/reference_sine.wav`
  - Seed: `12345`
  - CFG scale: `1.0`, steps: `8`, max_frames: `32`
- Ground truth: the `canonical.wav` output produced by the canonical PyTorch run with the above tuple
- Usage: when executing the harness with `--execute both`, the `results.json` records the SHA256 of the canonical output. Future ggml runs should produce an output whose transcript-recall and speaker-similarity metrics meet the acceptance threshold relative to this ground truth.

### How to rerun the fixture setup from a clean checkout
1. Ensure `../kugelaudio-open` is checked out alongside this repo (the harness uses `../../../kugelaudio-open` as the canonical repo)
2. (Re-)generate the reference audio fixture if needed: `uv run scripts/generate_eval_fixture.py`
3. Verify fixture wiring without model artifacts: `uv run tests/test_kugelaudio_eval_fixture.py`
4. Verify plan plumbing: `uv run tests/test_kugelaudio_eval_plan.py`
5. To actually execute the canonical-vs-ggml comparison, first convert a KugelAudio checkpoint to GGUF (see `docs/conversion.md`), then run:
   ```bash
   uv run scripts/eval_kugelaudio_divergence.py \
     --config tests/fixtures/kugelaudio_eval_config.json \
     --execute both
   ```
   This writes `plan.json` and `results.json` into the config-defined `output_dir`.

### Reference audio fixture
- File: `tests/fixtures/reference_sine.wav`
- Properties: 24 kHz, mono, 16-bit PCM, 3.0 s, 440 Hz sine @ -18 dBFS
- Generator: `uv run scripts/generate_eval_fixture.py`
- Config wiring: `tests/fixtures/kugelaudio_eval_config.json` points to it with `"reference_audio": "reference_sine.wav"` (relative to fixtures dir)
- Purpose: acceptance baseline; any downstream deterministic generation expected to produce the same full output vector for the same seed/settings/model

### Canonical-vs-ggml harness
- Script: `uv run scripts/eval_kugelaudio_divergence.py --config tests/fixtures/kugelaudio_eval_config.json --execute <none|canonical|ggml|both>`
- Backend selection: use config field `"ggml_backend"` or override with `--ggml-backend cpu|cuda|vulkan`
- Plan mode: `./scripts/eval_kugelaudio_divergence.py --config ... --execute none`
  - Does not require model artifacts; validate config plumbing and command shapes only
- Execution mode: `./scripts/eval_kugelaudio_divergence.py --config ... --execute both --write-plan /tmp/plan.json`
  - Runs TTS (canonical + ggml), then evaluates both outputs with external evaluators
  - Writes `plan.json` and `results.json` (return codes, output hashes, transcripts, transcript recall, speaker similarity) to the config-defined `output_dir`
  - Canonical side runs inline Python in the `../kugelaudio-open` checkout
  - ggml side runs `kugelaudio-cli ...` with `KUGELAUDIO_BACKEND` controlled by the plan/config (`cpu` by default, overrideable to `cuda` or `vulkan`)
  - Transcript-fidelity side uses `faster-whisper`
  - Speaker-similarity side uses SpeechBrain ECAPA-TDNN cosine similarity
- Acceptance metrics:
  - transcript recall against the source text: ggml must be >= 95% of canonical recall, with absolute floor 0.80
  - speaker similarity against the reference WAV: ggml must be >= 95% of canonical similarity, with absolute floor 0.60
- q8_0 execution path: `KUGELAUDIO_Q8_MODEL` env var points to a q8_0 quantized model
  - Same generation parameters and reference fixture
  - Validated by `tests/test_kugelaudio_q8_0_smoke.cpp` (non-empty, finite, non-silent output)
- Failure diagnostics: the harness produces separate logs for every step (TTS plus transcription/speaker-similarity JSON outputs for both canonical and ggml), plus `results.json` with structured status, return codes, SHA256s, transcripts, transcript recall, and speaker similarity values. When the threshold fails, `threshold_check.message` states exactly which metric and boundary was crossed (ratio or floor).
- Logging contract:
  - Converter: `convert: mode=kugelaudio checkpoint=... src=...` shows the mode and source
  - CLI TTS: `tts: model_family=... internal_variant=... quantization_hint=...` + `tts: generation_settings frames=... steps=... cfg=... seed=... conditioning=... ref_count=... chunk_continuity=... continuity_tail_ms=... chunk_boundary_cleanup=... final_decoder=... step_embed_f16=...`
  - Eval harness: startup log with `eval: config=... execute=... text_summary=chars=... sha256=... ref_summary=present=... sha256=... generation seed=... cfg=... steps=... max_frames=... canonical_model=... ggml_model=... ggml_backend=...`
  - Runtime backend init: `backend: requested=... device_index=... selected=... reason=...`, `backend: available_devices=...`, and `backend: flash-attn=...`
  - By default, raw prompt text and raw audio paths are not emitted; logs use length/hash summaries instead
- Quantization: use `scripts/quantize_gguf.py --src f16.gguf --out q8_0.gguf --type q8_0` to produce the q8_0 artifact from a converted f16 model

## Long-form chunking listening status

Current quality recommendation: use clean independent chunks with cached raw-reference conditioning:

```bash
--chunk-continuity none
```

Current opt-in experiment:
- `single-sequence` is a true model-state baseline: the full text is generated
  in one LM/KV-cache sequence using `max_frames * planned_chunks` as the total
  frame budget. It avoids independent chunk restarts, but may expose
  long-sequence autoregressive drift.

Retired continuity experiments:
- `tail-reference` made seams cohesive but fed decoded waveform artifacts back through the reference encoder, amplifying noise from chunk 2 onward in earlier synthetic/weak-reference trials. A later limited private real-reference run sounded very good and preferred `tail-reference`, so this remains worth stress-testing as an explicit diagnostic rather than promoting as a default.
- `clean-tail-reference` sanitized and RMS-matched a short generated voiced island before appending it to the original reference, but earlier trials still caused noise and speaker drift. In the limited real-reference revisit it also sounded good, though not preferred over plain `tail-reference`.
- `latent-prefix` avoided waveform feedback but listening showed worse noise and intonation drift; the real-reference revisit still immediately stopped chunk 2 after applying the latent prefix, so it remains structurally suspect.
- `prompt-instruction` avoided noise but still changed speaker identity between chunks, so it is not recommended for quality-critical output.

For long-form iteration, prefer clean independent chunks by default, or the `single-sequence` model-state baseline as a coherence oracle. Generated waveform/latent feedback modes should stay env-gated diagnostics until they survive longer real-reference stress tests. First real-reference stress results reopen `tail-reference` as the most promising diagnostic: with a private real reference, German Saint-Malo text, f16 CUDA, `cfg=2.0`, `max_frames=320`, and `overlap_sentences=0`, `tail-reference` at `continuity_tail_ms=1200` beat `none` by ASR coverage on both `mw60`/3 chunks and `mw40`/4 chunks while maintaining stable RMS. Keep this as an experiment, not default parity behavior.

Opt-in final-cut diagnostic: `--text-end-padding ellipsis` normalizes the generated text end to terminal ASCII `...`. On confirmed short German cutoff fixtures (`Ich bringe das Paket zurück.`, seeds 1003/1007), this fixed the audible cutoff. In chunked mode it is applied only to the final generation unit; applying it to every chunk regressed long-form ASR. Default remains `none` to preserve prompt parity.

#### Prompt and overlap-context caveat

Canonical KugelAudio prompt construction in
`../kugelaudio-open/src/kugelaudio_open/processors/kugelaudio_processor.py`
uses this VibeVoice-derived system prompt:

```text
 Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.
```

The canonical section order is:

```text
<system prompt>
 Voice input:
 Speaker 0:<speech placeholders>
 Text input:
 Speaker 0: <text>
 Speech output:
 <speech_start>
```

Public KugelAudio prompting docs say speech is shaped directly by input text;
there is no supported separate voice-direction or non-spoken context layer. The
only documented supported tags are `<spell>` and `<prosody rate="...">`; other
markup is stripped/ignored by the hosted stack. Therefore sentence overlap that
is prepended inside `Text input:` is semantically ambiguous: it may be spoken as
normal text, or the model may skip some/all of it depending on the trajectory.
Do not assume overlap is a non-spoken context hint.

Current mitigation, when overlap is used, is to count overlap words against
`--max-words-per-chunk` and trim overlap if it would consume too much of the
spoken budget. This reduces frame-budget pressure but does not turn overlap into
a true context-only mechanism. If context-only overlap is needed later, treat it
as a new prompt/parity experiment, not a bug fix to the v1 prompt.

References:
- https://docs.kugelaudio.com/features/prompting
- https://github.com/Kugelaudio/kugelaudio-open
- https://deepwiki.com/vibevoice-community/VibeVoice/5-usage-examples

CUDA final decoder note: one-shot GPU final decode reproduced `ggml_cuda_compute_forward: IM2COL failed` before the vendored ggml CUDA im2col grid-Y fix was backported. One-shot CUDA now passes short samples, but longer chunks can require very large temporary im2col buffers and OOM. The `auto` final decoder mode therefore uses streamed final decode on GPU backends and one-shot decode on CPU, keeping CUDA iteration on-GPU while avoiding long-output one-shot decoder allocations. Use `--final-decoder-backend active` only as a diagnostic to force one-shot active-backend decode.

### Deferred-feature seams for post-v1 work
These are the current places where future features should widen behavior,
rather than patching the v1 path ad hoc.

- **Request-shape seam**
  - Code: `src/vibevoice_tts.hpp` / `src/vibevoice_tts.cpp`
  - Entrypoints: `KugelAudioRequestPolicy`, `kugelaudio_v1_request_policy()`,
    `validate_kugelaudio_request(...)`
  - Intended future use: widen request validation for multi-reference input,
    speaker-tagged dialog, or other conditioning shapes by adding/changing a
    policy profile instead of scattering new conditionals across CLI/runtime/CAPI.

- **Prompt-builder seam**
  - Code: `src/vibevoice_tts.cpp`
  - Entrypoints: `build_prompt_15b_legacy(...)` and
    `build_kugelaudio_prompt_single_speaker(...)`
  - Intended future use: keep legacy compatibility prompt behavior isolated
    while allowing future KugelAudio prompt shapes (e.g. multi-speaker or
    chunked text) to land in dedicated builders without regressing the v1
    builder.

- **Eval / quantization seam**
  - Code: `scripts/eval_kugelaudio_divergence.py`
  - Entrypoints: normalized config fields such as `ggml_model`,
    `ggml_model_q8_0`, and the shared `results.json` contract
  - Intended future use: extend acceptance coverage to broader quantization
    modes and related execution targets without redefining the fixture/eval
    surface from scratch.

- **Deferred-scope boundary**
  - Docs: `prd.md` Slice 6 and this note
  - Intended future use: long-text chunking, language hints, multi-speaker
    support, KV-cache quantization, and broader quantization formats are
    explicitly deferred and should land through the seams above, not as partial
    changes in the v1 critical path.

### External evaluator assumptions for acceptance
The harness now treats transcript fidelity and speaker similarity as **external evaluation concerns**, not shipped runtime features:
- Transcript generation uses `faster-whisper` in the Python harness
- Speaker similarity uses SpeechBrain ECAPA-TDNN cosine similarity (`speechbrain/spkrec-ecapa-voxceleb`)
- Both evaluators run on CPU by default and are intentionally kept outside the product CLI/runtime surface
- The harness still supports plain-text transcript normalization plus the older `"Content":"..."` extraction helper so future evaluator swaps do not force a results-contract rewrite

If the external evaluator output or backend changes, the transcript-normalization helper, threshold tests, and docs should be updated together.

## Targeted tests tied to these notes
- `tests/test_kugelaudio_generation_tokens.cpp`
- `tests/test_kugelaudio_reused_components.cpp`
- `tests/test_kugelaudio_reuse_coverage.cpp`
- `tests/test_kugelaudio_determinism.cpp`
- `tests/test_kugelaudio_cli_seed.cpp`
- `tests/test_dpm_solver.cpp`
- `tests/test_acoustic.cpp`
- `tests/test_kugelaudio_decode_smoke.cpp`
- `tests/test_kugelaudio_q8_0_smoke.cpp`
