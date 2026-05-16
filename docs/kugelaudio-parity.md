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
- `VIBEVOICE_BACKEND=cpu`
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
  - backend selection is process-wide and lazy; without `VIBEVOICE_BACKEND=cpu`, the runtime may pick a GPU-class backend first
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
- Usage: when executing the harness with `--execute both`, the `results.json` records the SHA256 of the canonical output. Future ggml runs should produce an output whose metrics (e.g., closed-loop ASR recall) meet the acceptance threshold relative to this ground truth.

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
- Plan mode: `./scripts/eval_kugelaudio_divergence.py --config ... --execute none`
  - Does not require model artifacts; validate config plumbing and command shapes only
- Execution mode: `./scripts/eval_kugelaudio_divergence.py --config ... --execute both --write-plan /tmp/plan.json`
  - Runs TTS (canonical + ggml) then ASR on both outputs, then computes word-level recall
  - Writes `plan.json` and `results.json` (return codes, output hashes, ASR transcripts, recall) to the config-defined `output_dir`
  - Canonical side runs inline Python in the `../kugelaudio-open` checkout
  - ggml side runs `vibevoice-cli tts ...` with CPU backend forced
  - ASR side runs `vibevoice-cli asr ...` (reuses the repo's ASR model) on both WAVs
- Acceptance metric: word-level recall against the source text
  - ggml recall must be >= 95% of canonical recall, with absolute floor 0.80
- q8_0 execution path: `VIBEVOICE_KUGELAUDIO_Q8_MODEL` env var points to a q8_0 quantized model
  - Same generation parameters and reference fixture
  - Validated by `tests/test_kugelaudio_q8_0_smoke.cpp` (non-empty, finite, non-silent output)
- Failure diagnostics: the harness produces separate `.log` files for every step (TTS and ASR for both canonical and ggml), plus `results.json` with structured status, return codes, SHA256s, transcripts, and recall values. When the threshold fails, `threshold_check.message` states exactly which boundary was crossed (ratio or floor).
- Quantization: use `scripts/quantize_gguf.py --src f16.gguf --out q8_0.gguf --type q8_0` to produce the q8_0 artifact from a converted f16 model

### ASR assumptions for closed-loop eval
The harness reuses the repo's existing ASR path (`vibevoice-cli asr` -> `vv::vibevoice_asr_transcribe()`) rather than introducing a new evaluator:
- The ASR model is loaded via `vibevoice_load()` with `variant == "asr-7b"`
- Audio input is RMS-normalized to -25 dBFS before encoding (same as upstream)
- Transcript output is raw decoded text that may contain JSON-like fields with `"Content":"..."` segments
- The harness's `extract_content()` is a Python mirror of the C++ `extract_content()` logic in `test_15b_closed_loop.cpp`
- Word-level recall is computed from the union of all Content fields, exactly matching the existing closed-loop metric

If the ASR output format changes (e.g., different JSON schema or plain text output), the `extract_content()` helper and threshold tests must be updated together.

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
