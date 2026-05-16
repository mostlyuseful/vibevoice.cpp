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

## Targeted tests tied to these notes
- `tests/test_kugelaudio_generation_tokens.cpp`
- `tests/test_kugelaudio_reused_components.cpp`
- `tests/test_dpm_solver.cpp`
- `tests/test_acoustic.cpp`
- `tests/test_kugelaudio_decode_smoke.cpp`
