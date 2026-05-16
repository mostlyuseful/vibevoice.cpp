# vibevoice.cpp KugelAudio Port

Port `kugelaudio/kugelaudio-0-open` into the existing ggml/C++ runtime as a single-speaker, raw-reference-audio, CLI-first TTS path with canonical behavior validated against `../kugelaudio-open`.

## Slice 1: Converter + loader happy path

Deliver a converted GGUF that the runtime can recognize, load, and reject clearly when unsupported. This is the smallest integrated checkpoint because every later slice depends on a trustworthy model artifact and metadata contract.

### Major Task: Define and implement the KugelAudio GGUF contract

Specify the metadata and tensor naming contract for `kugelaudio/kugelaudio-0-open`, then implement it in `scripts/convert_vibevoice_to_gguf.py` and the runtime loader. Allow migration compatibility with legacy `vibevoice.*` keys if it keeps the diff small, but make the intended KugelAudio schema explicit and tested.

AC:
- [x] The converter accepts `kugelaudio/kugelaudio-0-open` inputs and emits a GGUF with an explicit KugelAudio metadata contract.
- [x] Unsupported checkpoints fail clearly during conversion.
- [x] The runtime loader accepts the converted KugelAudio GGUF and surfaces clear errors for missing or incompatible metadata.
- [x] The schema is documented well enough that a future agent can add validation without re-deriving intent.

Testing:
- Unit: metadata parsing, variant/config detection, tensor-name rewrite coverage, unsupported-checkpoint failure cases.
- Integration: convert the published checkpoint and load it via runtime/model-loader code.
- Manual/End-to-end: run the converter on a real checkpoint and inspect emitted metadata/logs.

#### Sub-task: Fix variant/config detection for KugelAudio

The current converter is VibeVoice-biased. Replace or narrow its heuristics so `kugelaudio/kugelaudio-0-open` is identified correctly from canonical config/tensor structure rather than accidentally classified as a VibeVoice flavor.

AC:
- [x] Detection logic correctly identifies the supported KugelAudio checkpoint.
- [x] Detection no longer relies on heuristics known to misclassify KugelAudio.
- [x] Failure messages explain why a checkpoint is unsupported.

Testing:
- Unit: config-shape fixtures and tensor-presence tests.
- Integration: conversion of `kugelaudio/kugelaudio-0-open`.
- Manual/End-to-end: inspect converter logs for correct mode selection.

#### Sub-task: Map all required TTS tensors and metadata

Implement the Qwen backbone, diffusion head, acoustic tokenizer/decoder, semantic tokenizer, connectors, scaling/bias, and special-token assumptions required for the supported path.

AC:
- [x] All tensors needed by the v1 TTS path are mapped or intentionally rejected.
- [x] Semantic-conditioning-related tensors are included for the supported path.
- [x] Missing required tensors fail fast with actionable diagnostics.

Testing:
- Unit: tensor rewrite table coverage and missing-key handling.
- Integration: converted GGUF loads without null required weights for the v1 path.
- Manual/End-to-end: inspect tensor inventory from a converted model.

### Major Task: Wire loader support for the new schema

Teach `model_loader` / `vibevoice_tts` load paths to recognize the new KugelAudio contract cleanly, including migration compatibility if both legacy and new keys are temporarily supported.

AC:
- [x] Loader recognizes the supported KugelAudio checkpoint and populates the TTS model state correctly.
- [x] Missing semantic/acoustic submodules fail clearly.
- [x] Loader behavior is deterministic and does not depend on implicit old VibeVoice defaults.

Testing:
- Unit: metadata key lookup and fallback behavior.
- Integration: converted GGUF loads into the runtime without ad hoc patches.
- Manual/End-to-end: load model from CLI and confirm model identity/features in logs.

#### Sub-task: Add schema validation and feature gating

Add explicit gating for v1-only support: single checkpoint, single-speaker, raw-reference path only.

AC:
- [x] Unsupported features are rejected before inference starts.
- [x] Error text distinguishes unsupported model/schema from unsupported runtime feature usage.
- [x] Logs state what was detected and what is enabled.

Testing:
- Unit: unsupported-feature gating cases.
- Integration: loader + CLI reject invalid combinations cleanly.
- Manual/End-to-end: try unsupported mode and verify error messaging.

## Slice 2: Canonical prompt + single-speaker voice-cloning path

Deliver the first true product path: single-speaker TTS from text plus raw reference audio, using canonical KugelAudio prompt formatting and conditioning structure.

### Major Task: Port canonical prompt construction exactly enough for parity

Adapt the prompt builder and tokenizer integration so the runtime matches `../kugelaudio-open` prompt semantics for the supported single-speaker path.

AC:
- [x] Prompt format matches canonical KugelAudio sections: system prompt, `Voice input:`, `Text input:`, `Speech output:`.
- [x] Prompt/tokenization logic is driven by KugelAudio semantics, not old VibeVoice assumptions.
- [x] Single-speaker input is the only supported path in v1 and is enforced explicitly.

Testing:
- Unit: prompt-string/prompt-token parity for fixed examples.
- Integration: generated prompt path feeds the runtime without manual adjustment.
- Manual/End-to-end: inspect verbose/debug output for a fixed sample and compare with canonical behavior.

#### Sub-task: Port special-token handling for speech placeholders

Ensure speech placeholder token positions, speech start/end handling, and any pad/diffusion token conventions line up with canonical KugelAudio.

AC:
- [x] Special token IDs and placeholder semantics match the supported KugelAudio checkpoint.
- [x] Placeholder positions are stable and testable for fixed inputs.
- [x] The runtime no longer assumes VibeVoice-only prompt token roles where KugelAudio differs.

Testing:
- Unit: token ID and placeholder-position tests.
- Integration: prompt tokens line up with voice embedding splice locations.
- Manual/End-to-end: compare placeholder counts/positions against canonical runs.

### Major Task: Implement single-reference raw-audio conditioning

Use the current repo’s audio loader path, then resample to 24 kHz mono and apply canonical RMS normalization before acoustic + semantic encoding.

AC:
- [x] Raw reference audio is accepted through the runtime/CLI path supported by the repo today.
- [x] Audio is resampled internally to 24 kHz mono.
- [x] Canonical RMS/loudness normalization is applied.
- [x] Both acoustic and semantic conditioning are used for the supported path.

Testing:
- Unit: preprocessing helpers for sample rate/channel/normalization cases.
- Integration: reference WAV produces acoustic and semantic features of expected shape.
- Manual/End-to-end: run with a real reference clip and verify non-empty conditioned generation.

#### Sub-task: Restrict and validate v1 conditioning mode

Keep only the supported conditioning mode: one raw reference input, one speaker.

AC:
- [x] Multiple references or pre-encoded voice inputs are rejected clearly.
- [x] Single reference path is fully wired through CLI -> preprocessing -> conditioning.
- [x] Error text explains the v1 limitation.

Testing:
- Unit: argument validation paths.
- Integration: CLI rejects unsupported combinations before generation.
- Manual/End-to-end: exercise valid and invalid conditioning invocations.

## Slice 3: Canonical generation-loop parity

Deliver a generation loop that behaves close enough to canonical KugelAudio to support real divergence testing, rather than merely producing audio.

### Major Task: Align constrained token generation, CFG, and speech-end behavior

Adapt the inference loop so it follows canonical KugelAudio behavior for valid-token restriction, classifier-free guidance, diffusion-token handling, and end-of-speech detection.

AC:
- [x] The runtime constrains generation to the canonical speech-path token set.
- [x] CFG behavior is aligned with the canonical implementation for the supported path.
- [x] Speech-end behavior is aligned with canonical handling and terminates correctly.
- [x] Final waveform decode path is integrated and produces usable output.

Testing:
- Unit: token-constraint logic, CFG branch behavior, speech-end decision logic.
- Integration: fixed seeded generation produces stable outputs and stops correctly.
- Manual/End-to-end: compare generation traces/logs against `../kugelaudio-open`.

#### Sub-task: Reuse existing TTS/solver/decoder pieces safely

Preserve maximum reuse of current ggml code where possible, but patch any semantic drift discovered during parity work.

AC:
- [x] Existing diffusion/decoder/connector code is reused where behavior remains valid.
- [x] Any divergence from canonical behavior is either corrected or explicitly justified.
- [x] Reused code paths remain covered by targeted tests.

Testing:
- Unit: low-level behavior around reused components as needed.
- Integration: end-to-end seeded generation through the reused path.
- Manual/End-to-end: inspect diff points between reused C++ and canonical PyTorch outputs.

### Major Task: Produce the first integrated CLI demo path

Expose the supported v1 flow through the CLI only: model path, tokenizer, text, reference audio, output WAV, and relevant generation settings.

AC:
- [x] CLI runs the supported single-speaker raw-reference path end-to-end.
- [x] Unsupported flags/features are rejected clearly.
- [x] Generated output is written as a valid waveform file.

Testing:
- Unit: CLI argument validation where practical.
- Integration: CLI invocation from tests on a fixed sample.
- Manual/End-to-end: one command generates a WAV from text + reference audio.

#### Sub-task: Add deterministic seeded runtime behavior for CPU evals

Pin CPU determinism expectations so parity and regression tests are meaningful.

AC:
- [x] Fixed seed/settings on CPU produce deterministic behavior suitable for regression testing.
- [x] Seed plumbing is exposed end-to-end through the CLI/eval path.
- [x] Known nondeterministic cases are documented if any remain.

Testing:
- Unit: seed/default handling.
- Integration: same seeded run repeated twice yields stable acceptance metrics.
- Manual/End-to-end: repeat a fixed command and verify stable output characteristics.

## Slice 4: Divergence testing + acceptance guardrails

Deliver the validation harness that proves the port is useful, not just operational. This slice turns the port into something maintainable.

### Major Task: Build canonical-vs-ggml divergence evaluation

Create a reproducible evaluation path comparing this runtime to `../kugelaudio-open` on the same checkpoint, prompt, reference audio, seed, and generation settings.

AC:
- [ ] Evaluation setup is scripted and reproducible.
- [ ] Inputs/settings are pinned and shared between canonical and ggml runs.
- [ ] Results are logged in a form suitable for regression checks.

Testing:
- Unit: config/argument plumbing for eval scripts/helpers.
- Integration: one full canonical-vs-ggml comparison run.
- Manual/End-to-end: execute the evaluation workflow and inspect outputs.

#### Sub-task: Define reproducible reference fixtures

Choose and document the fixed prompt(s), reference WAV(s), and generation settings used for acceptance.

AC:
- [ ] At least one reproducible reference sample is defined.
- [ ] At least one reproducible voice-cloned sample is defined.
- [ ] Fixture selection is documented well enough for a fresh developer to rerun.

Testing:
- Unit: fixture-path/config validation if scripted.
- Integration: fixtures run through both canonical and ggml pipelines.
- Manual/End-to-end: rerun from clean repo state and reproduce the setup.

### Major Task: Add closed-loop ASR regression as the main objective metric

Use the repo’s ASR path to measure recall and enforce the v1 thresholds.

AC:
- [ ] Closed-loop ASR regression is automated for the acceptance path.
- [ ] `f16` must reach at least 95% of canonical recall with a floor of 0.80.
- [ ] `q8_0` must complete conversion, load, and end-to-end generation on the same path.
- [ ] Failures surface enough context to distinguish model drift from harness issues.

Testing:
- Unit: metric calculation helpers if introduced.
- Integration: acceptance regression test(s) for `f16` and smoke/e2e path for `q8_0`.
- Manual/End-to-end: run acceptance workflow and inspect recall outputs.

#### Sub-task: Preserve and adapt existing ASR helper paths

Reuse current ASR and shared speech helpers rather than inventing a separate evaluator unless forced.

AC:
- [ ] Existing ASR code is reused where it keeps the eval path simple.
- [ ] Closed-loop harness integrates cleanly with the current test setup.
- [ ] Any ASR-specific assumptions needed for KugelAudio eval are documented.

Testing:
- Unit: none beyond helper-level additions.
- Integration: TTS -> WAV -> ASR roundtrip on fixed acceptance fixtures.
- Manual/End-to-end: inspect roundtrip transcript quality and metric output.

## Slice 5: Hardening, diagnostics, and migration cleanup

Deliver a safer day-2 development experience: better failures, clearer logs, and less accidental dependence on old VibeVoice assumptions.

### Major Task: Harden errors, observability, and operator diagnostics

Make failures explicit and logs useful enough to debug converter, loader, prompt, and eval issues without re-deriving context.

AC:
- [ ] Converter, loader, conditioning, and unsupported-feature errors are clear and actionable.
- [ ] Logs include checkpoint/config, converter mode, active conditioning, quantization mode, and eval configuration.
- [ ] Sensitive prompt/audio contents are not dumped by default.

Testing:
- Unit: error-path formatting where practical.
- Integration: invalid invocations produce stable, informative diagnostics.
- Manual/End-to-end: review logs from success and failure cases.

#### Sub-task: Remove or quarantine misleading legacy paths from the critical path

Reduce confusion from old VibeVoice-specific code paths that are not part of KugelAudio v1 acceptance.

AC:
- [ ] Legacy paths that are not part of v1 acceptance are clearly marked or kept off the critical path.
- [ ] Maintainer docs reflect the KugelAudio-first reality.
- [ ] Acceptance scripts/tests do not depend on dropped features.

Testing:
- Unit: none.
- Integration: acceptance path uses only intended v1 components.
- Manual/End-to-end: fresh maintainer can follow docs without falling into legacy flows.

### Major Task: Prepare the codebase for V2 without implementing V2

Leave clean seams for later work: chunking, language hints, multi-speaker, KV-cache quantization, and broader quantization support.

AC:
- [ ] V1 design does not block deferred features.
- [ ] Deferred-feature seams are identified in code/docs where relevant.
- [ ] No V2 feature is half-implemented in the critical path.

Testing:
- Unit: n/a unless a seam adds config validation.
- Integration: v1 path still passes after cleanup/refactor.
- Manual/End-to-end: code review confirms deferred features are clearly separated from v1.

#### Sub-task: Document migration assumptions in repo-facing docs

Keep `spec.md`, maintainer guidance, and any eval/conversion notes aligned with observed reality.

AC:
- [ ] Developer-facing docs reflect current v1 scope and acceptance criteria.
- [ ] Canonical reference points into `../kugelaudio-open` are documented.
- [ ] Known divergences or temporary compatibility shims are recorded succinctly.

Testing:
- Unit: n/a.
- Integration: docs match current scripts/tests/CLI shape.
- Manual/End-to-end: another agent/developer can pick the next task without re-asking settled questions.

## Slice 6: Optional V2 / deferred roadmap

This slice is explicitly outside the v1 critical path.

### Major Task: Canonical long-text chunking parity

Implement chunk planning/stitching with parity to canonical KugelAudio behavior.

AC:
- [ ] Long-text chunking matches canonical behavior closely enough for regression testing.
- [ ] Pause/crossfade/overlap behavior is tested and documented.

Testing:
- Unit: chunk planner and stitching helpers.
- Integration: multi-chunk generation on canonical fixtures.
- Manual/End-to-end: listen to stitched outputs.

### Major Task: Broader quantization and KV-cache work

Expand beyond `f16` parity and `q8_0` execution.

AC:
- [ ] Additional quantization formats are named, implemented, and benchmarked.
- [ ] KV-cache quantization is exposed in a controlled, testable way.

Testing:
- Unit: quantization option validation.
- Integration: per-format load/e2e checks.
- Manual/End-to-end: run benchmark/eval comparisons.

### Major Task: Language hints and multi-speaker support

Add deferred canonical features once v1 parity is stable.

AC:
- [ ] Language hints behave like the canonical implementation.
- [ ] Multi-speaker dialog/ref handling is implemented and tested.

Testing:
- Unit: prompt-building and validation logic.
- Integration: canonical-vs-ggml tests for each feature.
- Manual/End-to-end: feature demos through CLI.
