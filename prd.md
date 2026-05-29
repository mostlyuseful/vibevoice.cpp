# kugelaudio.cpp KugelAudio Port

Port `kugelaudio/kugelaudio-0-open` into the existing ggml/C++ runtime as a single-speaker, raw-reference-audio, CLI-first TTS path with canonical behavior validated against `../kugelaudio-open`.

## Slice 1: Converter + loader happy path

Deliver a converted GGUF that the runtime can recognize, load, and reject clearly when unsupported. This is the smallest integrated checkpoint because every later slice depends on a trustworthy model artifact and metadata contract.

### Major Task: Define and implement the KugelAudio GGUF contract

Specify the metadata and tensor naming contract for `kugelaudio/kugelaudio-0-open`, then implement it in `scripts/convert_kugelaudio_to_gguf.py` and the runtime loader. Allow migration compatibility with legacy `vibevoice.*` keys if it keeps the diff small, but make the intended KugelAudio schema explicit and tested.

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

### Major Task: Fix review-blocked converter/loader contract regressions

Resolve the code-review findings that currently block the documented `kugelaudio/kugelaudio-0-open` conversion and can mis-populate the runtime layer split from newly converted artifacts.

AC:
- [ ] Converter-supported signature matches the canonical `../kugelaudio-open` config for the supported checkpoint, including `tts_backbone_num_hidden_layers = 20`.
- [ ] `detect_variant()` accepts `kugelaudio/kugelaudio-0-open` and still rejects other KugelAudio configs clearly.
- [ ] Loader resolves `cfg.n_layers_lm = 8` and `cfg.n_layers_tlm = 20` for converter-produced KugelAudio GGUFs.
- [ ] The canonical `kugelaudio.*` keys and compatibility `vibevoice.*` keys agree on the LM/TTS layer split, or a mismatch fails clearly instead of silently loading the wrong stack depth.

Testing:
- Unit: `tests/test_convert_kugelaudio_metadata.py` passes against the canonical 7B config fixture.
- Unit: loader metadata fixture covers converter-style artifacts carrying both `kugelaudio.decoder.num_hidden_layers = 28` and `vibevoice.n_layers_lm = 8`.
- Integration: convert the supported checkpoint, load it, and verify runtime logs/config report `layers=8+20`.
- Manual/End-to-end: run convert -> optional quantize -> CLI load smoke after the fix.

#### Sub-task: Fix supported KugelAudio signature detection

The converter currently expects `tts_backbone_num_hidden_layers` to be absent/`None`, but the canonical supported config and docs specify `20`. Align the signature check with the real config so the supported checkpoint is not rejected before conversion starts.

AC:
- [ ] `SUPPORTED_KUGELAUDIO_SIGNATURE` matches `../kugelaudio-open/src/kugelaudio_open/configs/kugelaudio_7b.json` for the supported checkpoint.
- [ ] The failure message for unsupported KugelAudio configs still includes the observed signature.
- [ ] Tests cover both accepted supported config and rejected alternate config.

Testing:
- Unit: direct `detect_variant(KUGEL_7B_CFG, ...) == "kugelaudio-0-open"`.
- Unit: direct rejected-config test for `KUGEL_15B_CFG` or another intentionally unsupported shape.
- Integration: converter reaches tensor validation for the supported checkpoint instead of failing during variant detection.

#### Sub-task: Fix loader layer-split precedence

The loader currently reads `kugelaudio.decoder.num_hidden_layers` before `vibevoice.n_layers_lm`; when both are present, it can keep the total layer count (`28`) as the lower LM stack depth instead of the compatibility split (`8`). Prefer the explicit compatibility split when present, or compute `num_hidden_layers - tts_hidden_layers` from canonical keys when compatibility keys are absent.

AC:
- [ ] Converted KugelAudio artifacts with both key families load as `n_layers_lm=8`, `n_layers_tlm=20`.
- [ ] KugelAudio-only artifacts without `vibevoice.n_layers_lm` compute the same `8+20` split from canonical keys.
- [ ] Ambiguous or inconsistent metadata is rejected with an actionable loader error.

Testing:
- Unit: loader fixture for both-key metadata and KugelAudio-only metadata.
- Integration: converted real checkpoint loads the expected number of `lm.blk.*` layers without looking for nonexistent `lm.blk.8..27` tensors.
- Manual/End-to-end: CLI startup log shows the expected KugelAudio layer split.

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

Use the current repo's audio loader path, then resample to 24 kHz mono and apply canonical RMS normalization before acoustic + semantic encoding.

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
- [x] Evaluation setup is scripted and reproducible.
- [x] Inputs/settings are pinned and shared between canonical and ggml runs.
- [x] Results are logged in a form suitable for regression checks.

Testing:
- Unit: config/argument plumbing for eval scripts/helpers.
- Integration: one full canonical-vs-ggml comparison run.
- Manual/End-to-end: execute the evaluation workflow and inspect outputs.

#### Sub-task: Define reproducible reference fixtures

Choose and document the fixed prompt(s), reference WAV(s), and generation settings used for acceptance.

AC:
- [x] At least one reproducible reference sample is defined.
- [x] At least one reproducible <private-audio-dir>d sample is defined.
- [x] Fixture selection is documented well enough for a fresh developer to rerun.

Testing:
- Unit: fixture-path/config validation if scripted.
- Integration: fixtures run through both canonical and ggml pipelines.
- Manual/End-to-end: rerun from clean repo state and reproduce the setup.

### Major Task: Define KugelAudio-native regression as the main objective metric

Use canonical-vs-ggml divergence metrics that fit a KugelAudio-only TTS branch rather than depending on a published ASR product surface.

Planned published metric set:
- external **text-fidelity** evaluation via `faster-whisper` in the Python eval harness
- external **speaker-similarity** evaluation via pretrained speaker embeddings / cosine similarity
- existing runtime sanity checks (non-empty audio, finite samples, non-silent output)

AC:
- [x] Published acceptance no longer depends on shipping or exposing ASR capability.
- [ ] `f16` must reach at least 95% of canonical quality on the selected KugelAudio-native metric set, with explicit per-metric floors documented.
- [x] `q8_0` must complete conversion, load, and end-to-end generation on the same path.
- [x] Failures surface enough context to distinguish model drift from harness/runtime issues.

Testing:
- Unit: metric calculation helpers if introduced.
- Integration: acceptance regression test(s) for `f16` and smoke/e2e path for `q8_0`.
- Manual/End-to-end: run acceptance workflow and inspect the divergence outputs.

#### Sub-task: Replace transitional ASR-based validation with TTS-native checks

Keep any temporary internal migration aids off the published critical path and converge the acceptance harness on external evaluators that do not imply a shipped ASR/runtime product.

Planned implementation shape:
- use `faster-whisper` from `scripts/eval_kugelaudio_divergence.py` for transcript generation / WER-style text fidelity
- use speaker-embedding cosine similarity for reference-vs-generated voice similarity
- keep both evaluators outside the product/runtime/CLI surface

AC:
- [x] Acceptance harness runs without requiring ASR build, CLI, or runtime support.
- [x] Replacement metrics are explicit, automatable, and documented in repo-facing docs.
- [x] Docs distinguish any temporary internal validators from the published KugelAudio branch surface.
- [x] Eval results clearly separate transcript-fidelity failure from speaker-similarity failure.

Testing:
- Unit: helper-level validation for the replacement metric set.
- Integration: canonical-vs-ggml acceptance run on fixed KugelAudio fixtures.
- Manual/End-to-end: inspect metric output and confirm it is actionable without ASR.

## Slice 5: Hardening, diagnostics, and migration cleanup

Deliver a safer day-2 development experience: better failures, clearer logs, and less accidental dependence on old VibeVoice assumptions.

### Major Task: Harden errors, observability, and operator diagnostics

Make failures explicit and logs useful enough to debug converter, loader, prompt, and eval issues without re-deriving context.

AC:
- [x] Converter, loader, conditioning, and unsupported-feature errors are clear and actionable.
- [x] Logs include checkpoint/config, converter mode, active conditioning, quantization mode, and eval configuration.
- [x] Sensitive prompt/audio contents are not dumped by default.

Testing:
- Unit: error-path formatting where practical.
- Integration: invalid invocations produce stable, informative diagnostics.
- Manual/End-to-end: review logs from success and failure cases.

#### Sub-task: Remove or quarantine misleading legacy paths from the critical path

Reduce confusion from old VibeVoice-specific code paths that are not part of KugelAudio v1 acceptance.

AC:
- [x] Legacy paths that are not part of v1 acceptance are clearly marked or kept off the critical path.
- [x] Maintainer docs reflect the KugelAudio-first reality.
- [x] Acceptance scripts/tests do not depend on dropped features.

Testing:
- Unit: none.
- Integration: acceptance path uses only intended v1 components.
- Manual/End-to-end: fresh maintainer can follow docs without falling into legacy flows.

### Major Task: Prepare the codebase for V2 without implementing V2

Leave clean seams for later work: chunking, language hints, multi-speaker, KV-cache quantization, and broader quantization support.

AC:
- [x] V1 design does not block deferred features.
- [x] Deferred-feature seams are identified in code/docs where relevant.
- [x] No V2 feature is half-implemented in the critical path.

Testing:
- Unit: n/a unless a seam adds config validation.
- Integration: v1 path still passes after cleanup/refactor.
- Manual/End-to-end: code review confirms deferred features are clearly separated from v1.

#### Sub-task: Document migration assumptions in repo-facing docs

Keep `spec.md`, maintainer guidance, and any eval/conversion notes aligned with observed reality.

AC:
- [x] Developer-facing docs reflect current v1 scope and acceptance criteria.
- [x] Canonical reference points into `../kugelaudio-open` are documented.
- [ ] Known divergences or temporary compatibility shims are recorded succinctly.

Testing:
- Unit: n/a.
- Integration: docs match current scripts/tests/CLI shape.
- Manual/End-to-end: another agent/developer can pick the next task without re-asking settled questions.

## Slice 6: GPU backend support (CUDA + Vulkan)

Deliver first-class CUDA and Vulkan build targets with runtime backend selection, correctness smoke coverage, and eval-harness integration. The repo already forwards ggml backend toggles and calls `ggml_backend_load_all()`; this slice hardens those paths into a supported, tested surface rather than an aspirational one.

### Major Task: Make CUDA and Vulkan first-class build targets

Ensure both backends build cleanly from the top-level CMake, document the recipes, and verify that `ggml-backend` targets (`ggml-cuda`, `ggml-vulkan`) are produced when the corresponding toggles are enabled.

AC:
- [x] `cmake -B build-cuda -DKUGELAUDIO_GGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release` completes. Verified on `CUDA workstation` with `CUDAToolkit_ROOT=/opt/cuda` and `CMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc`.
- [ ] `cmake -B build-vulkan -DKUGELAUDIO_GGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release` completes.
- [ ] Both build directories produce a working `bin/kugelaudio-cli`. CUDA verified on `CUDA workstation`; Vulkan remains separate.
- [ ] Build recipes are documented in `README.md`.

Testing:
- Unit: n/a.
- Integration: CI-adjacent manual build verification.
- Manual/End-to-end: operator can follow documented commands and get a working binary.

#### Sub-task: Add backend summary logging at startup

Extend `src/backend.cpp` to log all registered device names from `ggml_backend_dev_count()`, the exact selected device, the fallback reason if any, and whether flash attention is supported on the active backend.

AC:
- [x] Backend init logs include: requested backend, selected backend, available device list, flash-attn status.
- [x] Fallback paths include an explicit reason in the log.

Testing:
- Unit: none.
- Integration: inspect stderr from CLI version/help under each backend.
- Manual/End-to-end: `KUGELAUDIO_BACKEND=cuda ./build-cuda/bin/kugelaudio-cli version` emits meaningful backend diagnostics.

### Major Task: Runtime backend selection and operator diagnostics

Make `KUGELAUDIO_BACKEND=cuda|vulkan|cpu` the canonical way to select a backend. Add optional env vars for device index and backend-verbose mode.

AC:
- [x] `KUGELAUDIO_BACKEND=cuda` selects CUDA backend when available.
- [ ] `KUGELAUDIO_BACKEND=vulkan` selects Vulkan backend when available.
- [ ] `KUGELAUDIO_BACKEND=cpu` forces CPU even when GPU backends are registered.
- [x] Missing/incompatible backends fall back cleanly with a logged reason.
- [x] Optional `KUGELAUDIO_BACKEND_DEVICE_INDEX` and `KUGELAUDIO_BACKEND_VERBOSE` env vars are supported.

Testing:
- Unit: none.
- Integration: CLI smoke under each backend variant.
- Manual/End-to-end: operator can switch backends at runtime and see the selection in logs.

### Major Task: Backend inference correctness audit

Audit all major graph builders for backend assumptions and confirm the KugelAudio inference graph actually executes on CUDA and Vulkan without crashes or unsupported-op failures. High-risk areas: conv1d / conv-transpose1d, acoustic/semantic encoder graphs, diffusion head, Qwen2 stack, flash-attn shape differences.

AC:
- [ ] TTS conditioning-only path completes on CUDA and Vulkan without unsupported-op errors.
- [ ] First diffusion step completes on CUDA and Vulkan.
- [ ] Full decode produces a valid, non-silent WAV on CUDA and Vulkan.
- [ ] Flash-attn unavailability on a backend is handled gracefully (no crash).

Testing:
- Unit: flash-attn probe helper already exists; extend if needed.
- Integration: backend-smoke tests (see below).
- Manual/End-to-end: run the sine fixture on each backend and verify non-empty output.

#### Sub-task: Hybrid Vulkan generation + CPU final decoder fallback

Prepare a backend-mixed execution mode for memory-constrained Vulkan systems without attempting full llama.cpp-style per-layer offload. The intended shape is narrower: keep the encoder / Qwen2 / diffusion path on Vulkan, then run the final acoustic decoder on CPU so longer latent sequences do not have to be staged through the fragile Vulkan decode upload path.

AC:
- [x] The repo-facing plan names hybrid mode as a targeted fallback for Vulkan length limits, not a general multi-backend scheduler.
- [x] The proposed split point is explicit: Vulkan for conditioning + autoregressive/diffusion generation, CPU for final waveform decode.
- [x] Acceptance criteria for the future implementation are documented: longer-than-current Vulkan outputs must complete without crash, with no silent output and no prompt/conditioning behavior change.
- [x] The plan explicitly defers true llama.cpp-style per-layer / `-ngl`-style offload until after a broader multi-backend scheduling design exists.

Testing:
- Unit: n/a until implementation exists.
- Integration: future smoke should compare pure-CPU, pure-Vulkan-short, and Vulkan+CPU-hybrid-long runs on the same fixture.
- Manual/End-to-end: validate that a reference clip exceeding the current Vulkan-safe frame cap finishes in hybrid mode.

#### Sub-task: Stream the acoustic decoder on Vulkan instead of one-shot full-sequence decode

Prepare a Vulkan-only path that reduces peak decoder memory by splitting the final latent-to-waveform decode into smaller chunks while preserving causal behavior. This is explicitly not the naive "decode independent chunks and concatenate" approach; it requires decoder-side cache/state so chunked output remains aligned with the single-shot causal decoder semantics.

AC:
- [x] The plan documents decoder streaming/chunking as distinct from the CPU-hybrid fallback and distinct from full llama.cpp-style multi-backend offload.
- [x] The proposed implementation shape is explicit: introduce decoder-side streaming/cache primitives and a chunked latent-sequence decode path rather than decoding each chunk cold.
- [x] Acceptance criteria for the future implementation are documented: longer latent sequences must complete on Vulkan with lower peak memory, no crash, and no audible seam/silence regression relative to the supported baseline.
- [x] The plan calls out the highest-risk areas: causal transposed-conv state, per-stage cache semantics, and chunk-vs-full parity testing.

Testing:
- Unit: future decoder cache/state helpers and transposed-conv boundary behavior.
- Integration: future parity tests comparing full decode vs chunked decode on fixed latent fixtures.
- Manual/End-to-end: validate that Vulkan can decode longer samples than the current one-shot path without introducing boundary artifacts.

### Major Task: Add backend-specific smoke tests

Add small, skippable C++ tests that load the model and run TTS on CUDA and Vulkan when the corresponding backend device is available.

AC:
- [x] `test_kugelaudio_cuda_smoke` passes on supported CUDA hardware, returns 77 (skip) otherwise.
- [x] `test_kugelaudio_vulkan_smoke` passes on supported Vulkan hardware, returns 77 (skip) otherwise.
- [x] `test_kugelaudio_cuda_conditioning` validates the encoder/connector graph on GPU.
- [x] `test_kugelaudio_vulkan_hybrid_decode` validates the explicit CPU final-decoder fallback.
- [x] `test_kugelaudio_vulkan_streamed_decode` validates the streamed final-decoder path.
- [x] Tests reuse existing model/tokenizer/ref-audio env-var conventions.

Testing:
- Unit: test registration and skip logic.
- Integration: run under `ctest` on a machine with the target GPU.
- Manual/End-to-end: `ctest --test-dir build-cuda` / `ctest --test-dir build-vulkan` pass applicable tests.

Blocking manual verification step before Slice 6 is considered complete:
- [x] Run `test_kugelaudio_cuda_smoke` and `test_kugelaudio_cuda_conditioning` on real CUDA hardware, capture the backend-selection logs, and confirm the tests pass without CPU-fallback-only execution masking CUDA failures. Verified on `CUDA workstation` / RTX 3090; smoke logs `backend=CUDA0` and auto streamed final decoder.

### Major Task: Backend-aware eval harness integration

Extend `scripts/eval_kugelaudio_divergence.py` with a `--ggml-backend` flag or config field so the same acceptance fixture can be executed across CPU, CUDA, and Vulkan from one harness invocation.

AC:
- [x] `--ggml-backend cpu|cuda|vulkan` controls `KUGELAUDIO_BACKEND` in the ggml CLI subprocess.
- [x] Config field `"ggml_backend"` is accepted in the eval JSON config.
- [x] Results are written into the shared `results.json` contract.

Testing:
- Unit: argument/config plumbing tests.
- Integration: produce separate `results.json` entries for CPU and CUDA from the same fixture.
- Manual/End-to-end: run `scripts/eval_kugelaudio_divergence.py --execute ggml --ggml-backend cuda`.

#### Sub-task: Define backend acceptance criteria

Document what "supported" means for each backend so future maintainers can evaluate regressions.

AC:
- [ ] CUDA must support: `f16` TTS, `q8_0` TTS, sine + real-reference fixtures, no crash/NaN/silence.
- [ ] Vulkan must support: `f16` TTS, sine + real-reference fixtures, no crash/NaN/silence.
- [ ] GPU backends are explicitly excluded from the determinism contract (CPU-only).

Testing:
- Unit: n/a.
- Integration: criteria are verifiable against the smoke test suite.
- Manual/End-to-end: acceptance document matches what the tests actually check.

### Major Task: Performance and memory follow-up

After correctness, identify and remove obvious GPU inefficiencies: reduce unnecessary host-device syncs from `ggml_backend_tensor_get`, cache small tensor reads on host where safe, verify encoder memory profiles on CUDA/Vulkan.

AC:
- [ ] At least the most expensive host-device sync paths are identified and documented.
- [ ] Encoder graph memory usage is characterized on CUDA and Vulkan.
- [ ] Optional backend timing/logging is exposed for operator diagnostics.

Testing:
- Unit: none.
- Integration: smoke tests still pass after any optimization.
- Manual/End-to-end: operator can enable timing logs and interpret output.

## Slice 7: KugelAudio-first publish cleanup

Polish the branch for publication by removing stale VibeVoice identity, tightening the published product surface to KugelAudio-only TTS, and eliminating repo-facing terminology that no longer matches the supported model family.

### Major Task: Canonicalize the published GGUF/runtime identity as KugelAudio

Stop emitting new artifacts that identify as `vibevoice` when they are produced from KugelAudio checkpoints. Make `kugelaudio` the primary published identity across converter, quantizer, loader logs, docs, and tests.

AC:
- [ ] Converter-produced KugelAudio GGUFs write `general.architecture = kugelaudio`.
- [ ] Quantizer-produced KugelAudio GGUFs preserve or emit `general.architecture = kugelaudio` rather than rewriting them as `vibevoice`.
- [ ] Loader/runtime logs describe the supported model as KugelAudio by default.
- [ ] No newly produced KugelAudio artifact requires a `vibevoice` architecture string to load successfully.
- [ ] If legacy `vibevoice`-tagged KugelAudio GGUFs remain readable during migration, that compatibility is explicit, tested, and documented as transitional.

Testing:
- Unit: converter/quantizer metadata tests for `general.architecture` and related KugelAudio keys.
- Integration: convert -> quantize -> load roundtrip on the supported checkpoint.
- Manual/End-to-end: inspect emitted GGUF metadata and runtime startup logs for a converted+quantized model.

#### Sub-task: Define the migration boundary for legacy metadata compatibility

Decide whether to keep read-only compatibility for old `vibevoice`-tagged KugelAudio GGUFs, and if so, ensure the direction is one-way: old artifacts may still load, but all newly written artifacts are KugelAudio-native.

AC:
- [ ] The migration policy is explicit: write-new-as-KugelAudio, optional read-old-as-legacy.
- [ ] Failure messages distinguish unsupported legacy VibeVoice artifacts from supported legacy-tagged KugelAudio artifacts.
- [ ] Docs record the migration expectation succinctly.

Testing:
- Unit: legacy/new metadata acceptance and rejection cases.
- Integration: loader accepts the intended migration set and rejects truly unsupported artifacts.
- Manual/End-to-end: inspect error text for ambiguous legacy inputs.

### Major Task: Remove VibeVoice-only terminology from published repo surfaces

Replace misleading `1.5B`, `tts15b`, and similar VibeVoice-era labels in operator-facing text with KugelAudio-accurate language.

AC:
- [x] Operator-facing docs, CLI help, logs, and tests no longer describe the supported KugelAudio path as `1.5B` or `tts15b`.
- [x] The supported model path is described consistently as KugelAudio TTS / supported KugelAudio checkpoint.
- [x] Remaining legacy/internal names, if any, are either removed or clearly quarantined from published/operator-facing surfaces.
- [x] Env vars, fixture names, and test names exposed to maintainers are KugelAudio-first where practical.

Testing:
- Unit: none beyond string/flag validation where practical.
- Integration: CLI help, README snippets, eval/config fixtures, and smoke tests agree on the supported naming.
- Manual/End-to-end: grep/audit published surfaces and confirm a fresh user would not infer VibeVoice 1.5B support from KugelAudio docs.

#### Sub-task: Decide the renaming boundary for internal symbols

Avoid churn for purely internal names unless they leak into published UX. Rename internal symbols only where they affect logs, tests, docs, or future maintainability materially.

AC:
- [x] Operator-visible naming is fully KugelAudio-first.
- [x] Internal renames are limited to places where stale naming creates confusion or maintenance risk.
- [x] The PRD/docs explicitly note any deferred internal renames that are intentionally left in place.

Testing:
- Unit: n/a.
- Integration: builds/tests pass after the selected rename scope.
- Manual/End-to-end: code review confirms the rename boundary is intentional rather than accidental.

### Major Task: Remove non-KugelAudio product surfaces from the published branch

Tighten the published scope to KugelAudio TTS only by removing repo-facing capabilities that are specific to legacy VibeVoice products and not part of the intended KugelAudio surface.

AC:
- [x] ASR is removed as a published/runtime capability from the branch.
- [x] Pre-baked voice artifacts / `voice.gguf` flow are removed from the published CLI, docs, tests, and scripts.
- [x] VibeVoice-specific CLI/documentation/test entrypoints that are not part of KugelAudio TTS are removed or explicitly dropped from the publish target.
- [x] Shared helpers that remain useful after ASR removal are relocated/renamed so they no longer imply a supported ASR product surface.
- [x] The published branch describes itself as KugelAudio TTS only, not a mixed VibeVoice/KugelAudio multiproduct repo.

Testing:
- Unit: build/test coverage updated to remove ASR assumptions.
- Integration: supported KugelAudio TTS path still converts, loads, quantizes, and generates end-to-end.
- Manual/End-to-end: published docs and CLI surface expose only intended KugelAudio functionality.

#### Sub-task: Replace ASR-based acceptance with KugelAudio-native validation

The current acceptance path depends on closed-loop ASR, which is not a KugelAudio product capability. Replace that dependency with acceptance checks that are appropriate for a KugelAudio-only TTS repo.

Chosen direction for the published path:
- `faster-whisper` provides transcript-fidelity scoring in the Python eval harness
- speaker-embedding cosine similarity provides voice-cloning fidelity scoring
- these remain external eval dependencies, not runtime product features

AC:
- [x] PRD acceptance criteria no longer require shipping or exposing ASR capability.
- [x] Canonical-vs-ggml validation remains reproducible without relying on a published ASR feature.
- [x] Replacement acceptance metrics are explicit, automatable, and documented.
- [x] Failure output still distinguishes transcript-fidelity drift, speaker-similarity drift, and harness/runtime failure.

Testing:
- Unit: replacement metric/helper validation.
- Integration: acceptance harness runs on the supported fixture without ASR runtime dependencies.
- Manual/End-to-end: rerun acceptance workflow and inspect the new results contract.

### Major Task: Prune legacy VibeVoice docs, tests, and scripts from the publish target

Remove or archive repo content that would mislead users into thinking the branch still supports old VibeVoice-only model families or workflows.

AC:
- [x] Legacy VibeVoice quickstarts and examples that are not part of the KugelAudio branch publish target are removed or moved out of the default path.
- [x] Legacy scripts such as unused voice-gguf / VibeVoice-only helpers are either deleted or explicitly archived.
- [x] Tests that only validate removed VibeVoice/ASR surfaces are deleted or moved out of the default suite.
- [x] Maintainer docs point only to the supported KugelAudio workflow for this branch.

Testing:
- Unit: n/a.
- Integration: default build/test/docs flow no longer depends on removed legacy pieces.
- Manual/End-to-end: a new maintainer can clone the branch and find only the intended KugelAudio-first workflow.

#### Sub-task: Rewrite repo-facing acceptance/docs around the new publish surface

After the removals, align `README.md`, `docs/conversion.md`, `docs/kugelaudio-parity.md`, `AGENTS.md`, and this PRD with the actual published scope.

AC:
- [x] README describes only the supported KugelAudio publish surface.
- [x] Conversion/quantization docs refer to KugelAudio artifacts and terminology.
- [x] PRD slices that currently mention ASR or VibeVoice-only capabilities are updated or retired.
- [x] Maintainer guidance no longer sends fresh contributors into removed legacy paths.

Testing:
- Unit: n/a.
- Integration: docs match actual CLI/scripts/tests after cleanup.
- Manual/End-to-end: doc audit from clean checkout succeeds without legacy contradictions.

## Slice 8: Long-form quality and CUDA iteration speed

This slice captures the current long-form listening plan after continuity experiments. The working conclusion is that generated-state carry is harmful: `tail-reference` improves seams but feeds decoded waveform noise forward, `clean-tail-reference` still causes noise/speaker drift, `latent-prefix` worsens noise/intonation, and system prompt continuity can smooth seams but still changes speaker identity. The practical path is clean independent chunks, fewer seams, true model-state baselines, and faster CUDA iteration.

### Major Task: Retire harmful continuity experiments and document the outcome

Keep default/parity behavior (`--chunk-continuity none`) as the recommended long-form mode. Retain explicit documentation for failed experiments so future work does not reintroduce generated audio/latent feedback without evidence.

AC:
- [x] `tail-reference` is rejected or quarantined from the recommended CLI path with a clear reason: decoded waveform artifacts feed back and amplify from chunk 2 onward.
- [x] `latent-prefix` is rejected or quarantined from the recommended CLI path with a clear reason: listening showed worse noise and intonation drift.
- [x] `prompt-instruction` is documented as a retired seam-smoothing attempt that can still alter speaker identity.
- [x] `clean-tail-reference` is documented as a retired generated-waveform feedback attempt that still caused noise and speaker drift.
- [x] README / parity docs recommend `--chunk-continuity none` for quality-critical long-form output.

Testing:
- Unit/integration: CLI rejection tests for retired generated-state modes.
- Manual/End-to-end: keep listening notes and sample paths in project memory.

### Major Task: Retire sanitized clean-tail reference continuity experiment

Run one final waveform-tail continuity experiment that avoids the failure mode of retired `tail-reference`: never use an indiscriminate chunk ending as reference. Always anchor on the original raw reference, append only a short selected clean voiced island from the previous chunk, and sanitize it before conditioning the next chunk.

Proposed mode name:
- `--chunk-continuity clean-tail-reference`

Proposed extraction/alignment:
- scan backward from the previous chunk end in short RMS windows to find voiced regions
- choose the cleanest voiced island near the end, not necessarily the literal last audio
- start with `--continuity-tail-ms 500`
- remove DC / mean from the selected tail
- match RMS to voiced original-reference RMS with conservative gain clamp (`0.5x..1.25x` initially)
- avoid min/max matching initially, because outliers can amplify artifacts
- apply short fade-in/fade-out and peak clamp
- optionally low-pass/high-pass if the first clean-tail run still carries hiss
- synthesize next reference as `original_ref + short_silence + aligned_clean_tail`

Experiment ladder:
1. original ref + first clean 500 ms of chunk 1
2. original ref + last clean 500 ms of chunk 1
3. original ref + selected clean voiced island near chunk end
4. same as 3 plus conservative filtering if needed
5. recursive vs non-recursive:
   - recursive: chunk N clean tail conditions chunk N+1
   - non-recursive: one clean chunk-1 tail reused for all chunks

AC:
- [x] Implement `clean-tail-reference` as opt-in only; default remains `none`.
- [x] Keep retired `tail-reference` rejected so the old unsafe mode is not silently revived.
- [x] Generate and compare the experiment ladder against `none` and prior failed continuity samples.
- [x] Decide whether clean-tail-reference is safe enough to keep as an experimental mode or should be retired/documented too. Decision: retire; listening still found noise collapse and speaker identity changes.

Testing:
- Unit/integration: extraction rejects silence/low-RMS tails and clamps gain conservatively.
- Manual/End-to-end: listening comparison for noise onset, seam identity, and recursive artifact accumulation.

### Major Task: Add true model-state continuity baseline

Implement a first true-state experiment that does not feed generated waveform or latent snippets back into reference conditioning. The initial baseline is `--chunk-continuity single-sequence`: use chunk planning only to estimate a total frame budget, then prefill the full text once and generate in a single LM/KV-cache sequence.

AC:
- [x] `single-sequence` is exposed as an opt-in continuity mode.
- [x] It uses the full original text prompt once rather than independent per-chunk prompts.
- [x] Its total frame budget is `max_speech_frames * planned_chunks`.
- [x] Generate q8/CUDA listening sample and compare against `none` chunking for speaker drift, noise, and within-sequence degeneration. Initial `mw80/f64` single-sequence stopped early at 259/640 frames (34.5s); min-frame suppression reached 576/640 frames but collapsed into near-silence after ~40s.
- [x] Prototype segmented KV carry (`--chunk-continuity segmented-state`) that inserts fresh text/control blocks into the same LM state between segment budgets. Listening showed repeated text across chunks, first-chunk truncation, and progressive softening, so this is an off-distribution diagnostic rather than a production path.

Testing:
- Integration: CLI accepts true-state diagnostic modes and logs total/segment frame budgets.
- Manual/End-to-end: compare long-form output to independent chunks.

### Major Task: Canonical non-chunked oracle and rolling-state parity diagnostics

Stop optimizing against canonical chunking. The target behavior is the canonical non-chunked long generation trajectory: a chunked or bounded-memory method should stay as close as possible to the baseline one-shot path while avoiding VRAM limits. If local/CUDA workstation VRAM is insufficient for canonical long dumps, provision a RunPod VM and keep the dump format portable.

Plan:
1. Add canonical tensor dumping to `../kugelaudio-open` for a non-chunked run: prompt IDs, reference conditioning, prefill hidden, per-frame LM hidden before diffusion, speech logits, diffusion initial noise, DPM step outputs/final latent, speech connector output, and final acoustic decoder input.
2. Add matching C++ tensor dumps for non-chunked generation.
3. Avoid relying on seed parity: canonical should dump exact per-frame diffusion noise, and C++ should be able to load that noise for teacher-forced comparison.
4. First compare canonical non-chunked vs C++ non-chunked. If this diverges early, fix parity before interpreting any chunking result.
5. Then compare bounded-memory experiments against the non-chunked oracle. Do **not** use canonical chunking as the target.
6. Preferred next experiment is rolling-KV / masked-history approximation: keep the original prompt/reference/text context plus only the last K generated speech embeddings/tokens, with K ladder such as 0, 16, 32, 64, 128, 256. Compare hidden cosine, speech-logit KL/rank, latent error, connector-output cosine, and audio RMS trajectory to the non-chunked oracle.

AC:
- [x] Canonical non-chunked dump script produces a self-describing dump directory for a reproducible text/ref/seed/settings tuple. Initial short f16/CUDA oracle dump completed on CUDA workstation.
- [x] C++ non-chunked dump produces comparable tensors and can consume canonical diffusion-noise tensors, including initial diffusion noise and per-DPM-step SDE variance noise.
- [ ] Comparison script reports max_abs, mean_abs, RMSE, cosine similarity, finite counts, and shape/missing-stage status. Relative L2 / first-bad frame-layer ranking still pending.
- [x] A short oracle fixture reaches acceptable C++ parity before any long-form chunking claims are made. After switching C++ KugelAudio DPM to canonical `sde-dpmsolver++` and loading canonical variance noise, frame-0/1 latent and connector cosines are >0.99998.
- [x] At least one rolling-KV K-ladder run is compared against the non-chunked oracle and documented. Initial 16-frame q8/f16 CUDA diagnostics with teacher-forced canonical step embeddings show K=8 tracks the oracle through frame 8 and reaches ~0.94 cond cosine by frame 15; K=16/full history is effectively oracle on this fixture; smaller K values degrade sooner.

Testing:
- Unit/integration: dump metadata validation and tensor-shape compatibility checks.
- Manual/End-to-end: if CUDA workstation OOMs, rerun canonical/C++ dumps on a larger RunPod GPU and record hardware/settings in the dump metadata.

### Major Task: Reduce continuous-feedback sensitivity

The non-chunked oracle work shows the LM/KV path can track canonical closely when canonical generated speech embeddings are teacher-forced, but free-running diverges because small diffusion/connector differences are fed back into future LM state. Treat this as a feedback-loop stability problem before promoting any long-form continuity mode.

Follow-up experiments:
1. Step-embedding blend diagnostic: feed `canonical_step_embed * (1 - alpha) + cpp_step_embed * alpha` back into the LM for alpha ladder `0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0` while keeping canonical diffusion noise fixed. Measure how much C++ feedback the oracle trajectory tolerates.
2. Step-embedding stabilizer prototypes if blend suggests a usable tolerance: norm/RMS matching to reference/generated canonical embedding distributions, outlier clamps, per-channel mean/std calibration, or conservative running-distribution blending.
3. Improve remaining base parity: inspect diffusion head, acoustic connector, CUDA/f16 math, and dtype drift. DPM step tracing shows frame-1 divergence accumulates late in SDE steps with canonical noise, but teacher-forcing canonical diffusion cond makes the trace near-exact; the main amplifier is tiny LM hidden/condition error, not scheduler structure or CUDA backend.
4. Use rolling-KV only as an error-cap tradeoff after the feedback tolerance is quantified; do not treat it as a standalone continuity fix.

AC:
- [x] Blend diagnostic runs on the 16-frame f16 oracle and records cond/latent/hidden cosine vs alpha. Result: stable through alpha≈0.85, sharp frame-15 collapse around alpha=0.90.
- [x] Decide whether stabilizing generated step embeddings is plausible, based on the largest alpha that remains close to the non-chunked oracle. Decision: plausible enough to prototype, because alpha=0.85 remains near-oracle on the 16-frame fixture while alpha≥0.90 collapses.
- [x] If plausible, add at least one opt-in stabilizer diagnostic and compare to the blend upper bound. Tried temporal step-embedding smoothing; it failed and is documented as a negative diagnostic, not a recommendation.
- [ ] If not plausible, document that production mitigation should focus on fewer larger independent chunks and/or higher-precision custom kernels rather than feedback shaping. Current result: oracle cond/step blends are promising upper bounds; naive temporal smoothing failed; f16 step-embedding cast is promising for f16 but not q8.

Testing:
- Integration: blend/stabilizer diagnostics remain env-gated and do not alter normal CLI output.
- Manual/End-to-end: compare any stabilizer candidate against the canonical non-chunked oracle and listening samples.

### Major Task: Find the largest clean independent chunk profile

Accept some per-chunk variation but reduce seam count by using larger blocks that remain clean. Prioritize q8/CUDA throughput for fast listening sweeps, then verify promising settings against f16.

Candidate ladder:
- `--max-words-per-chunk 120 --max-frames 96`
- `--max-words-per-chunk 160 --max-frames 128`
- `--max-words-per-chunk 224 --max-frames 160`

Common settings:
- `--chunk-continuity none`
- `--chunking-strategy syntax-aware` or `heuristic` as appropriate
- `--pause-mode punctuation`
- `--crossfade-ms 60`
- `--cfg 1.0`
- `--steps 4` initially; only increase steps if noise appears denoising-limited rather than autoregressive drift

AC:
- [ ] q8/CUDA sweep generates all candidate ladder samples without OOM or crash. First rung (`mw120/f96`) completed with streamed GPU final decode.
- [ ] Listening identifies the largest block size that stays clean enough for long-form use.
- [ ] The selected q8/CUDA profile is rerun with f16 for quality comparison.
- [ ] Docs record the recommended long-form profile and caveats.

Testing:
- Integration: sweep script can run the ladder and write logs/results per sample.
- Manual/End-to-end: listening comparison against prior `none`, `tail-reference`, `latent-prefix`, and `prompt-instruction` samples.

### Major Task: Fix CUDA final acoustic decode so long-form iteration can stay on GPU

The current CUDA workaround uses `--final-decoder-backend cpu` because pure CUDA final decode has crashed with `ggml_cuda_compute_forward: IM2COL failed`. Fix or replace that path before large-block sweeps so iteration is not bottlenecked by CPU decode.

AC:
- [x] Reproduce the pure-CUDA final decoder failure with a minimal logged command and preserve the failing log.
- [x] Determine whether failure is limited to one-shot full-sequence decoder, streamed decoder, specific frame counts, or q8/f16 model memory pressure. Result: pre-fix one-shot hit CUDA grid-Y `IM2COL` launch limit; post-fix short one-shot passes, but long one-shot can OOM due large temporary im2col buffers, so streamed GPU decode remains the default for `auto`.
- [x] Pure CUDA final decode (`--final-decoder-backend active`) succeeds for the short smoke fixture after the im2col fix; `auto` succeeds via streamed GPU decode.
- [x] Pure CUDA or CUDA-streamed final decode succeeds for at least the first long-form block profile without `IM2COL failed`.
- [x] If full CUDA decode remains unsupported, `--final-decoder-backend stream` is documented as the GPU-only workaround and selected automatically where safe.

Testing:
- Integration: `test_kugelaudio_cuda_smoke` and `test_kugelaudio_cuda_conditioning` pass on real CUDA hardware without CPU-fallback-only masking.
- Manual/End-to-end: generate q8 and f16 CUDA samples with final decoder on GPU and inspect logs for backend selection and non-silent output.

### Major Task: Characterize CUDA memory and speed for long-form q8/f16

Use the fixed CUDA decode path to quantify whether q8 enables larger clean blocks and faster iteration.

AC:
- [ ] Record wall-clock, generated duration, WPS/RTF, peak/failure mode if available, and final decoder mode for q8 ladder runs.
- [ ] Repeat the selected profile with f16.
- [ ] Decide whether q8 is only an iteration target or also an acceptable listening target for long-form previews.

Testing:
- Integration: sweep output includes backend/model/final-decoder metadata.
- Manual/End-to-end: compare listenability and iteration speed across q8/f16.

### Major Task: Add transcript coverage QA for generated long-form audio

Chunked long-form output can sound good while silently omitting or cutting off
words near a chunk/frame-budget boundary. Add an optional quality-assurance pass
that transcribes the generated WAV with an external ASR backend (for example
faster-whisper) and compares it against the requested text before recommending a
profile or accepting a sample.

Intended shape:
- Normalize input text and ASR transcript with the same recall helpers used by
  the divergence harness where possible.
- Report word/phrase recall, missing spans, and likely tail truncation.
- For chunked generation, optionally run per-chunk or time-window transcript
  checks so failures can be tied back to chunk boundaries and frame budgets.
- Keep ASR external to the shipped TTS runtime; this is an eval/QA tool, not a
  product dependency.

AC:
- [ ] Add a script or eval-harness mode that accepts `--text-file` and `--audio`
  and emits coverage JSON plus a concise human-readable missing-span summary.
- [ ] Use faster-whisper by default when available, with clear setup/error
  messages if the optional dependency is missing.
- [ ] Detect common long-form failures: missing final words, low transcript
  recall, and chunk-boundary omissions.
- [ ] Document the QA command in the long-form README/docs path.
- [ ] Run the QA pass on the private real-reference samples and use it to
  tune frame budgets, including the observed last-words cutoff.

Testing:
- Unit: text normalization / recall / missing-span helpers on synthetic examples.
- Integration: QA script produces deterministic JSON fields for a fixed fixture
  or mocked ASR transcript.
- Manual/End-to-end: transcribe a generated long-form sample, inspect missing
  spans, and compare against listening notes.

### Major Task: Prototype chunk-boundary cleanup for stitched long-form audio

Long-form chunking can currently produce two independent seam artifacts:
chunks may end while speech is still high-energy and then hard-cut into inserted
silence, while later chunks often contain their own leading silence that stacks
with punctuation pauses. Some chunks can also generate long trailing silence,
creating much larger gaps than the requested pause. Add an opt-in boundary
cleanup pass that operates on generated chunk waveforms before stitching, without
changing prompt/parity behavior by default.

Intended shape:
- Keep default stitching byte-compatible unless explicitly opted in.
- Add an experimental CLI/runtime switch for boundary cleanup.
- For chunks after the first, trim excessive generated leading silence to a
  small target.
- For chunks before the last, trim excessive generated trailing silence to a
  small target.
- Apply short fade-in/fade-out ramps around chunk boundaries so high-energy
  chunk endings do not hard-cut into zeros.
- Preserve verbose boundary diagnostics: generated head/tail quiet, applied
  trims/fades, pause/crossfade, and estimated audible gap.
- Validate first with the current strongest real-reference profile:
  `tail-reference`, `continuity_tail_ms=1200`, f16 CUDA, `cfg=2.0`, generous
  frame budget.

AC:
- [x] Add opt-in CLI controls for chunk-boundary cleanup and silence/fade
  targets; defaults leave output unchanged.
- [x] Reject invalid negative cleanup values with clear CLI/runtime errors.
- [x] `--verbose` reports per-chunk cleanup decisions and post-cleanup boundary
  gap estimates.
- [x] Generate at least one private real-reference stress sample comparing
  cleanup on/off.
- [x] Run ASR coverage on the cleanup sample and ensure it does not regress
  obvious omitted/cut-off text. Current safer cleanup probe matches baseline ASR
  coverage; aggressive cleanup regressed and should not be used as default.
- [x] Probe a simple EOS/speech_end guard. Result: unconditional one/two/four-frame EOS
  suppression generated mostly extra silence and worsened or failed to improve
  coverage/seam gaps; keep it diagnostic-only, not a recommended seam fix.

Testing:
- Unit: waveform cleanup trims only excess leading/trailing silence and applies
  bounded fades without changing disabled output. TODO: add focused synthetic
  unit coverage for the cleanup helper if the mode is promoted beyond diagnostic.
- Integration: seeded chunked CLI output remains byte-identical when cleanup is
  disabled.
- Manual/End-to-end: compare seam smoothness and pause length on the mw40/mw60
  tail-reference stress outputs.

### Major Task: Investigate upstream final-phoneme cut-off behavior

Upstream KugelAudio issue #4 reports rigid end cutoffs in canonical PyTorch,
including missing final Russian sounds and similar English <private-audio-dir> endings:
https://github.com/Kugelaudio/kugelaudio-open/issues/4. Our chunk-boundary
work shows a related local symptom: some chunks stop while decoded tail energy
is still high. Blind stop-token suppression was tested and mostly generated
extra silence, so the next work should target decoded-tail detection and
training-aligned continuation/padding rather than unconditional EOS masking.

Intended shape:
- Reproduce the upstream issue text/reference path in canonical PyTorch when a
  suitable reference is available, and in this C++ runtime with a comparable
  short phrase.
- Add a small tail-cut detector over decoded waveform output: final-window RMS,
  peak, zero-crossing/diff ratio, trailing quiet duration, and optional spectral
  discontinuity proxy.
- Compare detected tail-cut cases against normal natural endings and against
  chunk boundary high-energy tails.
- Experiment with prompt/text end padding that may be training-aligned, e.g.
  terminal punctuation normalization, optional trailing pause marker, or a
  harmless extra delimiter/sentence boundary, while keeping default prompt
  parity unchanged.
- Prototype a targeted retry/continuation path only when the decoded final tail
  looks cut: generate a small continuation budget, decode it, and stitch/fade it
  to the original chunk if it begins with useful speech rather than silence.
- Keep all approaches opt-in diagnostics until they improve listening and ASR
  without causing extra silence, repeated words, or new hallucinated speech.

AC:
- [ ] Add a reproducible short-phrase fixture or script that demonstrates the
  final-phoneme cutoff on at least one reference/text pair.
- [ ] Add tail-cut metrics to verbose output or sidecar metadata for both
  non-chunked and chunked generation.
- [ ] Document why simple EOS/speech_end guard forcing is not recommended,
  including the one/two/four-frame guard results.
- [x] Test one prompt/text padding strategy: terminal ASCII adjacent ellipsis
  (`...`) fixes the confirmed German short cutoff fixtures by listening and
  tail metrics; per-chunk ellipsis regressed long-form ASR, so the implemented
  option pads only the final generation unit in chunked mode.
- [ ] Test one decoded-tail continuation/retry strategy.
- [x] For the ellipsis candidate, run ASR coverage and listening comparison
  against baseline on the short cutoff fixtures and run ASR on a long-form
  chunked sample. Short fixtures pass by listening; long-form final-only padding
  matches baseline ASR.

Testing:
- Unit: tail-cut detector classifies synthetic high-energy endings, trailing
  silence, and normal faded endings correctly.
- Integration: candidate mitigation is disabled by default and leaves seeded
  baseline WAV bytes unchanged.
- Manual/End-to-end: compare final phoneme naturalness and seam smoothness on
  the upstream-style cutoff fixture plus the private-reference tail-reference stress
  profile.

### Major Task: Emit long-form sidecar metadata JSON

Generated long-form WAVs should have a machine-readable sidecar that describes
how they were produced and how they were stitched. This is required for reliable
ASR QA, debugging cutoffs, comparing profiles, and tying listening feedback back
to chunk boundaries.

Intended shape:
- General metadata: schema version, command/settings, model/tokenizer/ref paths
  or hashes, backend/final-decoder mode, seed, cfg, steps, chunking strategy,
  continuity/refinement settings, output sample rate/duration/sample count.
- Chunk metadata: chunk index, prompt text/new text/overlap prefix, boundary
  type, planned word count, generated frame count, generated sample start/length
  before and after stitching, pause/crossfade info, stop reason (`eos`, `budget`,
  error), and optional per-chunk RMS/peak.
- QA metadata slots: ASR transcript, ASR coverage/recall, missing spans, tail
  coverage, speaker-similarity score, and verifier command/results path.

AC:
- [ ] Add a CLI option such as `--metadata-out sample.json` or deterministic
  default sidecar naming next to `--out`.
- [ ] Populate general run metadata for all TTS outputs, including non-chunked
  and single-sequence runs.
- [ ] Populate per-chunk metadata for chunked runs, including start/length and
  stop reason sufficient to diagnose cutoffs.
- [ ] Let the optional ASR verifier update or emit compatible QA fields.
- [ ] Document the sidecar schema and include a real long-form example.

Testing:
- Unit: sidecar JSON schema/field presence for synthetic chunk plans.
- Integration: chunked CLI run writes sidecar with chunk count, sample ranges,
  settings, and output duration matching the WAV.
- Manual/End-to-end: inspect sidecar for the private real-reference runs and
  use it to identify final-word/chunk-boundary truncation.

### Major Task: Prototype latent img2img-style refinement for chunk identity drift

Independent chunk/block generation still re-samples speaker identity even with
f16 step-embedding feedback casting, larger blocks, and moderate CFG. Prototype a
second pass that refines generated speech latents rather than feeding decoded
waveform or arbitrary state forward. The analogy is Stable Diffusion img2img:
start from first-pass chunk latents, add noise according to a low denoise
strength, then run a shortened reverse diffusion pass under the same
text/reference conditioning.

Intended shape:
- first pass generates chunks normally and preserves their diffusion latents
  before final acoustic decoding
- refinement pass per chunk/frame starts from the generated latent, adds
  deterministic scheduler noise at strength `s`, and denoises back to x0 using
  the same diffusion head conditioning
- initial ladder: strengths `0.10`, `0.20`, `0.30`, `0.40`; keep CFG around
  `2.0` if it remains the best identity anchor without overacting
- decode only the refined latent sequence, then stitch as usual
- do **not** use decoded waveform tails, generated reference audio, or hidden/KV
  state carry as conditioning for future chunks

AC:
- [ ] Add an opt-in diagnostic surface, e.g. `--latent-refine-strength` and
  `--latent-refine-steps` or env-gated equivalents, with default disabled.
- [ ] Preserve first-pass latents for chunked KugelAudio generation without
  changing default output when refinement is disabled.
- [ ] Implement deterministic partial-noise + reverse-diffusion refinement using
  the existing canonical SDE DPM solver configuration or a clearly documented
  approximation.
- [ ] Run/listen to a ladder on the long-form fixture, prioritizing
  `mw160/f192/cfg2.0` and `mw224/f256/cfg2.0` at strength `0.20` first.
- [ ] Record whether refinement improves speaker identity, chunk loudness, seam
  harshness, or only acts as volume/timbre polish.

Testing:
- Unit: partial-noise timestep/strength mapping and deterministic seed behavior
  for refinement noise.
- Integration: refinement-disabled output remains byte-identical to baseline;
  refinement-enabled run produces finite non-silent WAVs and logs strength/steps.
- Manual/End-to-end: listening comparison against unrefined `cfg=2.0` large
  block samples and the coherent natural single-sequence sample.

## Slice 9: Optional V2 / deferred roadmap

This slice is explicitly outside the publish-critical path.

### Major Task: Canonical long-text chunking parity

Implement chunk planning/stitching with parity to canonical KugelAudio long-text behavior. For this repo's published CLI surface, land the sentence/clause/hard-wrap planner, sentence-overlap context, pause/crossfade stitching, and a lightweight syntax-aware oversized-sentence planner without adding a heavy NLP dependency.

AC:
- [x] Long-text chunking matches canonical heuristic sentence/clause/hard-wrap behavior closely enough for regression testing.
- [x] Pause/crossfade stitching behavior is tested and documented.
- [x] Sentence-overlap context is supported on the published CLI surface.
- [x] A syntax-aware chunk planning mode is exposed on the published CLI surface.
- [x] The syntax-aware mode remains lightweight/no-extra-dependency rather than attempting full canonical `pysbd` parity.

Testing:
- Unit: chunk planner and stitching helpers.
- Integration: multi-chunk CLI generation with the same raw-reference conditioning reused per chunk.
- Manual/End-to-end: listen to stitched outputs.

### Major Task: Broader quantization and KV-cache work

Expand beyond `f16` parity and `q8_0` execution.

AC:
- [x] Core C++ quantizer exposes family-level overrides for matmul-safe tensor groups on the
  supported KugelAudio path (`attn`, `ffn`, `lm_head`, `embed`, `ac.*`, `sc.*`,
  diffusion-head FFN/MLP blocks, and acoustic transposed-conv FFN linears).
- [ ] Additional quantization formats are named, implemented, and benchmarked.
- [ ] KV-cache quantization is exposed in a controlled, testable way.

Testing:
- Unit: quantization option validation.
- Integration: per-format load/e2e checks.
- Manual/End-to-end: run benchmark/eval comparisons.

### Major Task: Optional external evaluator backends

Keep optional evaluator alternatives out of the publish-critical path while recording plausible future replacements for the chosen `faster-whisper` path.

AC:
- [ ] Optional evaluator backends are clearly marked as stretch goals, not v1 publish requirements.
- [ ] `whisper.cpp` is documented as a possible future external transcript-fidelity backend for the eval harness.
- [ ] Any optional backend preserves the same product boundary: external evaluator only, not a shipped ASR runtime feature.

Testing:
- Unit: n/a until implemented.
- Integration: future harness backend-selection coverage if added.
- Manual/End-to-end: optional backend can be swapped into the harness without changing the KugelAudio runtime surface.

### Major Task: Language hints and multi-speaker support

Add deferred canonical features once v1 parity is stable.

AC:
- [ ] Language hints behave like the canonical implementation.
- [ ] Multi-speaker dialog/ref handling is implemented and tested.

Testing:
- Unit: prompt-building and validation logic.
- Integration: canonical-vs-ggml tests for each feature.
- Manual/End-to-end: feature demos through CLI.
