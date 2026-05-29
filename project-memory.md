# Project memory

## Decisions

### Explicit hybrid final decoder fallback for Vulkan
- Context: on the AMD RADV iGPU path, longer KugelAudio Vulkan runs were not failing during LM/diffusion generation; they were failing later in `decode_latent_sequence(...)` when the final acoustic decoder tried to allocate/upload a large one-shot latent->waveform graph.
- Chosen default: add an explicit opt-in final decoder override (`VibeVoiceTTSParams::final_decoder_backend = Cpu`, surfaced in the CLI as `--final-decoder-backend cpu`) instead of making CPU final decode automatic.
- Rationale: this keeps the scope narrow, preserves the option to pivot later to a fully Vulkan chunked decoder, and makes the backend split explicit at the exact problem boundary (final acoustic decode only).
- Observed reality: a pure Vulkan q8 run on the laptop failed around 37 latent frames during final decode with Vulkan device memory peaking around ~17.3 GiB; the explicit hybrid path completed the same run and wrote ~4.93 s / 118400-sample output.
- Implementation note: the runtime lazily builds a CPU shadow copy of just the acoustic decoder weights from the active backend tensors once per loaded model, then runs the final latent->waveform graph on a local CPU backend. Conditioning, LM, KV cache, and diffusion stay on the active backend.
- Future caution: this is intentionally not a general multi-backend scheduler and not llama.cpp-style layer offload. If a Vulkan chunked decoder lands later, treat this hybrid fallback as a separate, narrower escape hatch rather than the long-term primary design.

### Streamed final decoder on the active backend
- Context: the one-shot final acoustic decoder graph scales with latent-sequence length and was the dominant Vulkan memory spike on the laptop.
- Chosen default: add a second explicit opt-in final decoder mode (`VibeVoiceTTSParams::final_decoder_backend = StreamActive`, surfaced in the CLI as `--final-decoder-backend stream`) that keeps the active backend but chunks the final latent->waveform decode.
- Implementation note: decoder streaming reuses `StreamingCache`, but extends it to cover causal transposed-conv via overlap-add semantics. For a transposed-conv layer with kernel `K` and stride `s`, each chunk emits exactly `T_chunk * s` finalized samples and carries a raw tail of length `K - s` into the next chunk. Bias is applied only after overlap resolution so overlap regions do not get double-biased.
- Implementation note: regular causal convs inside the decoder (`stem`, `Block1D` mixer convs, `head`) reuse the existing `sconv1d_causal_streaming` cache path; only the transposed-conv stages needed new streaming logic.
- Operator knob: `VIBEVOICE_KUGELAUDIO_STREAM_DECODER_FRAMES` controls latent frames per streamed decoder chunk (default `8`) for local tuning/debugging.
- Observed reality: the explicit streamed path completed the same longer Vulkan run that previously crashed, producing ~4.93 s / 118400 samples with `--final-decoder-backend stream`.
- Future caution: on this laptop the Vulkan path is still flaky earlier in model-load/encoder upload under some runs, so a failed smoke before final decode does not by itself implicate the streamed decoder implementation.

### KugelAudio GGUF metadata migration contract
- Context: Slice 1 started with converter-side KugelAudio support while the runtime/loader still keyed off legacy `vibevoice.*` metadata and the GGUF arch string used by existing code.
- Chosen default at that stage: emit a dual metadata contract for KugelAudio conversion:
  - new explicit `kugelaudio.*` keys, with `kugelaudio.checkpoint = kugelaudio-0-open` and `kugelaudio.schema_version = 1`
  - legacy `vibevoice.*` compatibility keys in the same GGUF during migration
  - keep GGUF writer arch as `vibevoice` until loader migration work lands
- Status: transitional decision; the publish-cleanup plan now targets writing new KugelAudio artifacts with `general.architecture = kugelaudio` while optionally keeping read-only compatibility for older `vibevoice`-tagged KugelAudio GGUFs.
- Rejected alternatives:
  - emit only `kugelaudio.*` immediately: cleaner, but likely broke the loader before Slice 1 loader work landed
  - keep only `vibevoice.*`: preserved compatibility, but failed the spec/PRD requirement for an explicit KugelAudio metadata contract
  - switch GGUF arch to `kugelaudio` immediately at the very start: desirable later, but was too risky before loader support existed
- Affected area: `spec.md` GGUF schema migration guidance and `prd.md` Slice 1 / “Define and implement the KugelAudio GGUF contract”.

### KugelAudio published artifact identity target
- Context: for branch publication, continuing to emit fresh KugelAudio artifacts that still identify as `vibevoice` would preserve migration compatibility but undermine the repo's published identity.
- Chosen default: treat `kugelaudio` as the write-path identity for newly produced KugelAudio artifacts (converter + quantizer), while deciding separately whether loader compatibility for older `vibevoice`-tagged KugelAudio GGUFs remains temporarily supported.
- Rejected alternatives:
  - keep writing `vibevoice` indefinitely for all KugelAudio artifacts: easiest short-term, but leaves the published branch mis-branded
  - drop all legacy read compatibility at the same time as the write-side switch: cleaner, but riskier than a one-way migration boundary
- Affected area: `prd.md` Slice 7 / "Canonicalize the published GGUF/runtime identity as KugelAudio".

### KugelAudio runtime-loader variant normalization
- Context: the current runtime only has behavior branches for `realtime-0.5b`, `1.5b`, and `asr-7b`, but converter-side KugelAudio metadata now identifies the checkpoint as `kugelaudio-0-open`.
- Chosen default: when loader metadata is KugelAudio-only, normalize `kugelaudio-0-open` onto the existing `1.5b` runtime branch for now, while still validating `kugelaudio.schema_version` and `kugelaudio.checkpoint` explicitly.
- Rejected alternatives:
  - add a brand-new runtime variant in this iteration: broader than the current PRD increment and under-tested without the later prompt/inference work
  - map KugelAudio onto `realtime-0.5b`: a worse architectural fit because the raw-reference conditioning path and encoder usage are closer to the current 1.5b path
  - leave the variant unnormalized: would bypass existing branch logic and fail to load any integrated runtime state
- Affected area: `prd.md` Slice 1 / loader compatibility work and upcoming Slice 2/3 runtime adaptation.

### KugelAudio schema documentation location
- Context: the next PRD item requires the GGUF schema to be documented clearly enough for future validation work, but the repo already has `docs/conversion.md` as the conversion-facing document and no existing dedicated schema doc.
- Chosen default: extend `docs/conversion.md` with a KugelAudio-specific schema contract section instead of creating a separate `docs/gguf-schema.md` right now.
- Rejected alternatives:
  - create a brand-new schema-only doc immediately: cleaner separation, but adds another place to keep in sync during rapid migration
  - document only in converter source comments: too discoverability-poor for future agents working from docs first
- Affected area: `prd.md` Slice 1 / "The schema is documented well enough that a future agent can add validation without re-deriving intent.".

### KugelAudio converter required-tensor gate
- Context: the next PRD increment requires proving that all tensors needed by the v1 TTS path are either mapped or rejected clearly, but the spec does not enumerate the exact converter-side minimum set.
- Chosen default: use the current raw-reference TTS loader path (`vibevoice_load` normalized onto the existing `1.5b` branch) as the authoritative required tensor set for converter validation.
- Rejected alternatives:
  - validate against every remappable tensor in the checkpoint: too broad for the v1 acceptance path and harder to keep aligned with runtime reality
  - validate only metadata plus a few anchor tensors: too weak to guarantee end-to-end TTS loadability
- Affected area: `prd.md` Slice 1 / "Map all required TTS tensors and metadata".

### KugelAudio semantic metadata fallback
- Context: the published KugelAudio config JSON does not serialize `semantic_tokenizer_config`, even though the canonical Python config class materializes semantic-tokenizer defaults and the v1 raw-reference path requires semantic conditioning.
- Chosen default: derive semantic GGUF metadata from canonical default semantic-tokenizer behavior when the config omits `semantic_tokenizer_config`, using the same ratios/depths and `vae_dim=64` shape expected by the shipped open checkpoint.
- Rejected alternatives:
  - require `semantic_tokenizer_config` to be present in JSON: would reject the actual published checkpoint format
  - omit semantic metadata entirely when the config omits it: would make future validation and loader behavior more ambiguous even though semantic tensors are required
- Affected area: `prd.md` Slice 1 / semantic-conditioning tensor and metadata coverage.

### KugelAudio missing-tensor diagnostics format
- Context: the converter already failed on missing required tensors, but the next PRD increment requires those failures to be actionable rather than a flat truncated list.
- Chosen default: group missing required tensors by runtime-relevant family (`lm`, `diffusion_head`, `acoustic_decoder`, `acoustic_encoder`, `semantic_encoder`, `semantic_connector`, etc.) and include both a family summary and a short example list in the error.
- Rejected alternatives:
  - emit the full raw missing-tensor list only: technically complete, but too noisy for large failures
  - report only the first missing tensor: compact, but not actionable enough for converter/debug work
- Affected area: `prd.md` Slice 1 / "Missing required tensors fail fast with actionable diagnostics.".

### Loader submodule failure anchors
- Context: the next loader increment requires semantic/acoustic submodule failures to be clear, but the existing runtime mostly fails later inside generic tensor-loading helpers.
- Chosen default: add explicit preflight checks for representative anchor tensors per submodule (`at.dec`, `at.enc`, `st.enc`, `sc.*`, `ac.*`) before deeper loading, so failures name the missing submodule and example required tensors.
- Rejected alternatives:
  - rely only on lower-level `load_encoder` / `load_decoder` failures: works eventually, but error messages are less obviously tied to the missing high-level submodule
  - enumerate every tensor name in the loader error: too noisy compared with a targeted submodule-level diagnosis
- Affected area: `prd.md` Slice 1 / "Missing semantic/acoustic submodules fail clearly.".

### KugelAudio loader explicit-metadata gate
- Context: the remaining loader item requires behavior not to depend on implicit old VibeVoice defaults, but the current metadata readers still allow silent fallback defaults for missing KugelAudio numeric fields.
- Chosen default: add an explicit KugelAudio metadata preflight that requires the canonical schema keys needed by the current load path before any config defaults are applied.
- Rejected alternatives:
  - keep relying on numeric fallback defaults (`64`, `24000`, etc.): simpler, but makes missing-schema cases load ambiguously
  - require every possible legacy and canonical key pair: over-constrains migration when canonical `kugelaudio.*` alone is sufficient
- Affected area: `prd.md` Slice 1 / "Loader behavior is deterministic and does not depend on implicit old VibeVoice defaults.".

### KugelAudio v1 feature-gating scope
- Context: the next PRD item requires unsupported features to fail before inference starts, but this repo still carries legacy VibeVoice 1.5B multi-speaker tests and surfaces that should not be broken while adding KugelAudio-specific gating.
- Chosen default: apply v1 feature gates only when the loaded model carries KugelAudio metadata, not to every runtime path normalized onto the existing `1.5b` branch.
- Rejected alternatives:
  - gate all `1.5b` models identically: simpler, but would regress legacy VibeVoice 1.5B capabilities and tests
  - postpone all gating to CLI only: too late for the PRD item, which requires rejection before inference starts in the runtime path
- Affected area: `prd.md` Slice 1 / "Unsupported features are rejected before inference starts.".

### KugelAudio loader detection log payload
- Context: the remaining loader item requires logs to say what was detected and what is enabled, but the repo already had a generic post-load shape log and no settled format for KugelAudio-specific state.
- Chosen default: emit one structured `VV_LOG_INFO` line after successful KugelAudio load that names the detected checkpoint, normalized runtime path, schema version, and the main v1 feature gates (`raw_ref_single_speaker`, `semantic_conditioning`, `pre_baked_voice`, `multi_speaker_dialog`).
- Rejected alternatives:
  - spread the same information across several logs: harder to assert in tests and noisier for operators
  - log every minor config field again: redundant with the existing shape/scaling summary
- Affected area: `prd.md` Slice 1 / "Logs state what was detected and what is enabled.".

### KugelAudio prompt-builder split
- Context: Slice 2 requires prompt-format parity with canonical KugelAudio, but the existing `build_prompt_15b(...)` helper still carries legacy VibeVoice 1.5B multi-speaker semantics and extra `<|vision_start|>...<|vision_end|>` wrapping in the voice-input section.
- Chosen default: keep the legacy 1.5B prompt builder for non-KugelAudio paths and add a dedicated single-speaker KugelAudio v1 prompt builder that mirrors the canonical processor sections exactly.
- Rejected alternatives:
  - replace the existing 1.5B builder for all callers: riskier because it could silently regress legacy VibeVoice-specific tests and flows
  - keep one builder with many conditional branches: workable, but less readable than an explicit split while prompt semantics are still diverging
- Affected area: `prd.md` Slice 2 / "Prompt format matches canonical KugelAudio sections...".

### KugelAudio section-wise token assembly
- Context: after splitting the prompt builders, the KugelAudio path still tokenized one big prompt string and then searched the resulting token stream for placeholder IDs, which preserved some old VibeVoice-style assumptions about prompt/token coupling.
- Chosen default: assemble KugelAudio prompt tokens section-by-section using the canonical processor semantics, inserting speech placeholder IDs (`<|vision_pad|>`) and speech-start IDs (`<|vision_start|>`) directly instead of relying on a full-string encode-and-scan pass.
- Rejected alternatives:
  - keep the full-string encode + scan approach: simpler short-term, but keeps token placement dependent on implicit tokenizer behavior rather than explicit KugelAudio prompt semantics
  - fully replace legacy 1.5B token assembly too: broader migration risk than needed for the current KugelAudio-only PRD increment
- Affected area: `prd.md` Slice 2 / "Prompt/tokenization logic is driven by KugelAudio semantics, not old VibeVoice assumptions.".

### KugelAudio strict single-speaker input policy
- Context: the next Slice 2 item requires single-speaker input to be the only supported v1 mode, but the canonical processor will happily preserve `Speaker 0:`-prefixed text and the repo still carries legacy multi-speaker prompt codepaths.
- Chosen default: for KugelAudio v1, reject any explicit `Speaker N:`-tagged dialog text at runtime, including `Speaker 0:` input, and require plain untagged text plus exactly one raw reference audio.
- Rejected alternatives:
  - allow `Speaker 0:` but reject only `Speaker 1+`: technically workable, but still leaves speaker-tagged dialog semantics in the supported surface
  - silently strip speaker tags back to plain text: too magical and can hide user mistakes when comparing against canonical behavior
- Affected area: `prd.md` Slice 2 / "Single-speaker input is the only supported path in v1 and is enforced explicitly.".

### KugelAudio speech special-token contract
- Context: the next Slice 2 sub-task requires special token IDs and placeholder semantics to match the supported checkpoint, but the current runtime carried the IDs only as local constants in `vibevoice_tts.cpp`.
- Chosen default: treat the canonical KugelAudio text-tokenizer mapping as the contract: `<|vision_start|>=151652`, `<|vision_end|>=151653`, `<|vision_pad|>=151654`, and `<|image_pad|>=151655`, with the prompt inserting only diffusion placeholders in the voice section and a single speech-start token in the output section.
- Rejected alternatives:
  - leave the mapping implicit in scattered local constants only: works, but harder to validate and easier to drift accidentally
  - insert `<|vision_end|>` into the single-speaker voice-input section like legacy VibeVoice 1.5B: diverges from the canonical KugelAudio processor semantics
- Affected area: `prd.md` Slice 2 / "Special token IDs and placeholder semantics match the supported KugelAudio checkpoint.".

### KugelAudio placeholder-position test fixture
- Context: the next prompt sub-task requires placeholder positions to be stable and testable for fixed inputs, but the repo does not ship a reusable tokenizer GGUF fixture by default.
- Chosen default: use a tiny synthetic byte-level tokenizer GGUF fixture for the prompt-position unit test, with the canonical speech special tokens registered at the Kugelaudio IDs.
- Rejected alternatives:
  - depend on a full downloaded production tokenizer fixture: heavier and less hermetic for a small prompt-layout test
  - test only logical placeholder counts without token positions: too weak for the PRD item
- Affected area: `prd.md` Slice 2 / "Placeholder positions are stable and testable for fixed inputs.".

### KugelAudio CFG negative-token seed
- Context: the remaining special-token PRD item requires removing VibeVoice-only token-role assumptions where KugelAudio differs. The current C++ CFG path still seeded the negative branch from a full prompt with `<|image_pad|>` replacements, while canonical KugelAudio inference seeds the negative branch from a single speech-start token.
- Chosen default: for KugelAudio only, initialize the CFG negative branch from one `speech_start` token and keep the legacy image-pad placeholder path for non-KugelAudio 1.5B behavior.
- Rejected alternatives:
  - keep using the image-pad placeholder prompt for KugelAudio too: easier, but encodes a VibeVoice-specific prompt-token role that canonical KugelAudio does not use
  - rewrite legacy VibeVoice 1.5B CFG in the same increment: broader than needed for the current KugelAudio-focused PRD item
- Affected area: `prd.md` Slice 2 / "The runtime no longer assumes VibeVoice-only prompt token roles where KugelAudio differs.".

### Shared KugelAudio single-reference gate
- Context: the next raw-reference item requires the supported single-reference path to be accepted through the runtime/CLI surface, but the validation currently lives only inside the runtime implementation.
- Chosen default: expose the KugelAudio v1 single-reference request validator through a shared helper used by runtime, CLI, and C API so the accepted shape and rejected shapes stay aligned.
- Rejected alternatives:
  - duplicate the validation logic in each frontend: simple short-term, but prone to drift from the runtime gate
  - test only the runtime call path: misses the PRD requirement that the currently supported public surfaces reject the same unsupported conditioning modes explicitly
- Affected area: `prd.md` Slice 2 / conditioning-mode validation and rejection behavior.

### CLI conditioning wiring verification hook
- Context: the next PRD item requires proving the accepted single-reference path is wired end-to-end through CLI -> preprocessing -> conditioning, but the lightweight synthetic KugelAudio fixture is not numerically valid for full generation.
- Chosen default: add a test-only environment hook that returns success immediately after KugelAudio conditioning features are prepared, so the CLI integration test can verify the accepted path reaches preprocessing + conditioning without needing a real model checkpoint.
- Rejected alternatives:
  - require a real 7B checkpoint for this slice: stronger, but too heavy for routine local validation of a wiring-only increment
  - test only the runtime helper and not the CLI binary: weaker than the PRD item, which explicitly mentions CLI wiring
- Affected area: `prd.md` Slice 2 / "Single reference path is fully wired through CLI -> preprocessing -> conditioning.".

### Reference-audio resampling validation point
- Context: the next raw-reference item requires proving that reference audio is resampled internally to 24 kHz mono, and the runtime already funnels that through `load_wav_24k_mono`.
- Chosen default: validate the behavior at `load_wav_24k_mono` directly in `test_audio_io.cpp`, since that is the shared convergence point used by the KugelAudio reference-audio path.
- Rejected alternatives:
  - add a heavier end-to-end TTS test just for resampling: broader and slower than needed for this focused acceptance item
  - leave resampling covered only indirectly by smoke tests: weaker signal if the I/O boundary regresses
- Affected area: `prd.md` Slice 2 / "Audio is resampled internally to 24 kHz mono.".

### Acoustic+semantic conditioning fusion point
- Context: the next raw-reference item requires proving that both acoustic and semantic conditioning are used, and the current KugelAudio path combined them inline in `tts_15b_generate`.
- Chosen default: extract the feature fusion into a shared helper and unit-test that helper directly, so the acceptance check proves both branches contribute to the final speech conditioning features.
- Rejected alternatives:
  - rely only on code inspection of the inline sum: not strong enough for a repeatable acceptance check
  - add a heavy end-to-end audio-quality regression just for this item: broader and noisier than needed for the current slice
- Affected area: `prd.md` Slice 2 / "Both acoustic and semantic conditioning are used for the supported path.".

### KugelAudio v1 limitation wording
- Context: the next conditioning-mode item requires the error text to explain the v1 limitation, not just reject invalid inputs.
- Chosen default: standardize the shared KugelAudio conditioning error strings around one explicit supported-shape sentence: `KugelAudio v1 supports only single-speaker TTS with exactly one raw reference audio input and plain untagged text`.
- Rejected alternatives:
  - keep short per-case rejections only: accurate, but weaker at telling operators what *is* supported
  - push the explanatory wording into CLI docs only: too weak for runtime/C API callers who only see the returned error/log text
- Affected area: `prd.md` Slice 2 / "Error text explains the v1 limitation.".

### KugelAudio speech-token constraint source
- Context: the first Slice 3 parity item requires constraining LM decisions to the canonical speech-path token set, but the current runtime does not yet load a dedicated EOS token ID from tokenizer/model metadata.
- Chosen default: hardcode the supported KugelAudio v1 token quartet from the canonical open checkpoint for now: `speech_start=151652`, `speech_end=151653`, `speech_diffusion=151654`, and `eos=151643`, and use that set when selecting speech-path tokens in the 1.5B KugelAudio loop.
- Rejected alternatives:
  - derive EOS from whatever tokenizer happens to be loaded at runtime: cleaner later, but underconstrained in the current C++ load path and risks drifting from the checkpoint contract being ported
  - defer token constraining until the full canonical token-by-token loop lands: broader delay than needed for this focused parity increment
- Affected area: `prd.md` Slice 3 / "The runtime constrains generation to the canonical speech-path token set.".

### KugelAudio control-token loop guard
- Context: aligning CFG control flow with canonical KugelAudio requires honoring `speech_start` as a non-audio control token that can reset the negative branch before the next diffusion step, but the current C++ runtime bounds generation by speech frames rather than total token steps.
- Chosen default: allow up to 32 consecutive KugelAudio control-token steps without producing a speech frame before failing clearly, which preserves the frame-based API while preventing accidental infinite loops if the LM keeps emitting `speech_start`.
- Rejected alternatives:
  - leave control-token iterations unbounded: simpler, but risks a hung generation loop on a bad model state
  - redesign the public API around a separate max-token budget in this slice: cleaner long-term, but broader than the current CFG-parity increment
- Affected area: `prd.md` Slice 3 / "CFG behavior is aligned with the canonical implementation for the supported path.".

### KugelAudio speech-end penalty default
- Context: the canonical PyTorch generation loop subtracts `speech_end_penalty=1.5` from the constrained `speech_end` logit before selecting the next speech-path token, but the current C++ runtime had no corresponding knob or metadata source.
- Chosen default: hardcode the canonical default penalty value `1.5f` for the supported KugelAudio v1 path and apply it before constrained token selection.
- Rejected alternatives:
  - leave the penalty at zero until a public setting is plumbed: simpler, but knowingly diverges from canonical stop behavior
  - add a new public API/CLI parameter in this slice: potentially useful later, but broader than the current parity-only increment
- Affected area: `prd.md` Slice 3 / "Speech-end behavior is aligned with canonical handling and terminates correctly.".

### KugelAudio final-decode validation scope
- Context: the next Slice 3 item requires proving the final waveform decode path produces usable output, but the lightweight synthetic KugelAudio GGUF fixture only carries zero-valued placeholder tensors and is not numerically valid for decoder execution.
- Chosen default: validate this acceptance item with a real-model gated smoke test that runs the supported KugelAudio path end-to-end through final latent decode and checks for non-empty, finite, non-silent waveform output.
- Rejected alternatives:
  - pretend the synthetic fixture proves decoder usability: fast, but misleading because its tensors are not shape/value-valid for real decoder execution
  - defer all decode validation to the later divergence harness: too late for this slice's explicit acceptance item
- Affected area: `prd.md` Slice 3 / "Final waveform decode path is integrated and produces usable output.".

### KugelAudio reused-component validation scope
- Context: the next reuse item asks whether the existing diffusion/decoder/connector pieces are still the implementation path for KugelAudio where behavior remains valid, but only diffusion (`test_dpm_solver`) and decoder (`test_acoustic`) already had direct low-level coverage.
- Chosen default: keep reusing the existing `dpm_solver_sample`, acoustic decoder, and speech-connector codepaths, and close the test gap by adding a small connector unit test rather than introducing a KugelAudio-specific connector fork.
- Rejected alternatives:
  - add a separate KugelAudio-only connector implementation for easier testing: broader and contrary to the slice goal of preserving safe reuse
  - rely on end-to-end smokes alone: weaker signal when the reuse claim is specifically about low-level shared components
- Affected area: `prd.md` Slice 3 / "Existing diffusion/decoder/connector code is reused where behavior remains valid.".

### KugelAudio parity-justification location
- Context: the next Slice 3 item requires any remaining canonical divergences to be either corrected or explicitly justified, and the existing rationale was scattered across tests and project-memory notes.
- Chosen default: add a dedicated repo doc (`docs/kugelaudio-parity.md`) plus short source comments at the reused solver/connector/decoder sites, so future agents can find both the justification and the exact canonical reference quickly.
- Rejected alternatives:
  - keep the justifications only in `project-memory.md`: useful during execution, but too easy to miss as long-term repo documentation
  - spread the rationale across many inline comments only: discoverable while reading code, but harder to audit as a parity checklist
- Affected area: `prd.md` Slice 3 / "Any divergence from canonical behavior is either corrected or explicitly justified.".

### KugelAudio reuse-coverage granularity
- Context: the next Slice 3 item requires reused code paths to stay covered by targeted tests, but the existing diffusion and decoder parity tests depend on optional fixture files and do not directly assert that the KugelAudio generation path is still wiring the canonical solver defaults.
- Chosen default: add a lightweight non-fixture unit test for KugelAudio solver-config wiring and run it alongside the focused connector/token tests, while continuing to rely on the existing acoustic and DPM solver parity tests for deeper math coverage when fixtures are available.
- Rejected alternatives:
  - add another heavy end-to-end KugelAudio run for this item: broader and flakier than needed to prove reuse wiring
  - treat the existing optional-fixture tests as sufficient on their own: weaker because they do not directly guard the KugelAudio-specific solver setup point
- Affected area: `prd.md` Slice 3 / "Reused code paths remain covered by targeted tests.".

### KugelAudio CLI end-to-end validation scope
- Context: the next Slice 3 CLI-demo item requires proving the supported single-speaker raw-reference path runs end-to-end through the CLI, but the lightweight synthetic fixture is only suitable for wiring checks and not for real generation.
- Chosen default: validate this item with a real-model gated CLI smoke test that exercises `vibevoice-cli tts` on the supported KugelAudio path and then checks that the emitted WAV is loadable, non-empty, and non-silent.
- Rejected alternatives:
  - reuse the conditioning short-circuit test as proof of end-to-end CLI behavior: too weak because it stops before real generation and final decode
  - require an always-on heavyweight model fixture in the repo: stronger, but outside the current repo/testing constraints
- Affected area: `prd.md` Slice 3 / "CLI runs the supported single-speaker raw-reference path end-to-end.".

### KugelAudio CLI unsupported-feature coverage scope
- Context: the next CLI-demo item requires unsupported flags/features to be rejected clearly, but the CLI surface mixes generic validation with KugelAudio-specific v1 limits.
- Chosen default: explicitly cover the three highest-value unsupported KugelAudio CLI cases with synthetic fixtures: removed `--voice` / pre-baked voice conditioning, multiple `--ref-audio`, and speaker-tagged dialog text.
- Rejected alternatives:
  - try to exhaustively test every malformed CLI combination in this increment: broader than the current PRD item and duplicates generic argument-parsing coverage
  - rely only on runtime/CAPI gating tests: weaker because the item is specifically about the CLI surface rejecting unsupported flags/features clearly
- Affected area: `prd.md` Slice 3 / "Unsupported flags/features are rejected clearly.".

### KugelAudio CLI waveform-validity validation point
- Context: the next CLI-demo item requires proving the generated output is a valid waveform file, and merely reloading it through `load_wav_24k_mono(...)` is useful but indirect.
- Chosen default: strengthen the real-model gated CLI end-to-end smoke test to validate the emitted file at the container boundary too: RIFF/WAVE signature, `fmt ` chunk presence, PCM16 encoding, mono channel count, and 24 kHz sample rate.
- Rejected alternatives:
  - add a separate standalone WAV-writer unit test only: helpful, but weaker for the CLI-specific acceptance item
  - trust successful reload alone as sufficient proof: simpler, but less explicit about the on-disk waveform contract
- Affected area: `prd.md` Slice 3 / "Generated output is written as a valid waveform file.".

### KugelAudio CPU determinism assertion shape
- Context: the next seeded-runtime item requires CPU determinism suitable for regression testing, but the repo supports multiple backends and the strongest practical assertion level is not spelled out.
- Chosen default: force `VIBEVOICE_BACKEND=cpu` inside the dedicated determinism test and require exact sample-vector equality across two runs with the same model/text/reference/seed/settings.
- Rejected alternatives:
  - allow approximate equality only: weaker than needed for regression-oriented CPU determinism
  - test whichever backend happens to initialize first: ambiguous, because the PRD item is explicitly about CPU eval determinism
- Affected area: `prd.md` Slice 3 / deterministic seeded runtime behavior on CPU.

### KugelAudio CLI seed-plumbing proof point
- Context: the next seeded-runtime item requires seed plumbing to be exposed end-to-end through the CLI/eval path, but the repo does not yet have the later eval harness and the CLI already accepts `--seed`.
- Chosen default: prove the plumbing at the current supported public surface by adding a CLI test that runs `vibevoice-cli tts` twice with the same `--seed` on CPU and requires byte-identical emitted WAVs.
- Rejected alternatives:
  - wait for the future eval harness before testing seed plumbing: too late for the current PRD item
  - test only parser/default handling without generating audio: weaker than an end-to-end CLI proof
- Affected area: `prd.md` Slice 3 / "Seed plumbing is exposed end-to-end through the CLI/eval path.".

### KugelAudio determinism documentation boundary
- Context: the next PRD item asks for documentation of known nondeterministic cases, but the code now has both a supported deterministic CPU regression path and broader backend/seed modes that are not regression targets.
- Chosen default: document the deterministic contract and the remaining non-guaranteed cases together in `docs/kugelaudio-parity.md`, rather than scattering caveats across tests only.
- Rejected alternatives:
  - document only the negative cases: less useful because future evaluators also need to know what *is* promised deterministic
  - add a separate determinism-only doc immediately: cleaner in isolation, but unnecessary overhead while the parity note already tracks Slice 3 behavior
- Affected area: `prd.md` Slice 3 / "Known nondeterministic cases are documented if any remain.".

### External evaluator stack for published acceptance
- Context: the publish-cleanup slices require replacing internal ASR-based acceptance without re-introducing a shipped ASR product surface.
- Chosen default: move the eval harness to external evaluators only — `faster-whisper` for transcript generation / transcript-recall scoring and SpeechBrain ECAPA-TDNN for speaker-similarity cosine scoring. Keep both dependencies in the Python eval harness rather than the runtime/CLI.
- Rejected alternatives:
  - keep reusing the repo ASR runtime internally: conflicts with the KugelAudio-only published surface and keeps ASR entangled with acceptance
  - switch immediately to `whisper.cpp`: still a viable future evaluator backend, but slower to wire into the current Python harness; recorded as a stretch goal in `prd.md`
  - drop transcript-fidelity checks entirely: too weak for acceptance because it loses the “did it say the requested words?” guardrail
- Affected area: `scripts/eval_kugelaudio_divergence.py`, `tests/fixtures/kugelaudio_eval_config.json`, `docs/kugelaudio-parity.md`, `prd.md` Slice 4 / Slice 7.

### Removed pre-baked voice publish surface
- Context: the branch publish target is KugelAudio/raw-reference TTS, while the old realtime VibeVoice `voice.gguf` path keeps leaking into CLI help, docs, and scripts.
- Chosen default: remove pre-baked `voice.gguf` from the published CLI/docs/test surface now: keep `--voice` only as an explicit rejection with a removal message, delete `scripts/convert_voice_to_gguf.py`, stop registering the legacy voice-dependent large tests, and remove runnable legacy operator quickstarts from `README.md`.
- Rejected alternatives:
  - keep the old flow documented as “legacy”: still too discoverable for a publish-cleanup branch and continues to suggest a supported surface that no longer matches the target
  - rip out all lower-level voice-loading internals immediately: broader churn than needed for this publish-surface step; internal/runtime cleanup can continue separately
- Affected area: `examples/cli/main.cpp`, `README.md`, `docs/conversion.md`, `docs/cuda.md`, `tests/CMakeLists.txt`.

### Removed ASR from the published CLI/docs/default-test surface
- Context: after moving acceptance to external evaluators, the remaining `vibevoice-cli asr` command and default ASR docs/tests still advertised a repo-facing product surface that Slice 7 is explicitly trying to remove.
- Chosen default: remove the `asr` subcommand from `vibevoice-cli`, delete ASR references from README/CUDA/operator docs, and unregister ASR-only default tests; keep lower-level ASR/CAPI code only as explicitly marked legacy/internal compatibility.
- Slice-7 completion boundary: treat the PRD item "ASR is removed as a published/runtime capability" as satisfied once operator-facing CLI/docs/default tests no longer expose ASR and the remaining entrypoints are clearly quarantined as legacy/internal compatibility.
- Rejected alternatives:
  - keep `vibevoice-cli asr` as a hidden or undocumented command: still leaves a discoverable repo-facing runtime capability that conflicts with the published KugelAudio scope
  - hard-delete all ASR internals/CAPI immediately: larger churn than needed for this publish-surface increment and riskier while compatibility shims still exist
- Affected area: `examples/cli/main.cpp`, `README.md`, `docs/cuda.md`, `docs/conversion.md`, `AGENTS.md`, `tests/CMakeLists.txt`, `include/vibevoice.h`, `include/vibevoice_capi.h`.

### Flattened published CLI shape
- Context: once `tts` became the only substantial repo-facing subcommand, keeping `tts`, `help`, and `version` as subcommands added extra publish-surface noise without adding product value.
- Chosen default: collapse the published CLI to a single top-level TTS interface: `vibevoice-cli --help`, `vibevoice-cli --version`, or `vibevoice-cli --model ... --tokenizer ... --ref-audio ...`.
- Rejected alternatives:
  - keep `tts` as a compatibility subcommand: unnecessary extra surface for the KugelAudio-only published path
  - preserve `help` / `version` subcommands in addition to flags: redundant once the CLI is no longer multi-command
- Affected area: `examples/cli/main.cpp`, `README.md`, `docs/cuda.md`, `docs/kugelaudio-parity.md`, `tests/test_kugelaudio_cli_*.cpp`.

### Renamed shared speech helper surface
- Context: `src/vibevoice_speech_helpers.hpp` still framed reusable speech-conditioning helpers as part of a VibeVoice/ASR-owned surface, which conflicted with the Slice 7 goal of quarantining ASR as legacy/internal compatibility.
- Chosen default: rename the header to `src/speech_conditioning_helpers.hpp` and rewrite its comments to present it as neutral shared speech-conditioning infrastructure reused by raw-reference TTS and legacy/internal compatibility code.
- Rejected alternatives:
  - leave the old header name and rely on comments only: weaker quarantine, since the file path itself still leaked the old surface
  - move the implementations out of `src/vibevoice_asr.cpp` in the same pass: broader churn than needed for this increment
- Affected area: `src/speech_conditioning_helpers.hpp`, `src/vibevoice_tts.cpp`, `src/vibevoice_asr.cpp`, `tests/test_kugelaudio_conditioning.cpp`, `AGENTS.md`.

### Backend selection/logging contract
- Context: Slice 6 needs backend choice to be inspectable and reproducible across local runs and the eval harness, especially when GPU availability differs between machines.
- Chosen default: backend init now logs `requested=... device_index=... selected=... reason=...`, `available_devices=...`, and `flash-attn=...` on startup. Missing/incompatible requests fall back cleanly with an explicit reason.
- Additional runtime knobs: `VIBEVOICE_BACKEND_DEVICE_INDEX` selects a matching device by index; `VIBEVOICE_BACKEND_VERBOSE=1` includes device descriptions/memory in the startup device list.
- Eval harness boundary: `scripts/eval_kugelaudio_divergence.py` already treats `ggml_backend` as part of the plan/results contract and allows `--ggml-backend` override without changing the product CLI surface.
- Affected area: `src/backend.cpp`, `src/backend.hpp`, `tests/test_backend_selection.cpp`, `docs/cuda.md`, `docs/kugelaudio-parity.md`, `prd.md` Slice 6.

### CUDA smoke/conditioning coverage shape
- Context: Slice 6 item #2 needs CUDA-specific smoke coverage without changing the published CLI surface or requiring CUDA on every developer machine.
- Chosen test split: `test_kugelaudio_cuda_smoke` uses `VIBEVOICE_KUGELAUDIO_Q8_MODEL` for end-to-end q8_0 generation on CUDA; `test_kugelaudio_cuda_conditioning` uses `VIBEVOICE_KUGELAUDIO_MODEL` and short-circuits after the conditioning connector stage to validate encoder + connector execution on CUDA.
- Added internal test hook: `VIBEVOICE_KUGELAUDIO_TEST_STOP_AFTER_CONNECTORS=1` returns early after acoustic/semantic connector fusion with a log marker, mirroring the existing earlier stop-after-conditioning hook but exercising more of the GPU path.
- Skip contract: both CUDA tests return `77` when no CUDA device is registered or the expected acceptance-path env vars are absent.
- Affected area: `src/vibevoice_tts.cpp`, `tests/test_kugelaudio_cuda_smoke.cpp`, `tests/test_kugelaudio_cuda_conditioning.cpp`, `tests/CMakeLists.txt`, `docs/cuda.md`.

### Long-text chunking scope landed from the canonical implementation
- Context: Slice 8 now needs a practical long-text path, but the published repo surface should stay smaller than the canonical Python utility's full chunking control set.
- Canonical reference followed first: `../kugelaudio-open/src/kugelaudio_open/utils/chunking.py`, `../kugelaudio-open/src/kugelaudio_open/utils/stitching.py`, `../kugelaudio-open/src/kugelaudio_open/utils/generation.py`, and `../kugelaudio-open/spec.md`.
- Chosen subset for this repo: implement the canonical heuristic planner shape (sentence split -> clause fallback -> hard-wrap), optional `overlap_sentences` prompt carry-over, punctuation/speaker-aware pause insertion, linear crossfade stitching, and a lightweight `syntax-aware` mode that uses the canonical conjunction-aware phrase-splitting regex for oversized sentences.
- Published CLI controls added: `--max-words-per-chunk`, `--overlap-sentences`, `--chunking-strategy`, `--pause-mode`, and `--crossfade-ms`.
- Runtime shape: chunking applies only to KugelAudio raw-reference TTS, reuses the same reference conditioning independently per chunk, bypasses stitching for single-chunk inputs, and logs chunk-count/boundary-type diagnostics including overlap and chunking-strategy settings.
- Intentional non-parity boundary: unlike canonical Python, this repo does not add an optional `pysbd` sentence segmenter; `syntax-aware` here is dependency-free and only changes the oversized-sentence fallback planner.
- Test coverage: `tests/test_kugelaudio_chunking.cpp` covers planner + stitching helpers; `tests/test_kugelaudio_cli_chunking.cpp` uses the connector-stop test hook to verify multi-chunk CLI stitching without requiring a full decode; `tests/test_kugelaudio_cli_chunking_e2e.cpp` adds real-weight CPU-backed CLI coverage for both multi-chunk stitching and the single-chunk bypass path; `tests/test_kugelaudio_cli_chunking_seed.cpp` adds byte-level seeded chunked-CLI regression coverage.
- Affected area: `src/kugelaudio_chunking.{hpp,cpp}`, `src/vibevoice_tts.cpp`, `src/vibevoice_tts.hpp`, `examples/cli/main.cpp`, `tests/test_kugelaudio_chunking.cpp`, `tests/test_kugelaudio_cli_chunking.cpp`, `prd.md` Slice 8.

### KugelAudio eval reference fixture shape
- Context: the next Slice 4 item requires a reproducible reference sample as an acceptance fixture, but the repo has no audio fixtures yet.
- Chosen default: commit a small synthetic 24 kHz mono 16-bit PCM WAV fixture to `tests/fixtures/reference_sine.wav` with deterministic properties (440 Hz sine, -18 dBFS, 3 s duration), plus a script to regenerate it. The eval config points to this fixture by a relative path so both canonical and ggml runs can use the same artifact directly.
- Rejected alternatives:
  - depend on an external/downloaded reference audio: harder to reproduce from a clean checkout
  - generate the fixture inline inside the test each run: easy to miss in a fresh checkout and harder to ensure stable hashing
  - commit a generic existing WAV unmodified: might lack clear known-good properties for regression testing
- Affected area: `prd.md` Slice 4 / "At least one reproducible reference sample is defined.".

### KugelAudio <private-audio-dir>d sample shape
- Context: the next Slice 4 item requires a reproducible <private-audio-dir>d sample (the canonical TTS output against which ggml is compared), but the repo cannot generate the actual output without the real model checkpoint.
- Chosen default: define the <private-audio-dir>d sample explicitly in the eval config as `voice_cloned_sample` with a stable name, description, and the exact input tuple. The harness plan exposes a `voice_cloned_sample.ground_truth` pointing to the canonical output path (`canonical.wav`). The actual ground-truth WAV is produced during a real canonical run and its SHA256 is recorded in `results.json`.
- Rejected alternatives:
  - commit a pre-generated canonical WAV to the repo: too large and tied to a specific canonical version
  - treat the canonical output as an implicit side effect with no named concept: weaker for regression and harder to explain to a fresh developer
- Affected area: `prd.md` Slice 4 / "At least one reproducible <private-audio-dir>d sample is defined.".

### KugelAudio eval setup config + plan mode
- Context: the first Slice 4 item requires the canonical-vs-ggml evaluation setup to be scripted and reproducible, but the repo cannot assume large model artifacts are always present during routine local validation.
- Chosen default: define the eval setup around one JSON config file plus a `--plan` dry-run mode in the harness script; `--plan` may be used with `--allow-missing-artifacts` for hermetic validation, while real execution still validates required artifacts.
- Rejected alternatives:
  - require all model artifacts even for setup/plan validation: stronger, but too heavy for routine script-level testing
  - encode the evaluation setup only as shell snippets in docs: less reproducible and harder to validate mechanically
- Affected area: `prd.md` Slice 4 / "Evaluation setup is scripted and reproducible.".

### KugelAudio shared eval-run contract
- Context: the next Slice 4 item requires inputs/settings to be pinned and shared between canonical and ggml runs, not merely present twice in separate command blocks.
- Chosen default: emit one explicit `shared_run` block in the normalized eval plan and derive both canonical and ggml commands from it; include a reference-audio SHA256 when the file exists so reruns can confirm they used the same artifact.
- Rejected alternatives:
  - keep duplicating values independently inside each command only: workable, but weaker as an auditable shared contract
  - require all artifact hashes unconditionally: stronger, but incompatible with the dry-run/allow-missing-artifacts plan mode
- Affected area: `prd.md` Slice 4 / "Inputs/settings are pinned and shared between canonical and ggml runs.".

### KugelAudio eval ASR phase + recall automation
- Context: an earlier Slice 4 item required closed-loop ASR regression to be automated, and the harness previously only planned/executed TTS, not ASR transcription or recall computation.
- Chosen default at that stage: extend the divergence harness to include an ASR phase after successful TTS for both canonical and ggml outputs, using the repo's existing `vibevoice-cli asr` command and a word-level recall metric against the source text. The recall helper mirrors the existing C++ `test_15b_closed_loop.cpp` logic (extract `"Content":"..."` fields, compute word overlap).
- Status: transitional decision; the publish-cleanup plan now treats ASR-based validation as an internal migration aid to be replaced by KugelAudio-native TTS regression metrics before the branch is presented as KugelAudio-only.
- Rejected alternatives:
  - add a separate standalone ASR+recall script: would fragment the acceptance path and make the harness less self-contained
  - embed ASR inside a new C++ test only: would not integrate with the divergence harness's side-by-side comparison
  - use an external ASR model for the canonical side: contradicted the earlier PRD requirement to use "the repo's ASR path"
- Affected area: `prd.md` Slice 4 / acceptance-metric migration planning.

### KugelAudio eval threshold enforcement shape
- Context: an earlier Slice 4 item required `f16` to reach at least 95% of canonical recall with an absolute floor of 0.80, but real model artifacts were not available during routine script validation.
- Chosen default at that stage: make the threshold a first-class executable check in the harness (`check_threshold()`). The harness automatically evaluates the threshold when `--execute both` completes successfully and records `threshold_check.passed` + `threshold_check.message` in `results.json`. The logic is unit-tested independently of real model execution.
- Status: transitional decision; the threshold-checking mechanism remains useful, but the published metric and floor definitions should be migrated from ASR recall to KugelAudio-native divergence/quality checks.
- Rejected alternatives:
  - defer threshold checking to a separate manual review step: weaker automation, harder to integrate into CI
  - require real model execution in the harness test: too heavy for routine local validation
- Affected area: `prd.md` Slice 4 / acceptance-metric migration planning.

### KugelAudio eval results schema v2 (ASR fields)
- Context: adding ASR to the harness required new fields in the already-defined `results.json` schema.
- Chosen default at that stage: add `asr_command`, `asr_log_path`, `asr_return_code`, `asr_transcript`, `recall` to both `canonical` and `ggml` blocks, keeping `schema_version = 1` since this was an additive change to the same schema version.
- Status: transitional decision; the publish-cleanup plan should either retire these ASR-specific fields from the published contract or mark them as deprecated/internal-only once TTS-native regression replaces ASR-based validation.
- Rejected alternatives:
  - bump schema_version to 2 for additive fields: unnecessary churn; the schema stayed the same conceptual contract with more optional fields
  - store ASR results in a separate file: harder to correlate with TTS results during regression comparisons
- Affected area: `prd.md` Slice 4 / result-contract migration planning.

### KugelAudio q8_0 acceptance path shape
- Context: the next Slice 4 item requires q8_0 to complete conversion, load, and end-to-end generation on the same acceptance path, but real q8_0 artifacts may not be available during routine validation.
- Chosen default: add a real-model gated C++ smoke test (`tests/test_kugelaudio_q8_0_smoke.cpp`) that loads a q8_0 model via `VIBEVOICE_KUGELAUDIO_Q8_MODEL`, generates with the same acceptance fixture/parameters, and asserts non-empty/finite/non-silent output. Also add `ggml_model_q8_0` to the eval config so the divergence harness has a slot for the q8_0 artifact.
- Rejected alternatives:
  - require q8_0 conversion in every test run: too heavy for routine local validation
  - skip q8_0 entirely and only test f16: would leave a gap in the acceptance surface
- Affected area: `prd.md` Slice 4 / "`q8_0` must complete conversion, load, and end-to-end generation on the same path.".

### KugelAudio quantize profile override surface
- Context: the next Slice 5 item requires selective, safer-than-legacy quantization experiments, but previously only baseline `--type` + built-in LM/FFN/attn/lm_head path was exposed in `examples/quantize`.
- Chosen default: keep default core families quantized through `--type` and add explicit optional overrides for matmul-safe secondary families (`--embed-type`, `--ac-connector-type`, `--sc-connector-type`, `--dh-type`, `--at-block-ffn-type`, `--st-block-ffn-type`).
- Rejected alternatives:
  - globally quantize every family behind `--type`: would silently quantize conv/gamma/norm paths that still need fp16
  - auto-quantize non-core families without explicit opt-in: too risky for silent regression in memory/quality tradeoffs
- Affected area: `examples/quantize/main.cpp`, `README.md`, `docs/conversion.md`, `tests/test_kugelaudio_quantize_cli.cpp`, `tests/CMakeLists.txt`.

### KugelAudio ASR reuse validation
- Context: an earlier Slice 4 sub-task required documenting ASR-specific assumptions and validating that the harness reused the repo's ASR path rather than inventing a new evaluator.
- Chosen default at that stage: extend the recall helpers test with realistic multi-segment ASR output examples (matching the C++ ASR's JSON Content-field format), and document the ASR assumptions in `docs/kugelaudio-parity.md` (ASR model loading, RMS normalization, transcript format, Content extraction, recall metric alignment).
- Status: transitional decision; this documentation/test surface should either be retired with ASR removal or explicitly demoted to an internal migration-only note before branch publication.
- Rejected alternatives:
  - add a heavy end-to-end ASR integration test: already covered by the harness execution path; repeating it as a separate test would have been redundant
  - document ASR assumptions only in code comments: less discoverable for future evaluators
- Affected area: `prd.md` Slice 4 / acceptance-metric migration planning.

### KugelAudio error clarity validation
- Context: the next Slice 5 item requires converter, loader, conditioning, and unsupported-feature errors to be clear and actionable, but there's no systematic validation of error message content beyond checking that errors are raised.
- Chosen default: add `tests/test_kugelaudio_error_clarity.py` which validates that harness threshold failure messages explicitly name the violated boundary (ratio or floor), and that the supported-checkpoint name appears in converter unsupported-checkpoint errors. Run this alongside the existing feature-gating, CLI rejection, and CAPI tests that already exercise the error paths.
- Rejected alternatives:
  - write a C++ stderr-capture test for loader errors: more complex and fragile than validating the Python-side messages that developers will see in practice
  - manually audit every error string: not repeatable as a regression check
- Affected area: `prd.md` Slice 5 / "Converter, loader, conditioning, and unsupported-feature errors are clear and actionable.".


### KugelAudio logging contract shape
- Context: the next Slice 5 item requires logs to include checkpoint/config, converter mode, active conditioning, quantization mode, and eval configuration, but the existing logging was sparse and scattered.
- Chosen default: add structured startup/summary logging to three key surfaces: (1) converter writes `convert: mode=... checkpoint=... src=...` with conditioning info, (2) CLI TTS writes `tts: model_variant=... kugelaudio=... quantization_hint=...` and `tts: generation_settings frames=... steps=... cfg=... seed=... conditioning=... ref_count=...`, (3) eval harness writes startup config summary with text, ref, generation settings, and model paths. Document the logging contract in `docs/kugelaudio-parity.md`.
- Rejected alternatives:
  - add a separate structured-logging library or format: broader than needed for the current acceptance surface
  - only log in verbose mode: misses the PRD requirement for these to be visible by default for operator diagnostics
- Affected area: `prd.md` Slice 5 / "Logs include checkpoint/config, converter mode, active conditioning, quantization mode, and eval configuration.".


### KugelAudio log redaction default
- Context: the next Slice 5 item requires that sensitive prompt/audio contents are not dumped by default, but the eval harness startup log was printing the full prompt text and raw reference-audio path.
- Chosen default: keep default logs useful while redacting contents. The eval harness now logs `text_summary=chars=... sha256=...` and `ref_summary=present=... sha256=...` instead of raw text/path. Raw content remains available only in explicit plan/results artifacts, not default stderr summaries.
- Rejected alternatives:
  - remove prompt/audio-related logging entirely: safer, but weaker for operators debugging eval mismatches
  - keep full prompt/path logging and rely on a future verbose/redaction flag: violates the default-safety requirement
- Affected area: `prd.md` Slice 5 / "Sensitive prompt/audio contents are not dumped by default.".

### KugelAudio legacy-path marking scope
- Context: the next Slice 5 item requires legacy paths outside KugelAudio v1 acceptance to be clearly marked or kept off the critical path, but the highest-traffic repo surfaces (README, conversion docs, CLI help) still presented several old VibeVoice flows as if they were equally current.
- Chosen default: mark legacy paths explicitly where developers are most likely to enter the repo: README section headings, `docs/conversion.md` for `convert_voice_to_gguf.py`, and CLI help for `--voice`. Keep the codepaths available, but label them as legacy/migration/reference and not part of KugelAudio v1 acceptance.
- Rejected alternatives:
  - remove the legacy codepaths immediately: too risky while migration/regression coverage still depends on them
  - only document this in maintainer notes: too easy for fresh developers to miss compared with README/CLI/doc entrypoints
- Affected area: `prd.md` Slice 5 / "Legacy paths that are not part of v1 acceptance are clearly marked or kept off the critical path.".

### KugelAudio maintainer-doc refresh scope
- Context: the next Slice 5 item requires maintainer docs to reflect the KugelAudio-first reality, but the repo's maintainer guide (`AGENTS.md`) still described the migration broadly without explicitly pointing maintainers at the current acceptance workflow, eval harness, fixture config, and parity notes.
- Chosen default: refresh `AGENTS.md` rather than creating another maintainer doc. Add a `Start here for the KugelAudio v1 acceptance path` section, expand the layout/what-is-real sections to include the eval harness and parity docs, and update the converter workflow to end at `results.json + per-step logs`.
- Rejected alternatives:
  - create a separate maintainer-onboarding doc: would split the KugelAudio-first guidance across too many files
  - rely on README only: maintainer guidance needs a stronger execution-oriented view than user-facing quickstarts
- Affected area: `prd.md` Slice 5 / "Maintainer docs reflect the KugelAudio-first reality.".

### KugelAudio acceptance-surface boundary
- Context: the next Slice 5 item requires proving that acceptance scripts/tests do not depend on dropped features, but the repo still contains many legacy VibeVoice regression tests and examples by design.
- Chosen default: define the acceptance surface narrowly as the KugelAudio eval harness (`scripts/eval_kugelaudio_divergence.py`), its fixture/config (`tests/fixtures/kugelaudio_eval_config.json`), and their dedicated acceptance tests. Add a regression test that asserts this surface does not use `--voice`, multiple `--ref-audio`, speaker-tagged dialog, `voice_cache`, or language-hint fields.
- Rejected alternatives:
  - require the entire repo test suite to avoid every dropped feature: impossible while legacy regression coverage is intentionally retained
  - only rely on documentation/inspection: weaker than a machine-checked boundary on the actual acceptance path
- Affected area: `prd.md` Slice 5 / "Acceptance scripts/tests do not depend on dropped features.".

### KugelAudio request-policy seam
- Context: the next Slice 5 item requires proving that the current v1 design does not block deferred features, but the KugelAudio request validation logic was hard-coded as one `validate_kugelaudio_single_speaker_request(...)` path with no explicit widening seam.
- Chosen default: introduce an explicit `KugelAudioRequestPolicy` plus `validate_kugelaudio_request(...)`, and keep the current v1 behavior as one named profile (`kugelaudio_v1_request_policy()`). This keeps current behavior unchanged while making later support for wider request shapes (e.g. multi-reference or speaker-tagged dialog) a policy expansion instead of a rewrite.
- Rejected alternatives:
  - leave the single-speaker helper as the only API: simpler short-term, but makes future widening look like special-case patching
  - add partial V2 fields to `VibeVoiceTTSParams` right now: broader than needed for the current v1-only increment
- Affected area: `prd.md` Slice 5 / "V1 design does not block deferred features.".

### KugelAudio deferred-feature seam documentation
- Context: the next Slice 5 item requires deferred-feature seams to be identified in code/docs where relevant, but those seams were only implicit across prior refactors.
- Chosen default: document the main widening seams explicitly in `docs/kugelaudio-parity.md` (request-shape seam, prompt-builder seam, eval/quantization seam, deferred-scope boundary) and add a small code comment in the eval harness near the quantized-model slots. Back this with a regression test that checks the seam markers remain visible.
- Rejected alternatives:
  - scatter one-off TODO comments only in code: too easy to miss and too weak as maintainer guidance
  - create a brand-new V2 design doc: broader than needed while the parity note already tracks the relevant acceptance/deferred boundaries
- Affected area: `prd.md` Slice 5 / "Deferred-feature seams are identified in code/docs where relevant.".

### KugelAudio critical-path non-feature marking
- Context: the next Slice 5 item requires ensuring no V2 feature is half-implemented in the critical path, but the main TTS request header still described legacy multi-speaker reference semantics without clearly separating them from the KugelAudio v1 surface.
- Chosen default: mark the shared `VibeVoiceTTSParams` surface explicitly as a legacy-vs-KugelAudio boundary. Update `src/vibevoice_tts.hpp` comments to say legacy VibeVoice 1.5B wider shapes are retained only for non-KugelAudio paths, while KugelAudio v1 stays narrow behind `detail::KugelAudioRequestPolicy`. Mirror that in the CLI 1.5B help/error text.
- Rejected alternatives:
  - remove the legacy semantics from the shared TTS params comment entirely: would hide real legacy behavior still present for regression/migration use
  - leave the header comment ambiguous and rely on external docs: too easy for future work to interpret as “almost-supported” V2 behavior in the critical path
- Affected area: `prd.md` Slice 5 / "No V2 feature is half-implemented in the critical path.".

### KugelAudio developer-doc scope anchor
- Context: a Slice 5 item required developer-facing docs to reflect the current v1 scope and acceptance criteria, but the README still led primarily into legacy VibeVoice quickstarts and `docs/conversion.md` still opened as a VibeVoice-only conversion guide.
- Chosen default at that stage: anchor the developer-facing surface around one explicit `KugelAudio v1 acceptance path` section in the README and align the conversion doc intro to that same scope/criteria. Keep the legacy material below, but put the supported path first.
- Status: transitional decision; the publish-cleanup plan now prefers removing or archiving non-KugelAudio publish surfaces rather than merely keeping them below the fold.
- Rejected alternatives:
  - rewrite the entire README around KugelAudio only immediately: was too destructive while legacy VibeVoice migration examples were still intentionally retained
  - rely only on AGENTS.md / maintainer docs: too hidden for developers arriving through the README/docs first
- Affected area: `prd.md` Slice 5 and Slice 7 / repo-facing publish-surface cleanup.

### KugelAudio canonical-reference doc anchors
- Context: a Slice 5 item required repo-facing docs to name the canonical reference points in `../kugelaudio-open`, but those links were previously scattered or implicit.
- Chosen default: add one explicit canonical-reference section to the README (developer-facing behavior/eval orientation) and one to `docs/conversion.md` (converter/runtime contract orientation), each naming the exact upstream files we use as ground truth.
- Rejected alternatives:
  - rely only on `AGENTS.md` / maintainer notes: too hidden for developers entering through README/docs
  - scatter file references ad hoc across many sections: harder to audit and more likely to drift
- Affected area: `prd.md` Slice 5 / "Canonical reference points into `../kugelaudio-open` are documented.".

### Publish-cleanup staged execution order
- Context: the publish-cleanup work spans identity, docs, acceptance, and feature removal; doing it in the wrong order would create temporary branch states that are either misleading or hard to validate.
- Chosen default: stage the work in this order: (1) switch new artifact/runtime identity to KugelAudio while preserving any required read-only migration compatibility, (2) remove operator-facing `1.5B` / `tts15b` / stale VibeVoice naming, (3) replace ASR-based published acceptance with KugelAudio-native regression criteria and results contract, (4) remove/archive non-KugelAudio product surfaces and legacy docs/tests/scripts.
- Rejected alternatives:
  - remove ASR/runtime surfaces first: too risky before the replacement acceptance harness is defined
  - do docs cleanup first without changing artifact identity: would leave published examples and emitted GGUFs inconsistent
  - attempt one giant delete/rename sweep: higher risk and harder to debug than a staged convergence plan
- Affected area: `prd.md` Slice 7 / overall publish-cleanup execution plan.

### Experimental tail-reference continuity for chunked TTS
- Context: listening to chunked long-form outputs showed improved speaker identity after cached raw-reference conditioning, but seams still restarted voice/prosody and long chunks showed end-of-chunk volume decay/noise.
- Chosen default: keep normal chunking parity-style by default, but add explicit opt-in `--chunk-continuity tail-reference` with `--continuity-tail-ms` (default 1200). This conditions chunk N+1 on the original raw reference plus a voiced, RMS-matched tail from chunk N, using a temporary reference WAV. `none` remains the default.
- Guardrails: tail-reference mode trims/gates the previous chunk tail, clamps RMS gain, deletes temp WAVs after use, and logs tail size/RMS. Cached conditioning preserves the seeded RNG stream so default chunking remains byte-identical for fixed seed/settings.
- Rejected alternatives:
  - make tail-reference default immediately: too risky because it can feed generation artifacts/noise forward
  - only increase crossfade/overlap text: does not address speaker-state restart
  - pass the whole previous chunk as reference: too likely to amplify long-run noise and content leakage
- Follow-up: listen to `/tmp/kugelaudio-continuity-tail-check/tailref-mw224-cfg1-0-tail1200.wav` against the previous sweep winner and tune `--continuity-tail-ms`, chunk size, and possible shorter per-chunk frame caps.

### Latent-prefix continuity experiment
- Context: `tail-reference` made seams nearly imperceptible, but listening reported background noise from chunk 2 onward, consistent with waveform-tail conditioning feeding generated noise back into the next chunk.
- Chosen default: add a second explicit opt-in mode, `--chunk-continuity latent-prefix`, that captures the previous chunk's final generated latent frames and advances the next chunk's LM state through the speech connector before generation. This avoids waveform/acoustic-decoder noise feedback while carrying some autoregressive speech state.
- Observed first CUDA sample: `latent-prefix` runs end-to-end and logs captured/applied latent frames, but the first RMS check shows it does not fix volume decay as strongly as `tail-reference`; it should be evaluated by listening for noise/seam quality rather than promoted as default.
- Follow-up: compare `/tmp/kugelaudio-continuity-latent-check/latentprefix-mw224-cfg1-0-tail1200.wav` with the tail-reference sample and consider a cleaner tail-selection or hybrid mode if `latent-prefix` is quieter but less continuous.

### Tail-reference continuity retired after listening
- Context: listening comparison found `tail-reference` made seams cohesive but introduced substantial background noise from chunk 2 onward, while `none` stayed noise-free but had voice/intonation jumps.
- Decision: retire `tail-reference` from the published CLI. The runtime now rejects it with an explicit error explaining that decoded waveform noise is fed back into later chunks. Keep the historical note because it is an attractive but harmful idea.
- Current experiment to prefer: `--chunk-continuity latent-prefix` plus shorter chunks/frames. It carries model speech latents instead of decoded waveform audio, avoiding reference-encoder noise feedback, though it may be weaker for continuity.
- Do not repeat: do not condition later chunks on decoded/generated waveform tails unless there is a denoising/quality gate that proves it does not amplify artifacts.

### Prompt-instruction continuity replaces generated-state carry
- Context: listening found `none` was clean but jumpy, `tail-reference` was cohesive but noisy, and `latent-prefix` was both jumpy/noisy. This indicates generated audio/latent carry is off-distribution or contaminated for continuity.
- Decision: expose only `--chunk-continuity none|prompt-instruction` on the CLI. `prompt-instruction` adds a system-side, non-spoken continuity instruction to chunk N+1, including a short previous-chunk text excerpt, but feeds no generated audio or latents back into the model.
- Retired: `tail-reference` and `latent-prefix` are now rejected by the CLI with explicit explanations so we do not repeat those experiments accidentally.
- Current test profile: short clean chunks plus text-only continuity: `--max-words-per-chunk 80 --max-frames 64 --overlap-sentences 1 --crossfade-ms 60 --chunk-continuity prompt-instruction`.

### Prompt-instruction v2: reference-anchored and applied to every chunk
- Listening found prompt-instruction v1 had no noise, but first chunk sounded like a woman and chunks 2+ switched to a cohesive male voice.
- Verification: this was not because only chunk 1 used reference audio. Chunk logs show every chunk uses cached raw-reference conditioning (`speaker 0: ref ... -> 23 compressed frames`, plus `reusing cached raw-reference conditioning across 10 chunks`).
- Likely cause: v1 only changed the system prompt for chunks 2+ and included a previous-text excerpt, pushing later chunks into a different prompt distribution.
- Change: prompt-instruction now uses a short reference-anchored instruction (`Use only the voice from Voice input Speaker 0...`) and applies it to every chunk, including chunk 1. It no longer includes previous chunk text in the system instruction; overlap text remains the continuity context.

### CUDA final decoder status: auto now streams on GPU
- Reproduced one-shot CUDA final decoder failure on q8 with `--final-decoder-backend auto` before the fix: `ggml_cuda_compute_forward: IM2COL failed` / CUDA invalid argument in `decode_latent_sequence(...)` at 64 frames. Failing log preserved remotely at `runs/cuda-final-decode-debug/q8-auto-f64.log`.
- Verified explicit `--final-decoder-backend stream` succeeds on CUDA for the same 64-frame q8 sample.
- Change: `FinalDecoderBackend::Auto` now uses one-shot decode on CPU but streamed final decode on non-CPU backends. Logs show `using streamed final decoder on backend=CUDA0 ... (auto)`.
- Verified q8/CUDA auto streamed short sample: remote `runs/cuda-final-decode-debug/q8-auto-streamed-f64.wav`.
- Verified first long-form CUDA/q8 profile without CPU final decode: `--chunk-continuity none --max-words-per-chunk 120 --max-frames 96 --final-decoder-backend auto`; remote `runs/longform-q8-cuda-stream-profile/none-q8-mw120-f96-auto.wav`, local `/tmp/kugelaudio-longform-q8-cuda-stream-profile/none-q8-mw120-f96-auto.wav`, elapsed about 120s for 64.8s generated audio. Auto used streamed CUDA final decode for all 5 chunks.

### Online research: ggml CUDA IM2COL failure likely grid-Y launch limit
- Research sources: ggml/whisper issues show `ggml_cuda_compute_forward: IM2COL failed` is a generic CUDA wrapper; reported causes include unsupported PTX/toolchain or missing kernel image when CUDA arch/toolchain is wrong, but our RTX 3090 local build reports `invalid argument`, not those errors.
- Most relevant upstream delta: current ggml `src/ggml-cuda/im2col.cu` defines both `MAX_GRIDDIM_Y` and `MAX_GRIDDIM_Z`, clamps `block_nums.y = MIN(OW, MAX_GRIDDIM_Y)`, and loops `for (iow = blockIdx.y; iow < OW; iow += MAX_GRIDDIM_Y)`. Our vendored file only clamps Z and launches `dim3(..., OW, ...)`.
- Hypothesis: one-shot acoustic decoder creates an im2col with `OW > 65535`, so CUDA kernel launch exceeds gridDim.y and returns `invalid argument`. Streamed final decode works because it keeps each decoder segment below that width.
- Recommended fix: backport upstream ggml im2col Y-clamp/loop into `third_party/ggml/src/ggml-cuda/im2col.cu`, then retest pure one-shot CUDA final decode before deciding whether `auto` should remain streamed on GPU.

### Backported ggml CUDA im2col grid-Y fix
- Implemented upstream-style `MAX_GRIDDIM_Y` clamp/loop in `third_party/ggml/src/ggml-cuda/im2col.cu` for both 1D and 3D im2col kernels.
- Added `--final-decoder-backend active` to force one-shot final decode on the selected backend for diagnostics; `auto` still streams on GPU backends to avoid large decoder allocations.
- Verified on `CUDA workstation` q8/CUDA: `--final-decoder-backend active --max-frames 64` now succeeds where it previously aborted with `ggml_cuda_compute_forward: IM2COL failed`.
- Long one-shot q8/CUDA `mw120/f96` no longer hits IM2COL invalid-argument; it now fails cleanly with CUDA OOM trying to allocate a ~15.6 GiB final decoder buffer. Therefore streamed GPU decode should remain the default for long-form CUDA iteration.

### Planned experiment: sanitized clean-tail reference
- User proposed revisiting waveform-tail continuity with a safer selector: scan backward through the previous chunk by RMS windows, find the last/clean voiced region rather than blindly using the literal ending, take up to a short duration, align it to original reference statistics, and append it to the original reference audio for the next chunk.
- Important nuance: the old `tail-reference` already did a simple voiced-end RMS gate and RMS match, but it still used a broad tail and allowed larger gain. The new experiment should be stricter: choose a clean voiced island, start around 500 ms, remove DC, match RMS to voiced original-reference RMS with conservative gain clamp (roughly 0.5x..1.25x), fade/peak clamp, and avoid min/max matching initially.
- Proposed mode name: `clean-tail-reference`. Keep old `tail-reference` retired/rejected so we do not accidentally re-enable the unsafe behavior.
- Experiment ladder: original ref + first clean 500 ms of chunk 1; original ref + last clean 500 ms of chunk 1; original ref + selected clean voiced island near chunk end; same with filtering if needed; compare recursive vs non-recursive tail reuse.

### Implemented `clean-tail-reference` continuity mode
- Added CLI/runtime mode `--chunk-continuity clean-tail-reference`; default remains `none`, while old `tail-reference` and `latent-prefix` remain rejected.
- Behavior: for chunk N+1, load the original raw reference, extract a clean voiced island from chunk N, sanitize it, and create a temporary reference WAV as `original_ref + 100 ms silence + clean_tail`.
- Extraction: non-overlapping 50 ms windows in the last ~6 s of the previous chunk; voiced windows require RMS above a gate and no clipping; candidate islands are scored by RMS with penalties for high zero-crossing, high first-difference ratio, clipping, and a small recency bonus.
- Sanitization: subtract tail mean/DC, match RMS to voiced original-reference RMS with conservative gain clamp `0.50..1.25`, clamp peaks to ±0.95, and apply ~10 ms fade-in/fade-out. We intentionally do not min/max match because outliers can amplify artifacts.
- Local CPU smoke passed with q8: `/tmp/kugelaudio-clean-tail-smoke/clean-tail.wav`; log showed chunk 2 used temp reference with 26 compressed frames and clean-tail stats.
- Remote CUDA long-form sample attempt was blocked by unrelated VRAM occupancy: `main.py --enable-cors-header ...` used ~15.7 GiB on the RTX 3090, leaving only ~8.3 GiB, so q8 model load failed. Ask before killing that process.

### Clean-tail-reference q8/CUDA listening sample generated
- User authorized killing the ComfyUI-like process if still using VRAM. On retry, PID 2156625 (`./.venv/bin/python main.py --enable-cors-header --enable-manager --listen 0.0.0.0 --port 8188`) was still using ~10.1 GiB VRAM and was killed. GPU memory dropped to ~33 MiB used.
- Generated q8/CUDA sample with `--chunk-continuity clean-tail-reference --continuity-tail-ms 500 --max-words-per-chunk 80 --max-frames 64 --overlap-sentences 1 --crossfade-ms 60 --steps 4 --cfg 1.0 --seed 12345 --final-decoder-backend auto`.
- Remote: `runs/clean-tail-reference-check/cleantail-q8-mw80-f64-tail500.wav`; local: `/tmp/kugelaudio-clean-tail-reference-check/cleantail-q8-mw80-f64-tail500.wav`; duration 87.13s; elapsed 176s.
- RMS by 10s: 0-10=0.03913, 10-20=0.01405, 20-30=0.00731, 30-40=0.02112, 40-50=0.01551, 50-60=0.02105, 60-70=0.01290, 70-80=0.02892, 80-87=0.02060. Overall RMS=0.02200, peak=0.40259.
- Clean-tail stats for chunks 2-10: 9 tails, 4800-12000 samples (mean ~6489), raw RMS 0.00880-0.05123, aligned RMS 0.01081-0.06255, gain always hit clamp 1.25. Several selected tails are low-energy, so listening should verify whether this causes volume drop or weaker continuity.

### Original-reference repeat experiment generated
- Tested the non-poisonous conditioning variant: no generated audio feedback. Created `reference-plus-reference.wav` as `original_ref + 100 ms silence + original_ref`, then ran normal chunking with `--chunk-continuity none` so all chunks reuse the same augmented original-reference conditioning.
- Settings: q8/CUDA, `--max-words-per-chunk 80 --max-frames 64 --overlap-sentences 1 --chunking-strategy heuristic --pause-mode punctuation --crossfade-ms 60 --steps 4 --cfg 1.0 --seed 12345 --final-decoder-backend auto`.
- Remote output: `runs/original-ref-repeat-check/refrepeat-q8-mw80-f64.wav`; local output: `/tmp/kugelaudio-original-ref-repeat-check/refrepeat-q8-mw80-f64.wav`; duration 87.13s, elapsed 163s.
- Log: cached raw-reference conditioning reused across 10 chunks with `frames=46` (versus 23 for the original single reference), and streamed CUDA final decoder used for all chunks.
- RMS by 10s: 0-10=0.02749, 10-20=0.02714, 20-30=0.02940, 30-40=0.04579, 40-50=0.03330, 50-60=0.02128, 60-70=0.02828, 70-80=0.03467, 80-87=0.02828. Overall RMS=0.03139, peak=0.31909. This is much more stable than clean-tail-reference but a bit lower RMS than the original `none` sample.

### Continuity feedback modes retired; single-sequence model-state baseline added
- Listening found `clean-tail-reference` still collapsed into noise and speaker identity drift. This confirms generated waveform feedback is unsafe even when selected/sanitized and RMS-aligned. Repeated original-reference conditioning also drifted, so the issue is independent chunk restarts rather than simply weak reference conditioning.
- Retired from CLI: `prompt-instruction`, `tail-reference`, `clean-tail-reference`, and `latent-prefix`. They now reject with explicit messages. Recommended quality mode remains `--chunk-continuity none`.
- Added first true model-state baseline: `--chunk-continuity single-sequence`. It uses chunk planning only to estimate `total_frame_budget = max_speech_frames * planned_chunks`, then disables chunking and generates the full original text in a single LM/KV-cache sequence. This avoids independent restarts without feeding generated audio/latents into reference conditioning.
- Next test: q8/CUDA `single-sequence` on the longform fixture with `--max-words-per-chunk 80 --max-frames 64`, total budget about 640 frames. Compare against `none` for speaker drift vs long-sequence autoregressive degeneration.

### Single-sequence q8/CUDA baseline result
- Generated `--chunk-continuity single-sequence` sample on `CUDA workstation`: remote `runs/single-sequence-check/singleseq-q8-mw80-f64.wav`, local `/tmp/kugelaudio-single-sequence-check/singleseq-q8-mw80-f64.wav`.
- Settings matched mw80/f64 plan; chunk planner estimated 10 chunks, so single-sequence total frame budget was 640 frames.
- Result: generation stopped early at `speech_end/eos before diffusion at frame 259`, producing only 34.53s audio (828800 samples) from a 640-frame/85s-ish budget. Runtime elapsed 68s; streamed CUDA final decode for 259 latent frames.
- RMS by 10s: 0-10=0.04558, 10-20=0.02182, 20-30=0.01053, 30-34.5=0.00702. This shows clear energy collapse by the end even before the intended full text length.
- Conclusion: single-sequence is true KV/model-state continuity but not enough as-is; it avoids restarts yet hits long-sequence early-stop/degeneration. Next true-state experiment, if pursued, should be a segmented KV carry / text-insertion design rather than one giant generation.

### Single-sequence min-frame suppression prototype
- Added internal `VibeVoiceTTSParams::min_speech_frames`. For `--chunk-continuity single-sequence`, min frames are set to 90% of the planned total frame budget (`max_frames * planned_chunks`) so early speech_end/eos is suppressed while testing true long single-KV continuity.
- Implementation suppresses `speech_end`/`eos` logits before token selection until `total_frames >= min_speech_frames`, logging suppressed stop tokens. Default remains 0, so normal generation/parity behavior is unchanged.
- Attempted q8/CUDA `singleseq-min90-q8-mw80-f64.wav`; run failed during full-text LM prefill with CUDA OOM allocating ~7.9 GiB because the ComfyUI-like `.venv/bin/python main.py ... --port 8188` process had restarted and was using ~7.0 GiB VRAM. Did not kill it this time because authorization was not repeated in the current request.

### Segmented-state KV-carry prototype result
- Implemented `--chunk-continuity segmented-state`, a true model-state/KV carry experiment. It starts with the normal KugelAudio prompt for chunk 1, generates up to `max_frames` for that segment, then inserts a text transition into the same LM KV cache for chunk N+1: `Text input -> Speaker 0: chunk text -> Speech output -> speech_start`. No generated waveform or latent snippets are used as reference conditioning.
- Current prototype requires `--cfg 1.0` and decodes one continuous latent sequence at the end. It does not insert audio pauses/crossfades; all continuity is in LM state.
- q8/CUDA run: remote `runs/segmented-state-check/segstate-q8-mw80-f64.wav`, local `/tmp/kugelaudio-segmented-state-check/segstate-q8-mw80-f64.wav`; settings mw80/f64, 10 planned chunks. Runtime elapsed 157s.
- Segment logs: chunks 1-8 hit 64-frame budget; chunk 9 stopped after 59 frames; chunk 10 stopped after 55 frames. Total 626 latent frames, output duration 83.47s.
- RMS by 10s: 0-10=0.04172, 10-20=0.02766, 20-30=0.01418, 30-40=0.02655, 40-50=0.01670, 50-60=0.02138, 60-70=0.01895, 70-80=0.01095, 80-83.5=0.00263. Overall RMS=0.02355, peak=0.40259.
- Compared to giant single-sequence, segmented-state avoids total silence after ~40s and reaches near full duration, but still shows energy decay and likely needs listening for speaker/intonation behavior.

### Oracle dump instrumentation started
- After committing the continuity/CUDA diagnostic work (`5c9b700 Add long-form continuity diagnostics`, with ggml submodule commit `987939b9 cuda: clamp im2col grid y dimension`), started Slice 8 oracle-dump work.
- Added C++ per-frame dump support controlled by `VIBEVOICE_KUGELAUDIO_DUMP_FRAMES=N`. When paired with `VIBEVOICE_KUGELAUDIO_DUMP_DIR`, non-chunked generation now writes `frame_XXXX_logits`, `selected_token`, `diffusion_noise`, `diffusion_cond_pos`, optional `diffusion_cond_neg`, `diffusion_latent`, `step_embed`, and `hidden_after_step`.
- Extended `scripts/dump_kugelaudio_canonical_tensors.py` with `--dump-frames` and a local mirror of canonical `sample_speech_tokens` so the initial per-frame diffusion noise can be dumped before denoising.
- Smoke-tested C++ dump locally with q8 CPU, `--max-frames 2 --steps 1 --cfg 1.0 --seed 12345`; `/tmp/kugel-cpp-dump` contains both `frame_0000_*` and `frame_0001_*` dumps.

### Canonical non-chunked oracle: short f16 parity result
- Enabled canonical/C++ noise alignment beyond the initial latent: canonical dump now writes per-frame per-DPM-step SDE variance noise (`frame_XXXX_dpm_step_NN_variance_noise`), and C++ can load it via `VIBEVOICE_KUGELAUDIO_NOISE_DIR`.
- Found a major parity issue: C++ DPM sampler implemented deterministic `dpmsolver++`, while canonical KugelAudio config uses `sde-dpmsolver++`. With only initial diffusion noise aligned, first latent cosine was ~0.946 and frame 1 latent cosine ~0.49 despite hidden/logit parity.
- Patched C++ DPM solver with `sde_dpmsolver_plus_plus` support and set KugelAudio solver config to SDE mode. With canonical initial + per-step variance noise loaded, short f16/CUDA parity improved dramatically:
  - prefill hidden cosine: ~0.999989
  - frame 0 latent cosine: ~0.9999993, max_abs ~8.3e-4
  - frame 0 step_embed cosine: ~0.9999999
  - frame 1 diffusion cond cosine: ~0.999981
  - frame 1 latent cosine: ~0.999983
  - frame 1 step_embed cosine: ~0.999993
- Remote artifacts:
  - canonical dump: `<external-workspace>/runs/oracle-short/canonical-f16-sde`
  - C++ dump: `<external-workspace>/runs/oracle-short/cpp-f16-sde`
  - compare report: `<external-workspace>/runs/oracle-short/compare-canonical-f16-sde-cpp-f16-sde.txt/json`
- CUDA workstation setup notes: use `UV_CACHE_DIR=<external-cache>/uv HF_HOME=<external-cache>/hf XDG_CACHE_HOME=<external-cache>/xdg` because home has too little free space for PyTorch/HF model downloads.

### Rolling-KV oracle diagnostics started
- Added diagnostic env knobs, not public CLI:
  - `VIBEVOICE_KUGELAUDIO_TEACHER_STEP_EMBEDS=1`: load canonical `frame_XXXX_step_embed` from `VIBEVOICE_KUGELAUDIO_NOISE_DIR` and feed it to the LM after each diffusion frame, while optionally dumping the C++-generated pre-forcing `frame_XXXX_step_embed_model`.
  - `VIBEVOICE_KUGELAUDIO_ROLLING_KV_FRAMES=K`: rebuild the LM KV each frame from the full original prompt plus only the last K generated speech embeddings. The retained speech embeddings keep their original absolute positions (`N + global_frame`) while the attention set is masked by omission of older speech KV entries.
- Added canonical `frame_XXXX_hidden_after_step` dumps, emitted after feeding each generated speech embedding into the canonical LM.
- Diagnostic result: with teacher-forced canonical step embeddings, even q8 C++ LM hidden state stays close to the canonical f16 oracle across 16 frames (`hidden_after_step` cosine around 0.9998). This means the severe 16-frame free-running divergence is not primarily resident-KV math; it is caused by small diffusion/connector differences feeding back into future LM state.
- Rolling-KV K ladder against 16-frame oracle, q8 + teacher-forced step embeddings, global absolute positions:
  - K=1 cond cosine at frames [0,1,4,8,15] = [0.999888, 0.999851, 0.800368, 0.772394, 0.673905]
  - K=2 = [0.999888, 0.999851, 0.948937, 0.924967, 0.758715]
  - K=4 = [0.999888, 0.999851, 0.999786, 0.952824, 0.839442]
  - K=8 = [0.999888, 0.999851, 0.999786, 0.999821, 0.939408]
  - K=16 is effectively full history for this 16-frame fixture and should match the non-rolling teacher-forced path.
- Rolling-KV free-running q8 still diverges because generated step embeddings diverge; for production-quality long-form the next useful experiment is not just a KV window, but either (a) improve base diffusion/connector parity in free-running f16 further, or (b) try larger K on true f16 once VRAM is available. CUDA workstation f16 C++ run OOMed while ComfyUI was using ~5.5 GiB VRAM; did not kill it.

### F16 oracle retry after freeing CUDA workstation VRAM
- User killed the competing process; CUDA workstation had ~24 GiB free. Retried the f16/CUDA 16-frame oracle diagnostics successfully.
- Full-history f16 C++ with canonical initial/SDE variance noise and teacher-forced canonical step embeddings now runs end-to-end:
  - remote dump: `<external-workspace>/runs/oracle-16b/cpp-f16-sde-tfstep`
  - compare: `<external-workspace>/runs/oracle-16b/compare-f16-tfstep.{txt,json}`
  - frame 15 diffusion cond cosine ~0.9999965, latent cosine ~0.9999981, hidden_after_step cosine ~0.9999967.
- Full-history f16 free-running (canonical noise loaded, but C++ step embeddings fed forward) still diverges by frame 15:
  - remote dump: `<external-workspace>/runs/oracle-16b/cpp-f16-sde-free`
  - frame 15 diffusion cond cosine ~0.7114, latent cosine ~0.5542, step_embed cosine ~0.6645, hidden_after_step cosine ~0.5945.
  - This confirms the autoregressive LM trajectory is highly sensitive to tiny diffusion/connector differences; teacher-forced hidden parity is excellent, but free-running feedback diverges.
- F16 rolling-KV teacher-forced K ladder with global absolute positions:
  - K=1 cond cos [f0,f1,f4,f8,f15] = [0.999990, 0.999987, 0.800316, 0.770322, 0.673035]
  - K=2 = [0.999990, 0.999987, 0.950012, 0.925665, 0.758888]
  - K=4 = [0.999990, 0.999987, 0.999995, 0.952653, 0.840031]
  - K=8 = [0.999990, 0.999987, 0.999995, 0.999996, 0.939560]
  - K=16 = [0.999990, 0.999987, 0.999995, 0.999996, 0.999997]
- Conclusion: for this 16-frame fixture, a rolling-KV window needs a fairly large K to approximate the non-chunked oracle. K=8 is decent but not exact by frame 15; K=16/full history is effectively oracle. This argues for measuring longer oracle windows before picking a production K.

### Step-embedding blend diagnostic
- Added env-gated blend diagnostic: `VIBEVOICE_KUGELAUDIO_BLEND_STEP_EMBEDS=alpha` with `VIBEVOICE_KUGELAUDIO_NOISE_DIR=<canonical dump>`. The runtime computes the C++ generated step embedding, loads canonical `frame_XXXX_step_embed`, and feeds `canonical * (1 - alpha) + cpp * alpha` into the LM. Normal CLI output is unchanged unless the env var is set.
- Ran f16/CUDA 16-frame oracle blend ladder on CUDA workstation using canonical f16 SDE noise/dumps at `runs/oracle-16b/canonical-f16-sde` and model `kugelaudio-f16.gguf`.
- Blend result by alpha (selected frame-15 metrics):
  - alpha=0.00: cond15 ~0.999997, latent15 ~0.999998, hidden15 ~0.999997
  - alpha=0.50: cond15 ~0.999988, latent15 ~0.999993, hidden15 ~0.999990
  - alpha=0.75: cond15 ~0.999788, latent15 ~0.999917, hidden15 ~0.999837
  - alpha=0.80: cond15 ~0.999528, latent15 ~0.999808, hidden15 ~0.999660
  - alpha=0.85: cond15 ~0.998121, latent15 ~0.999466, hidden15 ~0.998661
  - alpha=0.90: cond15 collapses to ~0.708381, latent15 ~0.640529, hidden15 ~0.638196
  - alpha=0.95/0.99/1.00 also collapse around cond15 ~0.70.
- Interpretation: the feedback loop tolerates surprisingly large C++ step-embedding contribution up to roughly alpha=0.85 on this 16-frame fixture, then hits a sharp trajectory cliff around alpha=0.90. This makes a stabilizer/blend-like intervention plausible: it may only need to remove the last ~10-15% of high-frequency/error component in the generated step embedding to stay near the oracle, at least for short trajectories.
- Remote artifacts:
  - `runs/oracle-16b/cpp-f16-blend-alpha-0p00` ... `0p99`, `1p00`
  - compare reports: `runs/oracle-16b/compare-f16-blend-alpha-*.{txt,json}`

### Temporal step-embedding stabilizer diagnostic failed
- Added env-gated temporal stabilizer diagnostic: `VIBEVOICE_KUGELAUDIO_TEMPORAL_STEP_EMBEDS=alpha`, feeding `alpha * current_step_embed + (1-alpha) * previous_fed_step_embed` into the LM. This was tested as a production-available approximation after the canonical blend diagnostic showed sensitivity to the last 10-15% of step-embed error.
- F16/CUDA 16-frame oracle runs with canonical diffusion noise but no canonical step embeddings:
  - alpha=0.75: cond cos [f4,f8,f12,f15] = [0.585654, 0.477497, 0.379812, 0.446780], latent15 ~0.514753, hidden15 ~0.545920
  - alpha=0.85: [0.855163, 0.589493, 0.569836, 0.642655], latent15 ~0.644633, hidden15 ~0.569895
  - alpha=0.90: [0.899010, 0.604725, 0.641468, 0.597989], latent15 ~0.487399, hidden15 ~0.568715
  - alpha=0.95: [0.973388, 0.623336, 0.684691, 0.632323], latent15 ~0.414012, hidden15 ~0.585417
- Conclusion: simple temporal smoothing is not a viable stabilizer. It hurts early trajectory tracking and does not prevent frame-15 divergence. Keep only as a negative env-gated diagnostic unless removed later.

### DPM trace diagnostic and dtype parity attempt
- Added per-DPM-step sample dumps for canonical and C++ (`frame_XXXX_dpm_step_NN_sample`) to isolate where diffusion divergence begins. C++ emits them through an optional DPM solver trace callback when `VIBEVOICE_KUGELAUDIO_DUMP_FRAMES` includes the frame.
- DPM trace on f16/CUDA teacher-forced 2-frame oracle:
  - frame 0 stays near exact through all 20 SDE steps (final mean_abs ~0.00031, cos ~0.999998).
  - frame 1 starts close but accumulates late-step divergence: step 10 mean_abs ~0.00229, step 14 ~0.0100, final step 19 ~0.0108, cos ~0.999499.
- Added canonical-like f16 sample casting after each scheduler step (`cast_sample_to_f16`) and f16 quantization of timestep sinusoidal inputs in C++ DPM. This mirrors canonical f16 behavior where scheduler output is cast back to model output dtype. It did not materially improve frame-1 DPM trace, so the remaining error is likely diffusion-head/math kernel parity or sensitivity to small condition differences rather than scheduler dtype alone.
- Remote DPM trace artifacts:
  - canonical: `runs/oracle-dpmtrace/canonical-f16`
  - C++ trace after cast: `runs/oracle-dpmtrace/cpp-f16-tfstep-castf16-tsf16`

### Diffusion-head model-output trace and teacher-cond isolation
- Added raw diffusion-head model-output dumps: `frame_XXXX_dpm_step_NN_model_output` for both canonical and C++.
- Re-ran 2-frame f16/CUDA DPM trace with canonical noise and teacher-forced canonical step embeddings:
  - frame 0 raw model output is near exact across all steps.
  - frame 1 raw model output starts close but late DPM steps diverge (e.g. step 14 model mean_abs ~0.0259, sample mean_abs ~0.0100; final sample mean_abs ~0.0108).
- C++ CPU vs CUDA produced very similar DPM trace errors, so this is not primarily a ggml CUDA backend issue.
- Added `VIBEVOICE_KUGELAUDIO_TEACHER_DIFFUSION_COND=1` diagnostic to load canonical `frame_XXXX_diffusion_cond_pos` before DPM sampling.
- With teacher-forced diffusion cond + teacher-forced step embeddings + canonical noise, frame-1 DPM trace becomes near exact again (step 14 sample mean_abs ~0.000394, final sample mean_abs ~0.000567). Therefore the late DPM divergence was caused mainly by tiny LM hidden/condition vector differences being amplified by diffusion, not by the scheduler itself.
- Remote artifacts:
  - canonical trace: `runs/oracle-dpmtrace2/canonical-f16`
  - C++ tf-step trace: `runs/oracle-dpmtrace2/cpp-f16-tfstep`
  - C++ CPU tf-step trace: `runs/oracle-dpmtrace2/cpp-f16-cpu-tfstep`
  - C++ tf-step+tf-cond trace: `runs/oracle-dpmtrace2/cpp-f16-tfstep-tfcond`

### Diffusion-cond blend and temporal-cond stabilizer diagnostics
- Added `VIBEVOICE_KUGELAUDIO_BLEND_DIFFUSION_COND=alpha` diagnostic. It loads canonical `frame_XXXX_diffusion_cond_pos` and feeds `canonical * (1-alpha) + cpp * alpha` into DPM. This is an oracle-only upper-bound test, analogous to step-embedding blend.
- F16/CUDA 16-frame cond-blend results with canonical noise and free-running step embeddings:
  - alpha=0.00: cond15/latent15/step15/hidden15 ~ 1.0 / 0.999999 / 1.0 / 0.999996
  - alpha=0.50: cond15 ~0.999997, latent15 ~0.999998, hidden15 ~0.999988
  - alpha=0.75: cond15 ~0.999928, latent15 ~0.999970, hidden15 ~0.999902
  - alpha=0.85: cond15 ~0.998952, latent15 ~0.999705, hidden15 ~0.998936
  - alpha=0.90: cond15 ~0.990521, latent15 ~0.995897, hidden15 ~0.994025
  - alpha=1.00: cond15 ~0.710930, latent15 ~0.616919, hidden15 ~0.610302
- Interpretation: the feedback trajectory can tolerate a lot of C++ condition vector, but the last 10% of condition-direction error is enough to cross the cliff. This reinforces that a useful stabilizer must target the condition/hidden direction, not just norm.
- Added `VIBEVOICE_KUGELAUDIO_TEMPORAL_DIFFUSION_COND=alpha` production-available temporal smoothing diagnostic. It failed similarly to step temporal smoothing:
  - alpha=0.95: cond [f4,f8,f12,f15] = [0.731892, 0.521002, 0.598972, 0.509417]
  - alpha=0.98: [0.817819, 0.473938, 0.559965, 0.511818]
  - alpha=0.99: [0.776212, 0.490647, 0.580921, 0.467723]
- Conclusion: naive temporal smoothing of either step embeddings or diffusion conditions is harmful. The oracle blends work because they correct direction toward canonical, not because they smooth or normalize.
- Remote artifacts:
  - `runs/oracle-16b/cpp-f16-condblend-alpha-*`
  - `runs/oracle-16b/compare-f16-condblend-alpha-*.{txt,json}`
  - `runs/oracle-16b/cpp-f16-temporal-cond-alpha-*`

### f16 step-embedding input cast diagnostic
- Hypothesis: canonical f16 inference feeds generated acoustic-connector step embeddings back into the LM as float16 tensors, while C++ was feeding f32 connector outputs. Added env-gated diagnostic `VIBEVOICE_KUGELAUDIO_CAST_STEP_EMBED_F16=1`, which rounds the generated step embedding to f16 before feeding it into the LM. Also added `VIBEVOICE_KUGELAUDIO_CAST_DIFFUSION_COND_F16=1` for condition casting.
- F16/CUDA 16-frame oracle with canonical noise, free-running step embeddings:
  - baseline free-running frame15: cond ~0.711, latent ~0.554, step ~0.665, hidden ~0.594
  - step-embed f16 cast frame15: cond ~0.985363, latent ~0.993897, step ~0.995342, hidden ~0.987717
  - cond-only f16 cast did not help (frame15 cond ~0.711, hidden ~0.610)
  - both cond+step cast matches step-only behavior.
- This is the first non-oracle production-available stabilizer that materially improves f16 free-running oracle tracking. It suggests dtype parity at the generated continuous feedback boundary is critical.
- q8 with step-embed f16 cast still diverges by frame 15 (cond ~0.693, hidden ~0.561), so this does not rescue q8 oracle tracking.
- Next: try f16 long-form listening sample with `VIBEVOICE_KUGELAUDIO_CAST_STEP_EMBED_F16=1`; consider making this the default for KugelAudio f16/CUDA if it does not harm shorter output.

### Long-form f16 step-cast listening feedback and larger-block samples
- User listening feedback for `/tmp/kugelaudio-longform-f16-stepcast-check/none-f16-stepcast-mw80-f64.wav`: good = no noise; bad = speaker identity still changes at each independent chunk seam. This confirms f16 step-embed casting stabilizes within-chunk feedback/noise, but it does not solve independent chunk restart identity drift.
- Tried f16 single-sequence with step-cast on CUDA workstation to avoid chunk restarts, but full-text prefill OOMed on 24 GiB RTX 3090 with f16 model: CUDA tried to allocate ~7.9 GiB during 694-token prompt prefill after loading the 18 GiB model. This is a RunPod/larger-VRAM candidate if we want to revisit one-sequence f16.
- Generated larger independent f16 step-cast samples to reduce seam count:
  - `mw160/f128`, 4 chunks, remote `runs/longform-f16-stepcast-largeblocks/none-f16-stepcast-mw160-f128.wav`, local `/tmp/kugelaudio-longform-f16-stepcast-largeblocks/none-f16-stepcast-mw160-f128.wav`, duration 68.87s, elapsed 165s, RMS 0.04558, peak 0.72897.
  - `mw224/f160`, 3 chunks, remote `runs/longform-f16-stepcast-largeblocks/none-f16-stepcast-mw224-f160.wav`, local `/tmp/kugelaudio-longform-f16-stepcast-largeblocks/none-f16-stepcast-mw224-f160.wav`, duration 64.40s, elapsed 154s, RMS 0.05847, peak 0.76218.
- Caveat: larger-block samples reduce speaker-reset count but are shorter than the 10-chunk mw80/f64 sample, so they may compress/omit text or hit per-chunk frame limits. Need listening/coverage check before treating either as a recommended profile.

### Natural-stop single-sequence diagnostic started
- Added env-gated diagnostic `VIBEVOICE_KUGELAUDIO_SINGLE_SEQUENCE_MIN_RATIO`, defaulting to `0.90` to preserve the prior min-frame suppression behavior for `--chunk-continuity single-sequence`. Setting it to `0` disables EOS/speech_end suppression and lets single-sequence stop naturally.
- Started remote CUDA workstation CPU run with f16 model, step-embed f16 cast, streamed final decoder, and `VIBEVOICE_KUGELAUDIO_SINGLE_SEQUENCE_MIN_RATIO=0`:
  - PID recorded in `runs/single-sequence-cpu-stepcast-natural/pid.txt`
  - output target `runs/single-sequence-cpu-stepcast-natural/singleseq-f16-stepcast-cpu-naturalstop-mw80-f64-stream.wav`
  - log `runs/single-sequence-cpu-stepcast-natural/singleseq-f16-stepcast-cpu-naturalstop-mw80-f64-stream.log`
- Purpose: measure canonical-aligned natural stop duration/energy without forced continuation, now that f16 step-cast improves the feedback path.

### Natural-stop f16 CPU single-sequence result
- Remote natural-stop run completed successfully with `VIBEVOICE_KUGELAUDIO_SINGLE_SEQUENCE_MIN_RATIO=0`, f16 model, CPU backend, step-embed f16 cast, streamed final decoder.
- Output synced locally:
  - remote `runs/single-sequence-cpu-stepcast-natural/singleseq-f16-stepcast-cpu-naturalstop-mw80-f64-stream.wav`
  - local `/tmp/kugelaudio-single-sequence-cpu-stepcast-natural/singleseq-f16-stepcast-cpu-naturalstop-mw80-f64-stream.wav`
- Natural stop occurred at frame 420/640 = 56.00s, elapsed 815s, rc=0. Energy stayed non-silent through the natural stop: overall RMS 0.04386, peak 0.63602, last 50-56s RMS 0.04322.
- This confirms the previous min90 single-sequence silence from 60-76.8s was largely forced off-policy continuation after natural EOS, not useful canonical-aligned speech. Single-sequence f16 step-cast is stable until its natural stop but does not cover full long text.

### Canonical PyTorch unchunked longform natural-stop result
- Canonical PyTorch unchunked longform run completed on CUDA workstation with cfg=1.0 and same raw reference/text.
  - remote `runs/canonical-longform-natural/canonical-natural-cfg1.wav`
  - local `/tmp/kugelaudio-canonical-longform-natural/canonical-natural-cfg1.wav`
  - elapsed 90s, rc=0
- Canonical audio duration is 39.60s, RMS 0.05484, peak 0.66406. This is shorter than C++ f16 single-sequence natural stop (56.00s, RMS 0.04386), so C++ is not stopping too early relative to canonical; if anything it continues longer for this long prompt/settings.
- Implication: full-text unchunked canonical does not cover the whole long text either. For complete longform coverage, chunking remains necessary; the quality issue is preserving identity across independent chunk restarts, not merely matching unchunked natural-stop length.

### Promoted coherent f16 single-sequence defaults
- Promoted the f16 step-embedding feedback cast from env-only diagnostic to the default CLI behavior for KugelAudio f16 artifacts. The CLI sets `VibeVoiceTTSParams::cast_step_embed_f16=1` when `quantization_hint=f16`; `VIBEVOICE_KUGELAUDIO_CAST_STEP_EMBED_F16=0` remains available to reproduce the older unstable path. q8_0 remains off by default.
- Changed `--chunk-continuity single-sequence` default min-frame ratio from forced `0.90` to canonical-aligned natural stop (`0.0`). `VIBEVOICE_KUGELAUDIO_SINGLE_SEQUENCE_MIN_RATIO=0.90` remains available only for the old forced-continuation diagnostic.
- README now documents the coherent long-prompt command without requiring diagnostic env vars: CPU backend, f16 model, `--chunk-continuity single-sequence`, and `--final-decoder-backend stream`.
- Remote CUDA smoke verified that an f16 CLI run logs `step_embed_f16=on` and `tts_15b: rounding generated step embeddings to f16 before LM feedback` by default.

### Default coherent command regression
- Reran the coherent single-sequence command using only the new defaults (no `VIBEVOICE_KUGELAUDIO_CAST_STEP_EMBED_F16`, no `VIBEVOICE_KUGELAUDIO_SINGLE_SEQUENCE_MIN_RATIO`) on CUDA workstation CPU:
  - remote `runs/single-sequence-default-natural/singleseq-f16-default-natural-mw80-f64-stream.wav`
  - local `/tmp/kugelaudio-single-sequence-default-natural/singleseq-f16-default-natural-mw80-f64-stream.wav`
  - elapsed 645s, rc=0, natural stop at frame 420/640 = 56.00s
  - sha256 `0ca8de8b42b678fb12ba2671899fccaaa9793043796ad6859bcd606e577c1001`
- The default-output WAV is byte-identical to the prior env-gated natural-stop f16 step-cast sample, confirming the promoted defaults reproduce the "awesome" path.
- Added CLI regression coverage in `test_kugelaudio_cli_chunking_seed`: a f16-path policy probe asserts `quantization_hint=f16`, `step_embed_f16=on`, the runtime f16 feedback log, and `min_ratio=0.000` for single-sequence natural stop.

### Large natural blocks default experiment
- Ran full-coverage compromise experiments using promoted defaults (no diagnostic env vars), f16 model, CUDA backend, `--chunk-continuity none`, large chunk budgets, streamed GPU final decode via auto:
  - `mw224/f192`: remote `runs/large-natural-blocks-default/naturalblocks-f16-default-mw224-f192.wav`, local `/tmp/kugelaudio-large-natural-blocks-default/naturalblocks-f16-default-mw224-f192.wav`; 3 chunks; duration 77.20s; elapsed 203s; RMS 0.05943; peak 0.76218; all 3 chunks hit the 192-frame budget.
  - `mw320/f240`: remote `runs/large-natural-blocks-default/naturalblocks-f16-default-mw320-f240.wav`, local `/tmp/kugelaudio-large-natural-blocks-default/naturalblocks-f16-default-mw320-f240.wav`; 2 chunks; duration 50.87s; elapsed 131s; RMS 0.08126; peak 0.74899; second chunk naturally stopped at frame 140.
- Current practical recommendation for fuller coverage with fewer speaker resets: try `--chunk-continuity none --max-words-per-chunk 224 --max-frames 192 --overlap-sentences 1 --crossfade-ms 60` and listen/check coverage. The 2-block `mw320/f240` profile is likely too compressed/early-stopping for the fixture.

### Listening feedback: large natural blocks still reset speaker
- User listening feedback:
  - `/tmp/kugelaudio-large-natural-blocks-default/naturalblocks-f16-default-mw224-f192.wav`: speaker identity changes and chunks vary in volume.
  - `/tmp/kugelaudio-large-natural-blocks-default/naturalblocks-f16-default-mw320-f240.wav`: chunk ends are cut off and speaker identity changes.
- Conclusion: larger independent chunks reduce seam count but do not solve speaker identity resets. The 2-block profile also creates unacceptable cutoffs/coverage compression. Volume leveling could polish stitching, but it would not solve identity.
- Tried stronger CFG as a reference/text anchoring experiment for the 3-block profile (`mw224/f192`):
  - cfg=2.0: remote `runs/large-natural-blocks-cfg/naturalblocks-f16-mw224-f192-cfg2p0.wav`, local `/tmp/kugelaudio-large-natural-blocks-cfg/naturalblocks-f16-mw224-f192-cfg2p0.wav`; 3 chunks; duration 77.20s; elapsed 256s; RMS 0.07705; peak 0.87888.
  - cfg=3.0: remote `runs/large-natural-blocks-cfg/naturalblocks-f16-mw224-f192-cfg3p0.wav`, local `/tmp/kugelaudio-large-natural-blocks-cfg/naturalblocks-f16-mw224-f192-cfg3p0.wav`; 3 chunks; duration 77.20s; elapsed 248s; RMS 0.09688; peak 0.99997 (near clipping).
- Listen to cfg=2.0 first; cfg=3.0 is probably too hot/clippy even if identity improves.

### Listening feedback: CFG anchors identity partly, high CFG overacts
- User listening feedback:
  - `naturalblocks-f16-mw224-f192-cfg2p0.wav`: identity still changes, but less severely; speakers sound like related voices rather than unrelated strangers. Speech is too fast, probably too much text per chunk for the frame budget.
  - `naturalblocks-f16-mw224-f192-cfg3p0.wav`: strong changes, too much intonation/pathos; not viable.
- Follow-up speed/coverage ladder at cfg=2.0:
  - `mw224/f256`: remote `runs/large-natural-blocks-cfg-speed/naturalblocks-f16-mw224-f256-cfg2p0.wav`, local `/tmp/kugelaudio-large-natural-blocks-cfg-speed/naturalblocks-f16-mw224-f256-cfg2p0.wav`; 3 chunks; duration 102.80s; elapsed 335s; RMS 0.07795; peak 0.87888; all chunks hit 256-frame budget.
  - `mw160/f192`: remote `runs/large-natural-blocks-cfg-speed/naturalblocks-f16-mw160-f192-cfg2p0.wav`, local `/tmp/kugelaudio-large-natural-blocks-cfg-speed/naturalblocks-f16-mw160-f192-cfg2p0.wav`; 4 chunks; duration 103.00s; elapsed 327s; RMS 0.07626; peak 0.87057; all chunks hit 192-frame budget.
- Listen order: try `mw224/f256/cfg2.0` first because it preserves the 3-seam profile but gives each block more time; try `mw160/f192/cfg2.0` if the former still feels too compressed or has cutoff issues.

### F16 segmented-state retry result
- F16 CPU segmented-state with default step-embed f16 feedback completed, but output is effectively near-silent and not viable.
  - remote `runs/segmented-state-f16-stepcast/segstate-f16-cpu-default-mw160-f128.wav`
  - local `/tmp/kugelaudio-segmented-state-f16-stepcast/segstate-f16-cpu-default-mw160-f128.wav`
  - settings: `--chunk-continuity segmented-state --max-words-per-chunk 160 --max-frames 128 --cfg 1.0 --steps 20 --final-decoder-backend stream`, CPU backend, f16 model
  - chunks=4, frames=512, duration 68.27s, elapsed 768s, rc=0
  - RMS 0.00112, peak 0.03073; all 10s windows around 0.001 RMS
- Conclusion: f16 step-cast does not rescue segmented-state; it avoids independent restarts structurally but is off-distribution and collapses to near-silence. Do not recommend.

### Latent refinement prototype wired
- Implemented opt-in latent img2img-style refinement diagnostics:
  - CLI flags `--latent-refine-strength X` and `--latent-refine-steps N`.
  - Runtime params `latent_refine_strength` / `latent_refine_steps`, default disabled.
  - New DPM entry `dpm_solver_sample_from_step(...)` continues the existing scheduler from an intermediate timestep.
  - Generation stores per-frame diffusion conditions when refinement is enabled, then refines first-pass latents before final decode only; LM feedback still uses first-pass latents, so this does not carry generated waveform/state into future chunks.
- Smoke-tested locally with q8 CPU chunked toy input and `--latent-refine-strength 0.5`; output succeeded and logs included `latent refinement frames=... start_step=... reverse_steps=...`.

### Latent refinement first real run result
- F16 CPU latent refinement run completed:
  - remote `runs/latent-refine-f16-cpu/refine-f16-cpu-mw160-f192-cfg2-s020.wav`
  - local `/tmp/kugelaudio-latent-refine-f16-cpu/refine-f16-cpu-mw160-f192-cfg2-s020.wav`
  - settings: `mw160/f192`, `cfg=2.0`, `--latent-refine-strength 0.20`, `steps=20`, CPU backend, f16 model, `--chunk-continuity none`, streamed final decoder
  - chunks=4, all chunks hit 192-frame budget, duration 103.00s, elapsed 2071s, rc=0
  - refinement mapping: start_step=16, reverse_steps=4, timestep=200 per chunk
  - RMS 0.07510, peak 0.83316; unrefined comparator `naturalblocks-f16-mw160-f192-cfg2p0.wav` RMS 0.07626, peak 0.87057
- Objective stats suggest refinement strength 0.20 mostly preserves timing/energy and slightly lowers peak. Needs listening for identity/timbre/seam effect.

### Latent refinement listening feedback and lower-strength ladder
- User listening feedback for `refine-f16-cpu-mw160-f192-cfg2-s020.wav`: chunk identities are more aligned, but audio sounds somewhat more robotic. Interpretation: refinement is moving timbre/identity in the right direction but strength=0.20 / four reverse steps is too strong or too smoothing.
- Since CUDA workstation VRAM was free again, ran lower-strength f16 CUDA ladder on the same `mw160/f192/cfg2.0` profile:
  - `s=0.10`: remote `runs/latent-refine-f16-cuda-lower/refine-f16-cuda-mw160-f192-cfg2-s010.wav`, local `/tmp/kugelaudio-latent-refine-f16-cuda-lower/refine-f16-cuda-mw160-f192-cfg2-s010.wav`; reverse_steps=2, timestep=100; duration 103.00s; elapsed 344s; RMS 0.07659; peak 0.85397.
  - `s=0.05`: remote `runs/latent-refine-f16-cuda-lower/refine-f16-cuda-mw160-f192-cfg2-s005.wav`, local `/tmp/kugelaudio-latent-refine-f16-cuda-lower/refine-f16-cuda-mw160-f192-cfg2-s005.wav`; reverse_steps=1, timestep=50; duration 103.00s; elapsed 329s; RMS 0.07654; peak 0.88229.
- Listen order: s=0.10 first (likely balance), then s=0.05 if s=0.10 is still robotic or too altered. Unrefined comparator remains `/tmp/kugelaudio-large-natural-blocks-cfg-speed/naturalblocks-f16-mw160-f192-cfg2p0.wav`.

### Latent refinement lower-strength listening feedback
- User listening feedback: `s=0.10` is better than `s=0.05`, and `s=0.20` aligned identities more but sounded robotic. However none of the latent-refinement variants reaches acceptable long-form quality; identity drift remains a blocker.
- Important caveat discovered/confirmed: most long-form experiments used `tests/fixtures/reference_sine.wav`, a 3-second synthetic sine-wave fixture, not a real speech reference. That fixture is useful for deterministic wiring but cannot provide meaningful speaker identity conditioning. Before declaring the long-form quality ceiling, rerun the best coherent/chunked/refinement profiles with a clean real spoken reference WAV.

### Real spoken reference: private_real_ref.wav runs
- User provided real reference audio `<private-real-reference.wav>`; local file is 48 kHz mono, 10.99s, RMS 0.06193, peak 0.34003. Synced to remote `runs/real-ref-private/private_real_ref.wav`.
- Generated three f16 samples with promoted defaults:
  - Coherent single-sequence natural stop, CPU backend: local `/tmp/kugelaudio-real-ref-private/singleseq-f16-realref-natural-mw80-f64.wav`, remote `runs/real-ref-private/singleseq-f16-realref-natural-mw80-f64.wav`; natural stop frame 405, duration 54.00s, elapsed 804s, RMS 0.01827, peak 0.25568.
  - Chunked large-ish baseline, CUDA backend: local `/tmp/kugelaudio-real-ref-private/chunks-f16-realref-mw160-f192-cfg2.wav`, remote `runs/real-ref-private/chunks-f16-realref-mw160-f192-cfg2.wav`; 4 chunks, duration 103.00s, elapsed 323s, RMS 0.04884, peak 0.26465.
  - Latent refinement s=0.10, CUDA backend: local `/tmp/kugelaudio-real-ref-private/refine-f16-realref-mw160-f192-cfg2-s010.wav`, remote `runs/real-ref-private/refine-f16-realref-mw160-f192-cfg2-s010.wav`; 4 chunks, duration 103.00s, elapsed 336s, RMS 0.04877, peak 0.27905; refinement start_step=18/reverse_steps=2/timestep=100 for each chunk.
- Listen order: single-sequence first for target identity/coherence, then chunked baseline vs s=0.10 refinement to assess whether real reference reduces chunk identity drift enough.

### Real reference listening: quality good, chunk cutoffs are frame-budget issue
- User listening feedback for real `private_real_ref.wav` reference: both unrefined and latent-refine s=0.10 chunked outputs are very good quality; user prefers chunked versions over single-sequence. Remaining recurring issue: the phrase `Nothing dramatic happens, but every detail matters because ...` cuts short in chunked generations.
- Diagnosis: the good chunked profiles (`mw160/f192`) were severely frame-budget constrained: 160 words / 25.6s per chunk is too fast, and all chunks hit max frame budget. The recurring cutoff is likely not a seam algorithm failure but insufficient per-chunk duration budget.
- Generated slower real-ref frame-budget ladder at cfg=2.0:
  - `mw160/f320`: local `/tmp/kugelaudio-real-ref-private-framebudget/chunks-f16-realref-mw160-f320-cfg2p0.wav`, remote `runs/real-ref-private-framebudget/chunks-f16-realref-mw160-f320-cfg2p0.wav`; 4 chunks; duration 168.47s; elapsed 524s; RMS 0.04888; peak 0.26465. First chunk now naturally stopped at frame 299 instead of maxing, later chunks hit 320.
  - `mw120/f256`: local `/tmp/kugelaudio-real-ref-private-framebudget/chunks-f16-realref-mw120-f256-cfg2p0.wav`, remote `runs/real-ref-private-framebudget/chunks-f16-realref-mw120-f256-cfg2p0.wav`; 5 chunks; duration 167.47s; elapsed 525s; RMS 0.04626; peak 0.31177. First chunk naturally stopped at frame 226, later chunks hit 256.
- Listen order: try `mw160/f320` first because it keeps 4 chunks and likely fixes the first phrase cutoff; try `mw120/f256` if later chunks still sound rushed/cut.

### Real reference frame budget: last words cutoff follow-up
- User listening feedback: `/tmp/kugelaudio-real-ref-private-framebudget/chunks-f16-realref-mw120-f256-cfg2p0.wav` sounds better, but the last few words are cut off.
- Added PRD task for transcript coverage QA using an external ASR backend (e.g. faster-whisper) to detect missing final words / chunk-boundary omissions before recommending long-form profiles.
- Generated `mw120/f320/cfg2.0` to address last-word truncation:
  - local `/tmp/kugelaudio-real-ref-private-framebudget/chunks-f16-realref-mw120-f320-cfg2p0.wav`
  - remote `runs/real-ref-private-framebudget/chunks-f16-realref-mw120-f320-cfg2p0.wav`
  - 5 chunks, duration 190.67s, elapsed 593s, RMS 0.04619, peak 0.31177
  - all chunks stopped naturally before the 320-frame cap: frames 226, 303, 288, 296, 311. This should fix the tail cutoff if the cutoff was frame-budget-related.

### New readable German long-form eval text
- User provided a new preferred long-form eval text in German about Saint-Malo/Rennes/Nantes heat records. Saved locally at `runs/longform_eval_de_saint_malo.txt` and should be used for future listening/eval runs instead of the synthetic sectioned English fixture when readability matters.

### Retired continuity revisit with real reference
- Re-enabled retired continuity modes behind explicit diagnostic env `VIBEVOICE_ENABLE_RETIRED_CONTINUITY=1` in CLI and runtime only; default CLI rejection remains unchanged and tests still pass.
- Revisited on real `private_real_ref.wav` reference with the new German long-form text, `mw90/f320`, `overlap_sentences=0`, `cfg=2.0`, f16 CUDA:
  - Baseline none: `/tmp/kugelaudio-real-ref-private-retired-revisit/none-f16-realref-de-mw90-f320-cfg2.wav`, 2 chunks, 52.33s, RMS 0.05244.
  - Latent-prefix cfg2: `/tmp/kugelaudio-real-ref-private-retired-revisit/latent-prefix-f16-realref-de-mw90-f320-cfg2.wav`, 23.19s, second chunk stops immediately at frame 0 after applying 9 latent-prefix frames; not viable.
  - Latent-prefix cfg1: `/tmp/kugelaudio-real-ref-private-retired-revisit/latent-prefix-f16-realref-de-mw90-f320-cfg1.wav`, 26.93s, second chunk also stops immediately at frame 0; not viable.
  - Tail-reference cfg2: `/tmp/kugelaudio-real-ref-private-retired-revisit/tail-reference-f16-realref-de-mw90-f320-cfg2.wav`, 51.40s, RMS 0.05157; listen against baseline for identity/noise.
  - Clean-tail-reference cfg2: `/tmp/kugelaudio-real-ref-private-retired-revisit/clean-tail-reference-f16-realref-de-mw90-f320-cfg2.wav`, 51.27s, RMS 0.05339; listen against baseline for identity/noise.
- Early conclusion from logs: latent-prefix remains structurally harmful even with real reference because the next chunk immediately predicts EOS. Tail/clean-tail are the only retired modes worth a quick listening comparison, but they still use generated waveform in future reference conditioning and must remain diagnostic-only unless proven safe.
- Listening update: user reported the three recommended comparison samples (`none`, `tail-reference`, `clean-tail-reference`) all sound "super awesome" on this limited 2-chunk real-reference run, with a preference for `tail-reference-f16-realref-de-mw90-f320-cfg2.wav`. This reopens waveform-tail continuity as a real-reference-only diagnostic worth stress-testing, but not as a default/promoted mode yet.

### Tail-reference real-reference stress test
- Ran explicit diagnostic stress tests with `VIBEVOICE_ENABLE_RETIRED_CONTINUITY=1`, real `private_real_ref.wav`, German Saint-Malo text, f16 CUDA, `cfg=2.0`, `steps=20`, `max_frames=320`, `overlap_sentences=0`, punctuation pauses, 60 ms crossfade.
- Local outputs in `/tmp/kugelaudio-real-ref-private-tail-stress/`; remote outputs in `runs/real-ref-private-tail-stress/`.
- 3-chunk `mw60` ladder:
  - `none-f16-realref-de-mw60-f320-cfg2.wav`: 53.33s, RMS 0.04968, ASR ordered 0.934, unique recall 0.933, tail 0.875, missing 9.
  - `tail-reference-f16-realref-de-mw60-f320-cfg2-tail500.wav`: 59.20s, RMS 0.07089, ASR ordered 0.920, unique recall 0.905, tail 0.792, missing 11. Short tail sounds/runs hotter by RMS and is worst by ASR.
  - `tail-reference-f16-realref-de-mw60-f320-cfg2-tail1200.wav`: 52.80s, RMS 0.05006, ASR ordered 0.956, unique recall 0.952, tail 0.917, missing 6. Best overall ASR in mw60 ladder.
  - `tail-reference-f16-realref-de-mw60-f320-cfg2-tail2000.wav`: 58.40s, RMS 0.04411, ASR ordered 0.949, unique recall 0.943, tail 0.958, missing 7. Best tail coverage but lower RMS/longer duration.
- 4-chunk `mw40` stress:
  - `none-f16-realref-de-mw40-f320-cfg2.wav`: 58.59s, RMS 0.04665, ASR ordered 0.905, unique recall 0.895, tail 0.833, missing 13.
  - `tail-reference-f16-realref-de-mw40-f320-cfg2-tail1200.wav`: 58.06s, RMS 0.05315, ASR ordered 0.942, unique recall 0.924, tail 0.958, missing 8. Tail-reference remains better than none on coverage and steadier by 10s RMS bins in this stress.
- Current best real-reference tail diagnostic from objective metrics: `tail-reference` with `continuity_tail_ms=1200`; keep env-gated pending listening on mw60/mw40 stress samples and at least one longer text.

### Tail-reference stress listening update
- User listened to the first two recommended stress outputs and reported they sound best:
  - `/tmp/kugelaudio-real-ref-private-tail-stress/tail-reference-f16-realref-de-mw60-f320-cfg2-tail1200.wav`
  - `/tmp/kugelaudio-real-ref-private-tail-stress/tail-reference-f16-realref-de-mw40-f320-cfg2-tail1200.wav`
- Listening note: both have some very low noise, but it is almost unnoticeable. This confirms the objective ASR/RMS result: `tail-reference` with `continuity_tail_ms=1200` is currently the strongest real-reference long-form continuity profile tested.
- Keep the mode diagnostic/env-gated for now, but it is no longer considered a dead end for real spoken references. Next validation should use a longer/more varied text or multiple real refs to check whether generated-tail noise accumulates beyond 4 chunks.

### Chunk boundary isolation: abrupt ends and long gaps
- Added verbose chunk-boundary audio diagnostics in `src/vibevoice_tts.cpp`. With `--verbose`, each generated chunk now logs duration, head/tail quiet duration, head/tail RMS, and each stitch boundary logs pause/crossfade plus estimated audible gap and tail/head RMS. This is diagnostic-only and does not alter waveform bytes for fixed settings.
- Isolated current boundary behavior on `tail-reference-f16-realref-de-mw40-f320-cfg2-tail1200`:
  - Every generated chunk starts with about 260-270 ms of silence (`head_quiet_ms`). This makes stitched pauses longer than requested even before explicit pause insertion.
  - Some chunks also end with generated silence; chunk 2 had about 870 ms of trailing silence, causing a very long 2->3 gap.
  - Other chunks end with high-energy speech and no trailing quiet; chunk 1 tail50 RMS 0.0315/peak 0.127 and chunk 3 tail50 RMS 0.0601/peak 0.132. The current punctuation stitch inserts zeros after those tails, so the perceived end can be abrupt.
  - Current `punctuation` + 60 ms crossfade inserts pause first and then crossfades the next chunk into the end of that silence. It does not fade the previous chunk out when `pause_ms >= crossfade_ms`, so it does not fix abrupt voiced chunk endings.
- Quick stitch-profile probes using the same generation settings:
  - Existing punctuation/xf60: `/tmp/kugelaudio-real-ref-private-tail-stress/tail-reference-f16-realref-de-mw40-f320-cfg2-tail1200.wav`, ASR ordered 0.942, recall 0.924, tail 0.958, missing 8.
  - `--pause-mode none --crossfade-ms 60`: `/tmp/kugelaudio-real-ref-private-tail-stress/tail-reference-f16-realref-de-mw40-f320-cfg2-tail1200-pausenone.wav`, 57.42s, ASR ordered 0.920, recall 0.905, tail 0.958, missing 11.
  - `--pause-mode none --crossfade-ms 200`: `/tmp/kugelaudio-real-ref-private-tail-stress/tail-reference-f16-realref-de-mw40-f320-cfg2-tail1200-pausenone-xf200.wav`, 57.00s, ASR ordered 0.949, recall 0.933, tail 0.958, missing 7. This is worth listening because it reduces inserted pauses and uses the next chunk's inherent leading silence as the gap/fade bed.
- Likely fix direction: separate boundary cleanup from generation. Trim excessive generated leading/trailing silence to a small target and add an explicit fade-to-silence for high-energy chunk tails before any programmed pause. Do not change default parity behavior until listening confirms the boundary cleanup improves seams without clipping words.

### Implemented opt-in chunk-boundary cleanup
- Added opt-in CLI/runtime controls:
  - `--chunk-boundary-cleanup none|trim-fade` (default `none`)
  - `--chunk-boundary-leading-silence-ms` (default 200 when cleanup is enabled)
  - `--chunk-boundary-trailing-silence-ms` (default 300 when cleanup is enabled)
  - `--chunk-boundary-fade-ms` (default 15 when cleanup is enabled)
- Cleanup runs after chunk generation and before stitching. It trims only excess generated leading silence on chunks after the first, trims only excess generated trailing silence on chunks before the last, then applies short fade-in/fade-out ramps at seam chunks. Default disabled path remains byte-identical.
- First aggressive cleanup probe (`leading=80 trailing=120 fade=25`) shortened the mw40 tail-reference sample from 58.06s to 56.76s and reduced estimated gaps, but ASR dropped from ordered 0.942/recall 0.924/missing 8 to ordered 0.920/recall 0.905/missing 11. Treat this as too aggressive.
- Safer cleanup probe (`leading=200 trailing=300 fade=15`) produced `/tmp/kugelaudio-real-ref-private-tail-stress/tail-reference-f16-realref-de-mw40-f320-cfg2-tail1200-cleanup-h200t300f15.wav`, 57.30s, RMS 0.05349. ASR matched baseline (`ordered=0.942`, `recall=0.924`, `tail=0.958`, `missing=8`) while reducing the worst estimated audible gap from ~1340 ms to ~700 ms and smoothing high-energy tails. These values are now the cleanup defaults.
- The earlier `pause-mode none + crossfade_ms=200` no-cleanup probe still has the best ASR of the boundary variants (`ordered=0.949`, `recall=0.933`, `missing=7`) and should remain in the listening set: `/tmp/kugelaudio-real-ref-private-tail-stress/tail-reference-f16-realref-de-mw40-f320-cfg2-tail1200-pausenone-xf200.wav`.

### EOS guard probe for abrupt chunk endings
- Added opt-in diagnostic `--chunk-eos-guard-frames N` (default 0). It suppresses the first N `speech_end`/EOS predictions after a chunk has emitted at least one audio frame. This is intentionally not default behavior.
- Rationale: a high-energy decoded tail can mean the model stopped before producing natural trailing silence. Direct high-energy-aware suppression is hard inside the LM loop because decoded waveform energy is only known after final acoustic decoding; the simple guard tests whether ignoring EOS for one extra frame helps.
- Probe run: `/tmp/kugelaudio-real-ref-private-tail-stress/tail-reference-f16-realref-de-mw40-f320-cfg2-tail1200-eosguard1.wav`, same mw40 tail-reference profile plus `--chunk-eos-guard-frames 1`.
- Result: not promising as an unconditional guard. Each chunk suppressed one EOS, generated 1-8 additional frames, then stopped, but the extra frames were mostly silence. Tail quiet grew to about 840-1050 ms and estimated seam gaps became worse (about 1.1-1.5s). Duration 58.59s, RMS 0.05598.
- Conclusion: do not use unconditional EOS forcing as the seam fix. A truly high-energy-aware version would need a post-decode tail-energy check plus targeted retry/continuation, or a cheap latent/decoder-tail proxy that is proven to correlate with waveform energy. For now, boundary cleanup/crossfade/pause shaping is safer.
- Follow-up after reading upstream issue #4 (`Speech is rigidly cut off at the end`): canonical PyTorch users report final phoneme cut-off too, especially Russian final sounds, so this is not just C++ stitching. Tested `--chunk-eos-guard-frames 1` combined with safer boundary cleanup. Output `/tmp/kugelaudio-real-ref-private-tail-stress/tail-reference-f16-realref-de-mw40-f320-cfg2-tail1200-eosguard1-cleanup.wav` had ASR ordered 0.927/recall 0.914/missing 10, worse than cleanup-only and baseline (both ordered 0.942/recall 0.924/missing 8). Extra guarded frames were mostly silence and then cleanup trimmed them. Keep EOS guard diagnostic-only; upstream issue supports investigating a smarter final-tail continuation/retry, not unconditional EOS suppression.
- Retried EOS guard after upstream issue #4 with larger guards under cleanup: `--chunk-eos-guard-frames 2` and `4` on the mw40 tail-reference profile. Outputs:
  - `/tmp/kugelaudio-real-ref-private-tail-stress/tail-reference-f16-realref-de-mw40-f320-cfg2-tail1200-eosguard2-cleanup.wav`: 57.27s, RMS 0.05338, ASR ordered 0.934, recall 0.924, tail 0.958, missing 9.
  - `/tmp/kugelaudio-real-ref-private-tail-stress/tail-reference-f16-realref-de-mw40-f320-cfg2-tail1200-eosguard4-cleanup.wav`: 57.00s, RMS 0.05675, ASR ordered 0.927, recall 0.914, tail 0.958, missing 10.
  - Baseline/cleanup-only remains better: ordered 0.942, recall 0.924, missing 8.
- Logs show larger guards mostly produce extra silent tail (`tail50_rms=0`, tail quiet ~0.9-1.4s before cleanup), not useful phoneme continuation. This strengthens the conclusion: EOS suppression alone is the wrong mechanism; upstream cut-off probably needs a post-decode continuation/retry or training-aligned end-padding behavior, not blind stop-token masking.

### Final-cut seed/text mining pass
- Mined short German/English phrases with real `private_real_ref.wav`, f16 CUDA, `cfg=2.0`, `steps=20`, `max_frames=96`, and tail-cut metrics (tail RMS/peak + trailing quiet). Local artifacts:
  - `/tmp/kugelaudio-final-cut-mining/`
  - `/tmp/kugelaudio-final-cut-mining-focused/`
  - `/tmp/kugelaudio-final-cut-mining-seeds/`
  - `/tmp/kugelaudio-final-cut-mining-english/`
  - `/tmp/kugelaudio-final-cut-mining-english-cfg3/`
- Best reproducible German cutoff candidates found:
  - Text: `Ich bringe das Paket zurück.` seed `1007`, output `/tmp/kugelaudio-final-cut-mining-seeds/00-de-seed1007.wav`, EOS at frame 12, duration 1.60s, trailing quiet 0 ms, tail50 RMS 0.06675, tail100 RMS 0.11146, tail200 RMS 0.12134, tail50 peak 0.2095. This is the strongest high-energy end-tail candidate.
  - Same text seed `1003`, output `/tmp/kugelaudio-final-cut-mining-seeds/00-de-seed1003.wav`, EOS at frame 15, duration 2.00s, trailing quiet 0 ms, tail50 RMS 0.03751, tail100 RMS 0.06723, tail200 RMS 0.08418, tail50 peak 0.1203.
- ASR on the German candidates still reports full word coverage (`ordered_coverage=1.0`), so ASR is not sufficient for final-phoneme cutoff. Use waveform-tail metrics plus listening for this fixture.
- English mining so far did not produce high-energy final-tail candidates with the private real-reference ref. Tested `cfg=2.0` and `cfg=3.0` phrase sets such as `Do not cut the word back.`, `Please say the final sound.`, `This is the final test.`, `It comes back.`, `Say it back.`, and `End with a hard cut.`; all mined English outputs had substantial trailing quiet and tail50 RMS near zero. Keep English in the fixture set as negative/control cases unless a real English cutoff seed appears later.
- User listening verification: both mined German candidates are confirmed cutoff cases:
  - `/tmp/kugelaudio-final-cut-mining-seeds/00-de-seed1007.wav`
  - `/tmp/kugelaudio-final-cut-mining-seeds/00-de-seed1003.wav`
- Treat these as the positive short cutoff fixture pair for mitigation testing. ASR falsely reports full word coverage, so score them primarily by listening plus tail metrics (trailing quiet, final RMS/peak), with ASR only as a regression guard for omitted/repeated words.

### Final-cut remedy first ladder
- Tested remedies on confirmed cutoff fixtures `Ich bringe das Paket zurück.` seeds 1007 and 1003. Local outputs in `/tmp/kugelaudio-final-cut-remedies/`, remote in `runs/final-cut-remedies/`.
- Best text-padding remedy: ASCII adjacent ellipsis, text `Ich bringe das Paket zurück...`:
  - `ellipsis-seed1007.wav`: duration 3.07s, EOS frame 23, trailing quiet 840 ms, tail50/100/200 RMS 0, ASR full coverage.
  - `ellipsis-seed1003.wav`: duration 2.93s, EOS frame 22, trailing quiet 870 ms, tail50/100/200 RMS 0, ASR full coverage.
  - This is the only punctuation-padding variant tested so far that fixed both positive seeds without changing sentence type to question/exclamation. Note: Unicode ellipsis and spaced ellipsis did *not* fix seed 1003; use literal `...` adjacent to the last word.
- Other punctuation results:
  - `!` and `?` fixed tail metrics on both seeds but change utterance intent/prosody, so they are not a neutral remedy.
  - Double-period, semicolon, and `zurück .` fixed seed 1007 but remained/worsened cutoff on seed 1003.
  - Newline after the period was byte/metric-equivalent to baseline; no effect.
- EOS guard on the short fixture (`--chunk-eos-guard-frames 1/2`) also produced trailing silence and full ASR on both seeds, but earlier long-form tests show unconditional guard mostly adds silence and worsens coverage/gaps. Keep as negative/control, not preferred.
- Added post-process masking controls only for listening: `postfade200-sil300-seed1007.wav` and `postfade200-sil300-seed1003.wav` apply a 200 ms fade-out plus 300 ms silence to the cutoff baselines. This can soften the hard cut but cannot restore the missing phoneme; use only as a perceptual control.
- Listening set for user: baseline seeds 1007/1003 vs `ellipsis-seed1007/1003.wav`, plus optional `postfade200-sil300` controls.
- User listening verification: `ellipsis-seed1003.wav` and `ellipsis-seed1007.wav` both sound very good. This confirms terminal ASCII adjacent ellipsis (`...`) as the first promising final-cut remedy for the German positive fixtures. Next step should be an opt-in text end-padding mode that appends literal `...` only for generation text when enabled, then validate on long-form/chunk boundaries and English controls.

### Implemented opt-in text-end ellipsis padding
- Added CLI/runtime option `--text-end-padding none|ellipsis` (default `none`). The `ellipsis` mode normalizes generated text ends to terminal ASCII adjacent `...`: a final `.` is replaced by `...`; existing `...` is left unchanged. Source text/chunk planning remain unchanged.
- For non-chunked and single-sequence generation, padding applies to the full generation text. For chunked generation, padding applies only to the final generation chunk. A per-chunk ellipsis trial regressed long-form ASR (`ordered=0.912`, `missing=12` vs baseline `ordered=0.942`, `missing=8`), so it was not kept.
- Flag validation on short fixtures produced byte-identical output to the manually edited ellipsis prompts:
  - `text-end-padding-ellipsis-seed1007.wav` SHA matches `ellipsis-seed1007.wav` (`4acd0e...`).
  - `text-end-padding-ellipsis-seed1003.wav` SHA matches `ellipsis-seed1003.wav` (`547e0d...`).
- Long-form final-only validation: `/tmp/kugelaudio-real-ref-private-tail-stress/tail-reference-f16-realref-de-mw40-f320-cfg2-tail1200-textpad-final-ellipsis.wav`, duration 57.66s, ASR ordered 0.942/recall 0.924/tail 0.958/missing 8, matching baseline. Only final chunk prompt was padded.
