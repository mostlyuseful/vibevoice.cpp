# Project memory

## Decisions

### KugelAudio GGUF metadata migration contract
- Context: Slice 1 starts with converter-side KugelAudio support, but the current runtime/loader still keys off legacy `vibevoice.*` metadata and the GGUF arch string used by existing code.
- Chosen default: emit a dual metadata contract for KugelAudio conversion now:
  - new explicit `kugelaudio.*` keys, with `kugelaudio.checkpoint = kugelaudio-0-open` and `kugelaudio.schema_version = 1`
  - legacy `vibevoice.*` compatibility keys in the same GGUF during migration
  - keep GGUF writer arch as `vibevoice` until loader migration work lands
- Rejected alternatives:
  - emit only `kugelaudio.*` immediately: cleaner, but likely breaks the current loader before Slice 1 loader work is done
  - keep only `vibevoice.*`: preserves compatibility, but fails the spec/PRD requirement for an explicit KugelAudio metadata contract
  - switch GGUF arch to `kugelaudio` immediately: desirable later, but too risky before loader support exists
- Affected area: `spec.md` GGUF schema migration guidance and `prd.md` Slice 1 / “Define and implement the KugelAudio GGUF contract”.

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
