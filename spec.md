# Specification: KugelAudio ggml Port via `vibevoice.cpp`

## Summary
Adapt `localai-org/vibevoice.cpp` to support KugelAudio as the primary edge-inference path.

This effort is explicitly focused on a **ggml-based end-to-end port**, not generic edge-runtime exploration and not ExecuTorch/Vulkan as the main path.

The first milestone should be a **proof-of-concept KugelAudio port** that can run **the published open checkpoint `kugelaudio/kugelaudio-0-open` end-to-end** in the `vibevoice.cpp` codebase, with:
- single-speaker generation working
- the raw reference-audio voice-conditioning path working
- a clear migration path for later iterations

This project should preserve the important product capabilities that matter for practical edge inference while intentionally dropping lower-priority features that add migration cost without helping the main objective.

---

## Goals
- Make `vibevoice.cpp` the concrete ggml-based edge-inference path for KugelAudio.
- Reuse as much of `vibevoice.cpp` as practical instead of building a new ggml runtime from scratch.
- Support the published open checkpoint `kugelaudio/kugelaudio-0-open` end-to-end in v1.
- Preserve **voice cloning** as a core capability.
- Preserve a path for **aggressive quantization** beyond just `f16` / `bf16` / `q8_0` in later iterations.
- Preserve a path for **KV-cache quantization** in later iterations.
- Produce a reproducible conversion / runtime workflow for the supported KugelAudio checkpoint.
- Use `../kugelaudio-open` as the behavioral ground truth for divergence testing.

---

## Non-goals
- No attempt to make ExecuTorch/Vulkan the primary v1 path.
- No generic backend-agnostic edge-runtime abstraction effort.
- No from-scratch Vulkan compute port.
- No broad cross-platform hardware validation in v1.
- No parity with every current Python/UI feature in v1.
- No support for **pre-encoded named voices** or **pre-encoded `voice_cache`** in this port.
- No **AudioSeal watermarking** support in this port.
- No **language hint** feature in v1.
- No **multi-speaker dialog** support in v1.
- No **long-text chunking** support in v1.
- No training or distillation of smaller models in v1.
- No polished production packaging in v1.

---

## Users / actors
- Developers trying to run KugelAudio on edge-like local systems with constrained GPU/CPU resources.
- Operators who want a native-style local binary/runtime instead of the full Python/Hugging Face stack.
- Researchers validating whether KugelAudio can be made practical on ggml-based runtimes.

---

## Use cases / workflows

### 1. Single-speaker TTS on ggml runtime
A developer converts the supported KugelAudio checkpoint into the `vibevoice.cpp` format, runs a CLI call, and gets generated speech end-to-end.

### 2. Voice cloning from raw reference audio
A developer provides:
- input text
- raw reference audio

The port performs single-speaker voice-conditioned generation with raw voice prompting.

### 3. Quantized local inference
A developer runs the port using `f16` weights for parity evaluation or `q8_0` weights for end-to-end execution.

---

## Functional requirements

### V1 required scope

#### 1. Base port target: adapt `vibevoice.cpp`
The implementation should use `localai-org/vibevoice.cpp` as the starting point rather than building a fresh ggml runtime.

Expected implications:
- reuse existing ggml/Qwen/VibeVoice-style runtime pieces where possible
- adapt the converter, loader, prompt builder, and inference loop for KugelAudio
- treat this as a KugelAudio-specific port effort, not a generic framework project

#### 2. Supported model scope for v1
V1 should support **one KugelAudio checkpoint** end-to-end: `kugelaudio/kugelaudio-0-open`.

The spec should not require multi-checkpoint parity in the first milestone.

Acceptance target:
- the converted `kugelaudio/kugelaudio-0-open` model loads and generates audio in the ggml runtime, subject to the explicit evaluation criteria defined later in this spec

#### 3. Converter support for KugelAudio checkpoints
A conversion path must exist from the current KugelAudio model/checkpoint format into the ggml runtime format used by `vibevoice.cpp`.

This likely requires substantial changes because the existing converter appears tightly coupled to upstream VibeVoice.

Expected responsibilities:
- detect KugelAudio checkpoint/config structure
- map KugelAudio tensor names and metadata into the target ggml format
- encode enough metadata for runtime loading and feature gating
- reject unsupported checkpoints clearly

The converter should explicitly handle or document:
- Qwen backbone mapping
- diffusion head mapping
- acoustic tokenizer / decoder mapping
- semantic tokenizer mapping
- tokenizer special-token assumptions relevant to TTS generation
- the required GGUF metadata contract for the supported KugelAudio checkpoint

The GGUF metadata contract should be explicit. Naming does not need to reuse existing `vibevoice.*` keys, but the loader may accept both legacy and KugelAudio-specific metadata during migration.

#### 4. Runtime support for KugelAudio prompt/inference behavior
The runtime must be adapted to KugelAudio’s actual generation behavior rather than assuming upstream VibeVoice parity.

At minimum, the port must account for:
- the canonical KugelAudio prompt template semantics from `../kugelaudio-open`
- the canonical prompt sections:
  - system prompt
  - `Voice input:`
  - `Text input:`
  - `Speech output:`
- constrained token generation for the speech path
- canonical CFG behavior
- canonical speech-end handling
- diffusion-token handling
- final waveform decode path
- the raw-reference single-speaker voice-conditioning path

Prompt formatting should be ported as-is where practical.

#### 5. Voice cloning is required
Voice cloning must remain in scope.

Required v1 behavior:
- support **raw reference-audio prompting** as the main and only voice-conditioning path
- implement both acoustic and semantic conditioning if required for canonical behavior
- support only the single-reference, single-speaker path in v1

#### 6. Quantization support requirements
The effort must preserve a path for broader ggml-style quantization support, but v1 requirements are narrower.

V1 requirements:
- `f16` must retain a similar quality level to the canonical PyTorch implementation on the defined evaluation set
- `q8_0` must convert, load, and run end-to-end on the defined evaluation path
- the design should avoid hard-coding only `f16` / `bf16` / `q8_0`

### Explicitly out of v1 scope

#### Drop
- no support for pre-encoded named voice registries in this port
- no support for pre-encoded `voice_cache`
- no AudioSeal watermarking

#### Defer to v2
- long-text chunking
- language hints
- multi-speaker dialog
- KV-cache quantization as a user-facing/runtime feature
- additional quantization-format validation beyond `f16` parity and `q8_0` end-to-end execution

The v1 design should avoid blocking these later additions.

---

## Non-functional requirements
- The port must be reproducible by another developer from repo state plus documented inputs.
- The implementation should minimize invasive rewrites where reuse from `vibevoice.cpp` is practical.
- The supported feature subset must be documented honestly.
- Failure modes should be explicit rather than silent.
- Conversion and runtime tooling should be scriptable.
- The design should preserve headroom for later quantization and cache experiments.
- The implementation should be deterministic for fixed seed/settings on CPU.
- V1 acceptance is CLI-only; no stable embedding API is required in this milestone.

---

## Architecture / technical approach

### High-level approach
Use `vibevoice.cpp` as a migration base and adapt it for KugelAudio.

Expected work areas:
1. **Conversion layer**
   - adapt or replace VibeVoice-specific conversion logic
   - emit Kugel-compatible gguf/runtime metadata
   - support clear failure at conversion time for unsupported checkpoints
2. **Model loading layer**
   - accept KugelAudio metadata/config variants
   - optionally accept both legacy and KugelAudio-specific metadata during migration
   - gate unsupported features clearly at load time
3. **Prompt / tokenizer / special-token integration**
   - match KugelAudio’s TTS prompt conventions closely enough for working generation
   - port the canonical prompt sections as-is where practical
4. **Inference loop adaptation**
   - align AR token generation, constrained token set, CFG behavior, and speech-end behavior with KugelAudio
5. **Voice cloning path**
   - support the single-speaker raw-reference path with both acoustic and semantic conditioning
6. **Evaluation / divergence testing**
   - compare against `../kugelaudio-open` as the behavioral ground truth using fixed prompts, fixed reference audio, fixed seed, fixed generation settings, and the same checkpoint

### Why not a generic llama.cpp/ggml path?
A generic llama.cpp-only investigation is too abstract for this project.

`vibevoice.cpp` already appears to implement end-to-end VibeVoice-style generation in ggml, making it a much more relevant base for KugelAudio than pure LLM runtimes.

### Why not a custom Vulkan rewrite?
A from-scratch Vulkan port is high-risk and likely would require substantial custom runtime/kernel work.

This should remain a deferred research note, not part of the main implementation path.

---

## Data model / state / persistence
No new persistent application database is required.

Primary artifacts:
- source KugelAudio checkpoint/config
- converted gguf / runtime model files
- raw reference audio inputs
- generated audio outputs
- evaluation logs / summaries

Important persistent outputs:
- converted model metadata
- conversion logs or validation summaries
- runtime compatibility notes per supported checkpoint

---

## External interfaces / APIs

### Converter interface
Need a script or CLI entry point that converts a supported KugelAudio checkpoint to the ggml runtime format.

Expected inputs:
- model path / HF source
- output directory
- optional quantization settings
- optional feature flags for supported/unsupported paths

Expected outputs:
- converted model artifacts
- metadata describing supported runtime features

### Runtime interface
Need at least one developer-usable execution interface.

Required v1 option:
- CLI only

Required runtime inputs:
- text
- raw reference audio
- model path
- generation settings needed for the supported path

Required runtime outputs:
- generated waveform file
- explicit errors for unsupported combinations

Non-goal for v1:
- no required stable C or C++ embedding API

---

## Error handling / edge cases
- Unsupported KugelAudio checkpoints must fail clearly at both conversion time and load time.
- Missing semantic/acoustic submodules required by a supported path must fail clearly.
- Unsupported voice-conditioning mode must fail clearly.
- Reference audio shape/rate issues must be validated clearly.
- Reference audio should be resampled internally to 24 kHz mono and RMS-normalized to match canonical preprocessing.
- If a quantization mode is incompatible with a given model component, report it explicitly.
- If the initial v1 path cannot support a feature that exists in Python, the limitation must be documented, not silently ignored.

---

## Security / privacy considerations
- Raw voice prompts are sensitive user data.
- Do not log full reference-audio contents.
- Avoid verbose logging of full prompts by default.
- Conversion/runtime tooling should avoid leaking local file paths unnecessarily in shared logs.

---

## Observability / logging
Useful logs should include:
- checkpoint/config detected
- converter mode selected
- runtime feature support detected
- whether semantic conditioning is active
- whether raw voice prompting is active
- selected quantization mode
- evaluation configuration used for divergence testing

Avoid:
- dumping full prompt text by default
- dumping raw audio payloads

---

## Testing plan

### Acceptance checklist
- convert the supported `kugelaudio/kugelaudio-0-open` checkpoint successfully
- load the converted artifact in the runtime
- generate one reproducible single-speaker reference sample end-to-end
- generate one reproducible voice-cloned sample end-to-end using raw reference audio
- pass divergence testing against `../kugelaudio-open` on the same checkpoint, prompt, reference audio, seed, and generation settings
- for `f16`, closed-loop ASR recall must reach at least **95% of canonical PyTorch recall**, with an absolute floor of **0.80 recall**
- for `q8_0`, conversion, loading, and end-to-end generation must succeed on the same evaluation path

### Unit tests
- converter metadata parsing for the supported KugelAudio checkpoint/config shape
- tokenizer/prompt-building logic relevant to the ggml runtime
- prompt formatting parity for the canonical prompt sections
- feature gating logic for unsupported paths
- quantization option parsing / validation

### Integration tests
- convert the supported checkpoint successfully
- load the converted artifact in the runtime
- generate the reproducible reference and voice-cloned samples end-to-end

### Regression tests
- supported checkpoint still loads after converter/runtime changes
- voice cloning path still works after inference-loop changes
- closed-loop ASR regression stays within the defined threshold against canonical PyTorch
- quantization-related metadata remains accepted by the runtime

### Manual tests
- listen to the reproducible reference sample
- listen to the reproducible voice-cloned sample

---

## V2 / later scope

### Planned deferred features
- long-text chunking with parity to canonical KugelAudio behavior
- language hints
- multi-speaker dialog
- KV-cache quantization as a runtime option
- additional quantization-format validation and benchmarking

### Stretch goals
- support additional KugelAudio checkpoints beyond the first supported one
- support stronger quantization variants and benchmark them
- add broader hardware validation on edge-like systems
- explore whether parts of the pipeline can be pushed further into ggml with less wrapper involvement

### Deferred-but-promising ideas (`keep`)
- distilled / compressed KugelAudio variants if runtime-only work is insufficient
- broader edge runtime comparison beyond ggml once the port is working
- a bounded Vulkan feasibility spike later if there is a concrete reason
- deeper kernel/runtime specialization only after the port proves value

---

## Explicit keep / drop decisions

### Must keep
- voice cloning
- canonical prompt semantics
- semantic conditioning if required for canonical behavior
- a migration path for broader quantization
- a migration path for KV-cache quantization

### Drop
- pre-encoded voices
- pre-encoded `voice_cache`
- watermarking

### Defer
- language hints
- multi-speaker dialog
- long-text chunking
- advanced quantization validation beyond `f16` parity and `q8_0` end-to-end execution
- KV-cache quantization as a user-facing/runtime feature

---

## Open questions / assumptions
- How much of the existing `vibevoice.cpp` inference loop can be reused without semantic drift?
- What exact GGUF metadata schema should be preferred for long-term KugelAudio support after the migration period?
- Which additional quantization formats remain practical once the KugelAudio-specific pieces are ported?

---

## Practical v1 success definition
V1 is successful if the acceptance checklist is satisfied and:
- single-speaker raw-reference voice cloning works end-to-end
- the design clearly preserves room for broader quantization, KV-cache quantization, and long-text chunking in later iterations
- pre-encoded voices, watermarking, language hints, multi-speaker dialog, and long-text chunking are intentionally absent rather than half-implemented
