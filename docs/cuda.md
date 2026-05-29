# CUDA / GPU build

`kugelaudio.cpp` builds against the embedded `ggml` submodule's CUDA
backend — no source changes needed, just a CMake flag.

> **Status:** the runtime now uses ggml's `ggml_backend_*` API rather than the
> old CPU-only graph shortcut. CUDA/Vulkan support is still correctness-first,
> but backend selection, fallback logging, and eval-harness integration are now
> part of the supported path.

## Build

```bash
cmake -B build \
    -DKUGELAUDIO_BUILD_TESTS=ON \
    -DKUGELAUDIO_BUILD_EXAMPLES=ON \
    -DKUGELAUDIO_GGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Forwarded ggml backend flags (any combination):

| CMake flag                     | What it enables                               |
| ------------------------------ | --------------------------------------------- |
| `-DKUGELAUDIO_GGML_CUDA=ON`     | NVIDIA CUDA (`ggml-cuda`)                     |
| `-DKUGELAUDIO_GGML_METAL=ON`    | Apple Metal (`ggml-metal`)                    |
| `-DKUGELAUDIO_GGML_VULKAN=ON`   | cross-vendor Vulkan compute (`ggml-vulkan`)   |
| `-DKUGELAUDIO_GGML_HIPBLAS=ON`  | AMD ROCm (`ggml-hipblas`)                     |

## Smoke test

```bash
# TTS — use the raw-reference path rather than the removed voice.gguf flow.
KUGELAUDIO_BACKEND=cuda ./build/bin/kugelaudio-cli \
    --model     models/kugelaudio-q8_0.gguf \
    --tokenizer models/tokenizer.gguf \
    --ref-audio tests/fixtures/reference_sine.wav \
    --text "Hello from CUDA." \
    --out hello.wav
```

For the CUDA-specific C++ smoke coverage:

```bash
ctest --test-dir build -R 'test_kugelaudio_cuda_smoke|test_kugelaudio_cuda_conditioning' --output-on-failure
```

## Notes

- ggml dispatches per-op to the available backend; ops that aren't
  implemented on the chosen backend fall back to CPU automatically.
- The `vv_capi_load(... n_threads)` argument is a CPU-thread count; on
  GPU the kernel grids do the parallelism. Pass any reasonable value
  (4 is fine).
- You can request a specific backend with `KUGELAUDIO_BACKEND=cuda|vulkan|cpu`.
- On multi-device systems, use `KUGELAUDIO_BACKEND_DEVICE_INDEX=N` to select a
  specific matching device.
- Set `KUGELAUDIO_BACKEND_VERBOSE=1` to include device descriptions/memory in
  startup logs.
- `CUDA_VISIBLE_DEVICES=0` (or similar) still works if you want to constrain
  CUDA visibility before launching `kugelaudio-cli`.
- The remaining published smoke path is KugelAudio raw-reference TTS; any
  ASR-specific compatibility checks should be treated as internal/legacy.
- `test_kugelaudio_cuda_smoke` uses `KUGELAUDIO_Q8_MODEL`,
  `KUGELAUDIO_TOKENIZER`, and `KUGELAUDIO_REF_WAV`.
- `test_kugelaudio_cuda_conditioning` uses `KUGELAUDIO_MODEL`,
  `KUGELAUDIO_TOKENIZER`, and `KUGELAUDIO_REF_WAV`, then stops after the
  conditioning connector stage to validate the encoder/connector graph on CUDA.
