// q8_0 end-to-end smoke test for KugelAudio v1 acceptance.
//
// Validates that a q8_0 quantized KugelAudio model loads and produces
// non-empty, finite, non-silent output on the same acceptance path as f16.
//
// Skips with rc=77 unless:
//   VIBEVOICE_KUGELAUDIO_Q8_MODEL -> q8_0 gguf
//   VIBEVOICE_TOKENIZER            -> tokenizer gguf
//   VIBEVOICE_REF_WAV              -> 24 kHz mono reference wav

#include "vibevoice_tts.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

bool file_ok(const char* p) {
    if (!p || !*p) return false;
    FILE* f = std::fopen(p, "rb");
    if (!f) return false;
    std::fclose(f);
    return true;
}

}  // namespace

int main() {
    const char* model_path = std::getenv("VIBEVOICE_KUGELAUDIO_Q8_MODEL");
    const char* tok_path   = std::getenv("VIBEVOICE_TOKENIZER");
    const char* ref_wav    = std::getenv("VIBEVOICE_REF_WAV");
    if (!file_ok(model_path) || !file_ok(tok_path) || !file_ok(ref_wav)) {
        std::fprintf(stderr,
            "skip: KugelAudio q8_0 smoke test needs VIBEVOICE_KUGELAUDIO_Q8_MODEL + VIBEVOICE_TOKENIZER + VIBEVOICE_REF_WAV\n");
        return 77;
    }

    vv::VibeVoiceModel model;
    if (!vv::vibevoice_load(model_path, &model)) {
        std::fprintf(stderr, "FAIL: q8_0 load %s\n", model_path);
        return 1;
    }
    if (!model.loader.has_key("kugelaudio.architecture")) {
        std::fprintf(stderr, "FAIL: q8_0 model %s is not detected as KugelAudio\n", model_path);
        return 2;
    }
    if (!model.tokenizer.load_from_file(tok_path)) {
        std::fprintf(stderr, "FAIL: tokenizer load %s\n", tok_path);
        return 3;
    }

    vv::VibeVoiceTTSParams p;
    p.ref_audio_paths   = {ref_wav};
    p.max_speech_frames = 32;
    p.n_diffusion_steps = 8;
    p.cfg_scale         = 1.0f;
    p.seed              = 12345;
    p.verbose           = false;

    std::vector<float> samples;
    const int rc = vv::vibevoice_tts_generate(&model, "Hello world.", p, &samples);
    if (rc != 0) {
        std::fprintf(stderr, "FAIL: q8_0 generate rc=%d\n", rc);
        return 4;
    }
    if (samples.empty()) {
        std::fprintf(stderr, "FAIL: q8_0 produced no samples\n");
        return 5;
    }
    if (samples.size() < 512) {
        std::fprintf(stderr, "FAIL: q8_0 produced too few samples (%zu)\n", samples.size());
        return 6;
    }

    double sq = 0.0;
    bool finite = true;
    for (float v : samples) {
        finite = finite && std::isfinite(v);
        sq += static_cast<double>(v) * v;
    }
    const double rms = std::sqrt(sq / samples.size());
    std::printf("kugelaudio_q8_0_smoke: %zu samples rms=%.6f\n", samples.size(), rms);

    if (!finite) {
        std::fprintf(stderr, "FAIL: q8_0 produced non-finite samples\n");
        return 7;
    }
    if (rms < 1e-5) {
        std::fprintf(stderr, "FAIL: q8_0 output is effectively silent (rms=%.8f)\n", rms);
        return 8;
    }
    std::printf("q8_0 end-to-end OK\n");
    return 0;
}
