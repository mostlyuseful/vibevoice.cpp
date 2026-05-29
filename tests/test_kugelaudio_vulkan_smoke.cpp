#include "backend.hpp"
#include "ggml-backend.h"
#include "vibevoice_tts.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

bool have_vulkan_device() {
    ggml_backend_load_all();
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        const char* name = ggml_backend_dev_name(dev);
        if (name && std::strstr(name, "Vulkan")) return true;
    }
    return false;
}

}  // namespace

int main() {
    if (!have_vulkan_device()) {
        std::fprintf(stderr, "skip: no Vulkan backend device registered\n");
        return 77;
    }

    const char* model_path = std::getenv("KUGELAUDIO_Q8_MODEL");
    const char* tok_path   = std::getenv("KUGELAUDIO_TOKENIZER");
    const char* ref_wav    = std::getenv("KUGELAUDIO_REF_WAV");
    if (!file_ok(model_path) || !file_ok(tok_path) || !file_ok(ref_wav)) {
        std::fprintf(stderr,
            "skip: Vulkan smoke test needs KUGELAUDIO_Q8_MODEL + KUGELAUDIO_TOKENIZER + KUGELAUDIO_REF_WAV\n");
        return 77;
    }

    setenv("KUGELAUDIO_BACKEND", "vulkan", 1);

    vv::VibeVoiceModel model;
    if (!vv::vibevoice_load(model_path, &model)) {
        std::fprintf(stderr, "FAIL: Vulkan q8_0 load %s\n", model_path);
        return 1;
    }
    if (!model.loader.has_key("kugelaudio.architecture")) {
        std::fprintf(stderr, "FAIL: Vulkan q8_0 model %s is not detected as KugelAudio\n", model_path);
        return 2;
    }
    if (std::string(vv::backend_name()).find("Vulkan") == std::string::npos) {
        std::fprintf(stderr, "FAIL: active backend is %s, expected Vulkan\n", vv::backend_name());
        return 3;
    }
    if (!model.tokenizer.load_from_file(tok_path)) {
        std::fprintf(stderr, "FAIL: tokenizer load %s\n", tok_path);
        return 4;
    }

    vv::VibeVoiceTTSParams p;
    p.ref_audio_paths   = {ref_wav};
    p.max_speech_frames = 24;
    p.n_diffusion_steps = 4;
    p.cfg_scale         = 1.0f;
    p.seed              = 12345;
    p.verbose           = false;

    std::vector<float> samples;
    const int rc = vv::vibevoice_tts_generate(&model, "Hello world.", p, &samples);
    if (rc != 0) {
        std::fprintf(stderr, "FAIL: Vulkan q8_0 generate rc=%d\n", rc);
        return 5;
    }
    if (samples.size() < 512) {
        std::fprintf(stderr, "FAIL: Vulkan q8_0 produced too few samples (%zu)\n", samples.size());
        return 6;
    }

    double sq = 0.0;
    bool finite = true;
    for (float v : samples) {
        finite = finite && std::isfinite(v);
        sq += static_cast<double>(v) * v;
    }
    const double rms = std::sqrt(sq / samples.size());
    std::printf("kugelaudio_vulkan_smoke: backend=%s samples=%zu rms=%.6f\n",
                vv::backend_name(), samples.size(), rms);
    if (!finite) {
        std::fprintf(stderr, "FAIL: Vulkan q8_0 produced non-finite samples\n");
        return 7;
    }
    if (rms < 1e-5) {
        std::fprintf(stderr, "FAIL: Vulkan q8_0 output is effectively silent (rms=%.8f)\n", rms);
        return 8;
    }
    return 0;
}
