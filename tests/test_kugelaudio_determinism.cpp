#include "backend.hpp"
#include "vibevoice_tts.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

bool file_ok(const char* p) {
    if (!p || !*p) return false;
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

}  // namespace

int main() {
    const char* model_path = std::getenv("KUGELAUDIO_MODEL");
    const char* tok_path   = std::getenv("KUGELAUDIO_TOKENIZER");
    const char* ref_wav    = std::getenv("KUGELAUDIO_REF_WAV");
    if (!file_ok(model_path) || !file_ok(tok_path) || !file_ok(ref_wav)) {
        std::fprintf(stderr,
                     "skip: set KUGELAUDIO_MODEL, KUGELAUDIO_TOKENIZER, and KUGELAUDIO_REF_WAV\n");
        return 77;
    }

    setenv("KUGELAUDIO_BACKEND", "cpu", 1);

    vv::VibeVoiceModel model;
    if (!vv::vibevoice_load(model_path, &model)) {
        std::fprintf(stderr, "FAIL: load %s\n", model_path);
        return 1;
    }
    if (!model.loader.has_key("kugelaudio.architecture")) {
        std::fprintf(stderr, "FAIL: model %s is not detected as KugelAudio\n", model_path);
        return 2;
    }
    if (std::string(vv::backend_name()) != "CPU") {
        std::fprintf(stderr, "skip: determinism check requires CPU backend, got %s\n", vv::backend_name());
        return 77;
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

    std::vector<float> a, b;
    int rc = vv::vibevoice_tts_generate(&model, "Hello world.", p, &a);
    if (rc != 0) {
        std::fprintf(stderr, "FAIL: first generate rc=%d\n", rc);
        return 4;
    }
    rc = vv::vibevoice_tts_generate(&model, "Hello world.", p, &b);
    if (rc != 0) {
        std::fprintf(stderr, "FAIL: second generate rc=%d\n", rc);
        return 5;
    }

    if (a != b) {
        std::fprintf(stderr,
                     "FAIL: seeded CPU runs diverged (sizes %zu vs %zu)\n",
                     a.size(), b.size());
        return 6;
    }
    if (a.empty()) {
        std::fprintf(stderr, "FAIL: deterministic run produced empty output\n");
        return 7;
    }

    std::printf("KugelAudio CPU determinism OK: %zu samples identical across repeated seeded runs\n",
                a.size());
    return 0;
}
