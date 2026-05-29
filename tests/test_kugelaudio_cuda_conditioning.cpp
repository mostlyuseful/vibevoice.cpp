#include "backend.hpp"
#include "ggml-backend.h"
#include "vibevoice.h"
#include "vibevoice_tts.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::string g_logs;

void capture_log(vv_log_level, const char* msg, void*) {
    if (!msg) return;
    g_logs += msg;
    g_logs += '\n';
}

bool file_ok(const char* p) {
    if (!p || !*p) return false;
    FILE* f = std::fopen(p, "rb");
    if (!f) return false;
    std::fclose(f);
    return true;
}

bool have_cuda_device() {
    ggml_backend_load_all();
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        const char* name = ggml_backend_dev_name(dev);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const char* reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;
        if ((name && std::strstr(name, "CUDA")) ||
            (reg_name && std::strstr(reg_name, "cuda"))) {
            return true;
        }
    }
    return false;
}

}  // namespace

int main() {
    if (!have_cuda_device()) {
        std::fprintf(stderr, "skip: no CUDA backend device registered\n");
        return 77;
    }

    const char* model_path = std::getenv("KUGELAUDIO_MODEL");
    const char* tok_path   = std::getenv("KUGELAUDIO_TOKENIZER");
    const char* ref_wav    = std::getenv("KUGELAUDIO_REF_WAV");
    if (!file_ok(model_path) || !file_ok(tok_path) || !file_ok(ref_wav)) {
        std::fprintf(stderr,
            "skip: CUDA conditioning test needs KUGELAUDIO_MODEL + KUGELAUDIO_TOKENIZER + KUGELAUDIO_REF_WAV\n");
        return 77;
    }

    vv_set_log_callback(capture_log, nullptr);
    setenv("KUGELAUDIO_BACKEND", "cuda", 1);
    setenv("KUGELAUDIO_TEST_STOP_AFTER_CONNECTORS", "1", 1);

    vv::VibeVoiceModel model;
    if (!vv::vibevoice_load(model_path, &model)) {
        std::fprintf(stderr, "FAIL: CUDA f16 load %s\n", model_path);
        return 1;
    }
    if (!model.loader.has_key("kugelaudio.architecture")) {
        std::fprintf(stderr, "FAIL: CUDA f16 model %s is not detected as KugelAudio\n", model_path);
        return 2;
    }
    if (!model.tokenizer.load_from_file(tok_path)) {
        std::fprintf(stderr, "FAIL: tokenizer load %s\n", tok_path);
        return 3;
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
    unsetenv("KUGELAUDIO_TEST_STOP_AFTER_CONNECTORS");
    if (rc != 0) {
        std::fprintf(stderr, "FAIL: CUDA conditioning generate rc=%d\n", rc);
        return 4;
    }
    if (std::string(vv::backend_name()).find("CUDA") == std::string::npos) {
        std::fprintf(stderr, "FAIL: active backend is %s, expected CUDA\n", vv::backend_name());
        return 5;
    }
    if (samples.size() != 240) {
        std::fprintf(stderr, "FAIL: CUDA conditioning test hook produced %zu samples, expected 240\n", samples.size());
        return 6;
    }
    if (g_logs.find("test hook stopping after conditioning connectors") == std::string::npos) {
        std::fprintf(stderr, "FAIL: conditioning connector hook log not observed\n%s\n", g_logs.c_str());
        return 7;
    }

    std::printf("kugelaudio_cuda_conditioning: backend=%s samples=%zu\n",
                vv::backend_name(), samples.size());
    return 0;
}
