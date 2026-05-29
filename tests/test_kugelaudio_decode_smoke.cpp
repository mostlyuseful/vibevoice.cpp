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
    const char* model_path = std::getenv("KUGELAUDIO_MODEL");
    const char* tok_path   = std::getenv("KUGELAUDIO_TOKENIZER");
    const char* ref_wav    = std::getenv("KUGELAUDIO_REF_WAV");
    if (!file_ok(model_path) || !file_ok(tok_path) || !file_ok(ref_wav)) {
        std::fprintf(stderr,
            "skip: KugelAudio decode smoke test needs KUGELAUDIO_MODEL + KUGELAUDIO_TOKENIZER + KUGELAUDIO_REF_WAV\n");
        return 77;
    }

    vv::VibeVoiceModel model;
    if (!vv::vibevoice_load(model_path, &model)) {
        std::fprintf(stderr, "FAIL: load %s\n", model_path);
        return 1;
    }
    if (!model.loader.has_key("kugelaudio.architecture")) {
        std::fprintf(stderr, "FAIL: model %s is not detected as KugelAudio\n", model_path);
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
        std::fprintf(stderr, "FAIL: KugelAudio generate rc=%d\n", rc);
        return 4;
    }
    if (samples.empty()) {
        std::fprintf(stderr, "FAIL: KugelAudio decode produced no samples\n");
        return 5;
    }
    if (samples.size() < 512) {
        std::fprintf(stderr, "FAIL: KugelAudio decode produced too few samples (%zu)\n", samples.size());
        return 6;
    }

    double sq = 0.0;
    double mn = 1e30;
    double mx = -1e30;
    bool finite = true;
    for (float v : samples) {
        finite = finite && std::isfinite(v);
        sq += static_cast<double>(v) * v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    const double rms = std::sqrt(sq / samples.size());
    std::printf("kugelaudio_decode_smoke: %zu samples rms=%.6f range=[%.4f, %.4f]\n",
                samples.size(), rms, mn, mx);

    if (!finite) {
        std::fprintf(stderr, "FAIL: KugelAudio decode produced non-finite samples\n");
        return 7;
    }
    if (rms < 1e-5) {
        std::fprintf(stderr, "FAIL: KugelAudio decode output is effectively silent (rms=%.8f)\n", rms);
        return 8;
    }
    if (mn < -1.5 || mx > 1.5) {
        std::fprintf(stderr, "FAIL: KugelAudio decode output range looks implausible [%.6f, %.6f]\n", mn, mx);
        return 9;
    }
    return 0;
}
