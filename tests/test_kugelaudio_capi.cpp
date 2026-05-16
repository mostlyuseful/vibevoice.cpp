#include "vibevoice_capi.h"
#include "vibevoice.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

namespace {
std::string g_last_log;

void capture_log(vv_log_level, const char* msg, void*) {
    g_last_log = msg ? msg : "";
}

bool file_ok(const char* p) {
    if (!p || !*p) return false;
    std::ifstream f(p, std::ios::binary);
    return f.good();
}
}  // namespace

int main() {
    const char* model = std::getenv("VIBEVOICE_KUGELAUDIO_MODEL");
    const char* tok = std::getenv("VIBEVOICE_KUGELAUDIO_PROMPT_TOKENIZER");
    if (!file_ok(model) || !file_ok(tok)) {
        std::fprintf(stderr, "skip: set VIBEVOICE_KUGELAUDIO_MODEL and VIBEVOICE_KUGELAUDIO_PROMPT_TOKENIZER\n");
        return 77;
    }

    vv_set_log_callback(capture_log, nullptr);

    int rc = vv_capi_load(model, nullptr, tok, nullptr, 0);
    if (rc != 0) {
        std::fprintf(stderr, "FAIL: vv_capi_load rc=%d\n", rc);
        return 1;
    }

    const char* refs[2] = {"a.wav", "b.wav"};
    g_last_log.clear();
    rc = vv_capi_tts("Hello world.", nullptr, refs, 2, "/tmp/kugelaudio_capi.wav", 20, 1.0f, 200, 123);
    if (rc != -2) {
        std::fprintf(stderr, "FAIL: vv_capi_tts multi-ref rc=%d want -2\n", rc);
        return 2;
    }
    if (g_last_log.find("unsupported KugelAudio runtime feature") == std::string::npos ||
        g_last_log.find("exactly one raw reference audio input") == std::string::npos) {
        std::fprintf(stderr, "FAIL: multi-ref capi log mismatch: %s\n", g_last_log.c_str());
        return 3;
    }

    g_last_log.clear();
    rc = vv_capi_tts("Hello world.", "/tmp/fake.voice.gguf", refs, 1, "/tmp/kugelaudio_capi.wav", 20, 1.0f, 200, 123);
    if (rc != -2) {
        std::fprintf(stderr, "FAIL: vv_capi_tts voice-path rc=%d want -2\n", rc);
        return 4;
    }
    if (g_last_log.find("unsupported KugelAudio runtime feature") == std::string::npos ||
        g_last_log.find("pre-baked voice gguf conditioning") == std::string::npos) {
        std::fprintf(stderr, "FAIL: pre-baked voice capi log mismatch: %s\n", g_last_log.c_str());
        return 5;
    }

    vv_capi_unload();
    vv_set_log_callback(nullptr, nullptr);
    std::printf("KugelAudio CAPI gating OK\n");
    return 0;
}
