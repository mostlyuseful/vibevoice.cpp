#include "vibevoice_tts.hpp"
#include "vibevoice.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
std::string g_last_log;

void capture_log(vv_log_level, const char* msg, void*) {
    g_last_log = msg ? msg : "";
}

bool file_ok(const char* p) {
    if (!p || !*p) return false;
    FILE* f = std::fopen(p, "rb");
    if (!f) return false;
    std::fclose(f);
    return true;
}

}  // namespace

int main() {
    const char* ok_path = std::getenv("VIBEVOICE_KUGELAUDIO_MODEL");
    const char* bad_schema_path = std::getenv("VIBEVOICE_KUGELAUDIO_BAD_SCHEMA_MODEL");
    const char* bad_checkpoint_path = std::getenv("VIBEVOICE_KUGELAUDIO_BAD_CHECKPOINT_MODEL");
    const char* missing_semantic_path = std::getenv("VIBEVOICE_KUGELAUDIO_MISSING_SEMANTIC_MODEL");
    const char* missing_acoustic_path = std::getenv("VIBEVOICE_KUGELAUDIO_MISSING_ACOUSTIC_MODEL");
    const char* missing_metadata_path = std::getenv("VIBEVOICE_KUGELAUDIO_MISSING_METADATA_MODEL");
    if (!file_ok(ok_path) || !file_ok(bad_schema_path) || !file_ok(bad_checkpoint_path) ||
        !file_ok(missing_semantic_path) || !file_ok(missing_acoustic_path) ||
        !file_ok(missing_metadata_path)) {
        std::fprintf(stderr,
                     "skip: set VIBEVOICE_KUGELAUDIO_MODEL, "
                     "VIBEVOICE_KUGELAUDIO_BAD_SCHEMA_MODEL, "
                     "VIBEVOICE_KUGELAUDIO_BAD_CHECKPOINT_MODEL, "
                     "VIBEVOICE_KUGELAUDIO_MISSING_SEMANTIC_MODEL, "
                     "VIBEVOICE_KUGELAUDIO_MISSING_ACOUSTIC_MODEL and "
                     "VIBEVOICE_KUGELAUDIO_MISSING_METADATA_MODEL\n");
        return 77;
    }

    vv_set_log_callback(capture_log, nullptr);

    vv::VibeVoiceModel model;
    if (!vv::vibevoice_load(ok_path, &model)) {
        std::fprintf(stderr, "FAIL: vibevoice_load rejected supported KugelAudio fixture\n");
        return 1;
    }
    if (model.variant != "1.5b") {
        std::fprintf(stderr, "FAIL: variant=%s want 1.5b normalized runtime path\n", model.variant.c_str());
        return 2;
    }
    if (model.cfg.hidden != 3584 || model.cfg.n_layers_lm != 8 || model.cfg.n_layers_tlm != 20 ||
        model.cfg.n_heads != 28 || model.cfg.n_kv_heads != 4 || model.cfg.head_dim != 128 ||
        model.cfg.latent != 64 || model.cfg.sample_rate != 24000) {
        std::fprintf(stderr, "FAIL: resolved metadata mismatch\n");
        return 3;
    }
    if (!model.w.lm_tok_embd || !model.w.ac_fc1_w || !model.sc_fc1_w || !model.w.dh.cond_proj ||
        !model.w.at_dec.head.kernel || !model.at_enc.head.kernel || !model.st_enc.head.kernel ||
        !model.lm_head) {
        std::fprintf(stderr, "FAIL: required KugelAudio tensors not wired\n");
        return 4;
    }

    g_last_log.clear();
    vv::VibeVoiceModel bad_schema;
    if (vv::vibevoice_load(bad_schema_path, &bad_schema)) {
        std::fprintf(stderr, "FAIL: unsupported schema fixture loaded successfully\n");
        return 5;
    }
    if (g_last_log.find("unsupported KugelAudio schema/config") == std::string::npos) {
        std::fprintf(stderr, "FAIL: bad schema log did not identify schema/config error: %s\n", g_last_log.c_str());
        return 10;
    }

    g_last_log.clear();
    vv::VibeVoiceModel bad_checkpoint;
    if (vv::vibevoice_load(bad_checkpoint_path, &bad_checkpoint)) {
        std::fprintf(stderr, "FAIL: unsupported checkpoint fixture loaded successfully\n");
        return 6;
    }
    if (g_last_log.find("unsupported KugelAudio schema/config") == std::string::npos) {
        std::fprintf(stderr, "FAIL: bad checkpoint log did not identify schema/config error: %s\n", g_last_log.c_str());
        return 11;
    }

    vv::VibeVoiceModel missing_semantic;
    if (vv::vibevoice_load(missing_semantic_path, &missing_semantic)) {
        std::fprintf(stderr, "FAIL: missing semantic submodule fixture loaded successfully\n");
        return 7;
    }

    vv::VibeVoiceModel missing_acoustic;
    if (vv::vibevoice_load(missing_acoustic_path, &missing_acoustic)) {
        std::fprintf(stderr, "FAIL: missing acoustic submodule fixture loaded successfully\n");
        return 8;
    }

    g_last_log.clear();
    vv::VibeVoiceModel missing_metadata;
    if (vv::vibevoice_load(missing_metadata_path, &missing_metadata)) {
        std::fprintf(stderr, "FAIL: missing metadata fixture loaded successfully\n");
        return 9;
    }
    if (g_last_log.find("unsupported KugelAudio schema/config") == std::string::npos) {
        std::fprintf(stderr, "FAIL: missing metadata log did not identify schema/config error: %s\n", g_last_log.c_str());
        return 12;
    }

    vv_set_log_callback(nullptr, nullptr);

    std::printf("KugelAudio loader contract OK\n");
    return 0;
}
