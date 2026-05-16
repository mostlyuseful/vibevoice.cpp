#include "vibevoice_tts.hpp"
#include "vibevoice.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

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
    if (!file_ok(ok_path)) {
        std::fprintf(stderr, "skip: set VIBEVOICE_KUGELAUDIO_MODEL\n");
        return 77;
    }

    vv_set_log_callback(capture_log, nullptr);

    vv::VibeVoiceModel model;
    if (!vv::vibevoice_load(ok_path, &model)) {
        std::fprintf(stderr, "FAIL: vibevoice_load rejected supported KugelAudio fixture\n");
        return 1;
    }

    std::vector<float> samples;

    vv::VibeVoiceTTSParams accepted;
    accepted.ref_audio_paths = {"a.wav"};
    std::string gate_error;
    if (!vv::detail::validate_kugelaudio_single_speaker_request("Hello world.", accepted, &gate_error)) {
        std::fprintf(stderr, "FAIL: accepted KugelAudio single-reference shape was rejected: %s\n", gate_error.c_str());
        return 2;
    }

    vv::VibeVoiceTTSParams multi_ref;
    multi_ref.ref_audio_paths = {"a.wav", "b.wav"};
    g_last_log.clear();
    int rc = vv::vibevoice_tts_generate(&model, "Hello world.", multi_ref, &samples);
    if (rc != -21) {
        std::fprintf(stderr, "FAIL: multi-ref gating rc=%d want -21\n", rc);
        return 3;
    }
    if (g_last_log.find("unsupported KugelAudio runtime feature") == std::string::npos ||
        g_last_log.find("KugelAudio v1 supports only single-speaker TTS") == std::string::npos) {
        std::fprintf(stderr, "FAIL: multi-ref log did not explain the supported v1 shape: %s\n", g_last_log.c_str());
        return 7;
    }

    vv::VibeVoiceTTSParams speaker_tagged;
    speaker_tagged.ref_audio_paths = {"a.wav"};
    g_last_log.clear();
    rc = vv::vibevoice_tts_generate(&model, "Speaker 0: Hello world.", speaker_tagged, &samples);
    if (rc != -22) {
        std::fprintf(stderr, "FAIL: speaker-tagged gating rc=%d want -22\n", rc);
        return 4;
    }
    if (g_last_log.find("unsupported KugelAudio runtime feature") == std::string::npos ||
        g_last_log.find("plain untagged text") == std::string::npos ||
        g_last_log.find("KugelAudio v1 supports only single-speaker TTS") == std::string::npos) {
        std::fprintf(stderr, "FAIL: speaker-tagged log did not explain the v1 limitation: %s\n", g_last_log.c_str());
        return 8;
    }

    vv::VibeVoiceVoice voice;
    vv::VibeVoiceTTSParams pre_baked_voice;
    pre_baked_voice.voice = &voice;
    pre_baked_voice.ref_audio_paths = {"a.wav"};
    g_last_log.clear();
    rc = vv::vibevoice_tts_generate(&model, "Hello world.", pre_baked_voice, &samples);
    if (rc != -20) {
        std::fprintf(stderr, "FAIL: pre-baked-voice gating rc=%d want -20\n", rc);
        return 5;
    }
    if (g_last_log.find("unsupported KugelAudio runtime feature") == std::string::npos ||
        g_last_log.find("pre-baked voice gguf conditioning") == std::string::npos ||
        g_last_log.find("KugelAudio v1 supports only single-speaker TTS") == std::string::npos) {
        std::fprintf(stderr, "FAIL: pre-baked-voice log did not explain the v1 limitation: %s\n", g_last_log.c_str());
        return 9;
    }

    vv::VibeVoiceTTSParams supported_shape;
    supported_shape.ref_audio_paths = {"a.wav"};
    rc = vv::vibevoice_tts_generate(&model, "Hello world.", supported_shape, &samples);
    if (rc != -2) {
        std::fprintf(stderr, "FAIL: supported single-ref shape should fall through to tokenizer-not-loaded rc=-2, got %d\n", rc);
        return 6;
    }

    vv_set_log_callback(nullptr, nullptr);

    std::printf("KugelAudio feature gating OK\n");
    return 0;
}
