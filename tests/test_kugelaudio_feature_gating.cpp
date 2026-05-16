#include "vibevoice_tts.hpp"

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
    const char* ok_path = std::getenv("VIBEVOICE_KUGELAUDIO_MODEL");
    if (!file_ok(ok_path)) {
        std::fprintf(stderr, "skip: set VIBEVOICE_KUGELAUDIO_MODEL\n");
        return 77;
    }

    vv::VibeVoiceModel model;
    if (!vv::vibevoice_load(ok_path, &model)) {
        std::fprintf(stderr, "FAIL: vibevoice_load rejected supported KugelAudio fixture\n");
        return 1;
    }

    std::vector<float> samples;

    vv::VibeVoiceTTSParams multi_ref;
    multi_ref.ref_audio_paths = {"a.wav", "b.wav"};
    int rc = vv::vibevoice_tts_generate(&model, "Hello world.", multi_ref, &samples);
    if (rc != -21) {
        std::fprintf(stderr, "FAIL: multi-ref gating rc=%d want -21\n", rc);
        return 2;
    }

    vv::VibeVoiceTTSParams speaker_tagged;
    speaker_tagged.ref_audio_paths = {"a.wav"};
    rc = vv::vibevoice_tts_generate(&model, "Speaker 0: Hello world.", speaker_tagged, &samples);
    if (rc != -22) {
        std::fprintf(stderr, "FAIL: speaker-tagged gating rc=%d want -22\n", rc);
        return 3;
    }

    vv::VibeVoiceVoice voice;
    vv::VibeVoiceTTSParams pre_baked_voice;
    pre_baked_voice.voice = &voice;
    pre_baked_voice.ref_audio_paths = {"a.wav"};
    rc = vv::vibevoice_tts_generate(&model, "Hello world.", pre_baked_voice, &samples);
    if (rc != -20) {
        std::fprintf(stderr, "FAIL: pre-baked-voice gating rc=%d want -20\n", rc);
        return 4;
    }

    vv::VibeVoiceTTSParams supported_shape;
    supported_shape.ref_audio_paths = {"a.wav"};
    rc = vv::vibevoice_tts_generate(&model, "Hello world.", supported_shape, &samples);
    if (rc != -2) {
        std::fprintf(stderr, "FAIL: supported single-ref shape should fall through to tokenizer-not-loaded rc=-2, got %d\n", rc);
        return 5;
    }

    std::printf("KugelAudio feature gating OK\n");
    return 0;
}
