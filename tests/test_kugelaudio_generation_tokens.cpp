#include "vibevoice_tts.hpp"

#include <algorithm>
#include <cstdio>
#include <vector>

int main() {
    const auto valid = vv::detail::kugelaudio_valid_speech_token_ids_for_test();
    if (valid.size() != 4) {
        std::fprintf(stderr, "FAIL: expected 4 canonical valid speech-path tokens, got %zu\n", valid.size());
        return 1;
    }
    if (valid[0] != vv::detail::kugelaudio_speech_start_id_for_test() ||
        valid[1] != vv::detail::kugelaudio_speech_end_id_for_test() ||
        valid[2] != vv::detail::kugelaudio_speech_diffusion_id_for_test() ||
        valid[3] != vv::detail::kugelaudio_eos_id_for_test()) {
        std::fprintf(stderr, "FAIL: valid token set does not match canonical KugelAudio speech-path quartet\n");
        return 2;
    }

    std::vector<float> logits(151656, -1000.0f);
    logits[42] = 100.0f; // invalid token should never win once constrained
    logits[vv::detail::kugelaudio_speech_start_id_for_test()] = -3.0f;
    logits[vv::detail::kugelaudio_speech_end_id_for_test()] = 0.5f;
    logits[vv::detail::kugelaudio_speech_diffusion_id_for_test()] = 1.5f;
    logits[vv::detail::kugelaudio_eos_id_for_test()] = 0.25f;
    const int picked = vv::detail::select_kugelaudio_speech_token_from_logits_for_test(logits);
    if (picked != vv::detail::kugelaudio_speech_diffusion_id_for_test()) {
        std::fprintf(stderr, "FAIL: constrained selection picked %d, expected diffusion token %d\n",
                     picked, vv::detail::kugelaudio_speech_diffusion_id_for_test());
        return 3;
    }

    std::fill(logits.begin(), logits.end(), -1000.0f);
    logits[vv::detail::kugelaudio_eos_id_for_test()] = 2.0f;
    logits[vv::detail::kugelaudio_speech_end_id_for_test()] = 1.0f;
    const int picked_eos = vv::detail::select_kugelaudio_speech_token_from_logits_for_test(logits);
    if (picked_eos != vv::detail::kugelaudio_eos_id_for_test()) {
        std::fprintf(stderr, "FAIL: constrained selection picked %d, expected eos token %d\n",
                     picked_eos, vv::detail::kugelaudio_eos_id_for_test());
        return 4;
    }

    std::printf("KugelAudio generation token constraints OK\n");
    return 0;
}
