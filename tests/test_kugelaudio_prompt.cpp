#include "vibevoice_tts.hpp"

#include <cstdio>
#include <string>
#include <vector>

int main() {
    const std::string prompt = vv::detail::build_kugelaudio_prompt_single_speaker_for_test(
        3, "Hello world."
    );

    const std::string expected =
        " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n"
        " Voice input:\n"
        " Speaker 0:<|vision_pad|><|vision_pad|><|vision_pad|>\n"
        " Text input:\n"
        " Speaker 0: Hello world.\n"
        " Speech output:\n"
        "<|vision_start|>";

    if (prompt != expected) {
        std::fprintf(stderr, "FAIL: prompt mismatch\nEXPECTED:\n%s\nGOT:\n%s\n",
                     expected.c_str(), prompt.c_str());
        return 1;
    }

    if (vv::detail::kugelaudio_speech_start_id_for_test() != 151652 ||
        vv::detail::kugelaudio_speech_end_id_for_test() != 151653 ||
        vv::detail::kugelaudio_speech_diffusion_id_for_test() != 151654 ||
        vv::detail::kugelaudio_image_pad_id_for_test() != 151655) {
        std::fprintf(stderr, "FAIL: KugelAudio special token IDs do not match the supported checkpoint contract\n");
        return 4;
    }

    const auto inserted = vv::detail::build_kugelaudio_inserted_speech_tokens_for_test(3);
    const std::vector<int32_t> expected_inserted = {151654, 151654, 151654, 151652};
    if (inserted != expected_inserted) {
        std::fprintf(stderr, "FAIL: inserted speech token semantics mismatch\n");
        return 5;
    }

    const std::string prompt_prefixed = vv::detail::build_kugelaudio_prompt_single_speaker_for_test(
        1, "Speaker 0: Already tagged"
    );
    if (prompt_prefixed.find(" Text input:\n Speaker 0: Already tagged\n") == std::string::npos) {
        std::fprintf(stderr, "FAIL: pre-tagged speaker text was not preserved\n");
        return 2;
    }
    if (prompt_prefixed.find("<|vision_start|><|vision_pad|><|vision_end|>") != std::string::npos) {
        std::fprintf(stderr, "FAIL: canonical KugelAudio voice section should not wrap placeholders in vision_start/vision_end\n");
        return 3;
    }

    std::printf("KugelAudio prompt builder OK\n");
    return 0;
}
