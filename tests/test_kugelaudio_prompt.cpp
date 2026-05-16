#include "vibevoice_tts.hpp"

#include <cstdio>
#include <string>

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
