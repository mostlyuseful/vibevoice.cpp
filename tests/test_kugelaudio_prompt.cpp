#include "vibevoice_tts.hpp"
#include "model_loader.hpp"

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
}

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
    const auto neg_seed = vv::detail::build_kugelaudio_negative_seed_tokens_for_test();
    const std::vector<int32_t> expected_neg_seed = {151652};
    if (neg_seed != expected_neg_seed) {
        std::fprintf(stderr, "FAIL: KugelAudio CFG negative seed should be a single speech_start token\n");
        return 6;
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

    const char* tok_path = std::getenv("VIBEVOICE_KUGELAUDIO_PROMPT_TOKENIZER");
    if (!file_ok(tok_path)) {
        std::fprintf(stderr, "skip: set VIBEVOICE_KUGELAUDIO_PROMPT_TOKENIZER\n");
        return 77;
    }

    vv::ModelLoader loader;
    if (!loader.load(tok_path)) {
        std::fprintf(stderr, "FAIL: failed to load tokenizer fixture %s\n", tok_path);
        return 7;
    }
    vv::Tokenizer tok;
    if (!tok.load(loader)) {
        std::fprintf(stderr, "FAIL: tokenizer load failed for %s\n", tok_path);
        return 8;
    }

    std::vector<int32_t> input_ids;
    std::vector<int> pad_positions;
    vv::detail::build_kugelaudio_prompt_input_ids_for_test(tok, 3, "Hello world.", &input_ids, &pad_positions);
    const std::string system_prompt =
        " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n";
    const std::string voice_header = " Voice input:\n";
    const std::string speaker_prefix = " Speaker 0:";
    const int expected_first_pad = static_cast<int>(system_prompt.size() + voice_header.size() + speaker_prefix.size());
    const std::vector<int> expected_pad_positions = {
        expected_first_pad,
        expected_first_pad + 1,
        expected_first_pad + 2,
    };
    if (pad_positions != expected_pad_positions) {
        std::fprintf(stderr, "FAIL: pad positions mismatch\n");
        return 9;
    }
    for (int pos : pad_positions) {
        if (input_ids[pos] != vv::detail::kugelaudio_speech_diffusion_id_for_test()) {
            std::fprintf(stderr, "FAIL: pad position %d does not contain diffusion placeholder id\n", pos);
            return 10;
        }
    }
    if (input_ids.back() != vv::detail::kugelaudio_speech_start_id_for_test()) {
        std::fprintf(stderr, "FAIL: final prompt token is not speech_start\n");
        return 11;
    }

    std::printf("KugelAudio prompt builder OK\n");
    return 0;
}
