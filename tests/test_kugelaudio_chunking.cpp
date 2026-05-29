#include "kugelaudio_chunking.hpp"

#include <cstdio>
#include <string>
#include <vector>

int main() {
    {
        vv::detail::KugelAudioChunkPlan plan;
        std::string error;
        if (!vv::detail::split_kugelaudio_text_into_chunks(
                "One two. Three four.", 2, 0, vv::ChunkingStrategy::Heuristic, &plan, &error)) {
            std::fprintf(stderr, "FAIL: sentence chunking rejected valid input: %s\n", error.c_str());
            return 1;
        }
        if (plan.chunks.size() != 2 ||
            plan.chunks[0] != "One two." ||
            plan.chunks[1] != "Three four.") {
            std::fprintf(stderr, "FAIL: sentence chunking shape mismatch\n");
            return 2;
        }
        if (plan.boundary_types.size() != 2 ||
            plan.boundary_types[0] != vv::detail::KugelAudioChunkBoundaryType::Sentence ||
            plan.boundary_types[1] != vv::detail::KugelAudioChunkBoundaryType::Sentence) {
            std::fprintf(stderr, "FAIL: sentence boundary typing mismatch\n");
            return 3;
        }
    }

    {
        vv::detail::KugelAudioChunkPlan plan;
        std::string error;
        if (!vv::detail::split_kugelaudio_text_into_chunks(
                "One two three, four five six.", 3, 0, vv::ChunkingStrategy::Heuristic, &plan, &error)) {
            std::fprintf(stderr, "FAIL: clause chunking rejected valid input: %s\n", error.c_str());
            return 4;
        }
        if (plan.chunks.size() != 2 ||
            plan.chunks[0] != "One two three," ||
            plan.chunks[1] != "four five six.") {
            std::fprintf(stderr, "FAIL: clause chunking shape mismatch\n");
            return 5;
        }
        if (plan.boundary_types.size() != 2 ||
            plan.boundary_types[0] != vv::detail::KugelAudioChunkBoundaryType::Clause ||
            plan.boundary_types[1] != vv::detail::KugelAudioChunkBoundaryType::Clause) {
            std::fprintf(stderr, "FAIL: clause boundary typing mismatch\n");
            return 6;
        }
    }

    {
        vv::detail::KugelAudioChunkPlan plan;
        std::string error;
        if (!vv::detail::split_kugelaudio_text_into_chunks(
                "one two three four five", 2, 0, vv::ChunkingStrategy::Heuristic, &plan, &error)) {
            std::fprintf(stderr, "FAIL: hard-wrap chunking rejected valid input: %s\n", error.c_str());
            return 7;
        }
        const std::vector<std::string> want = {"one two", "three four", "five"};
        if (plan.chunks != want) {
            std::fprintf(stderr, "FAIL: hard-wrap chunking mismatch\n");
            return 8;
        }
        for (auto t : plan.boundary_types) {
            if (t != vv::detail::KugelAudioChunkBoundaryType::HardWrap) {
                std::fprintf(stderr, "FAIL: hard-wrap boundary typing mismatch\n");
                return 9;
            }
        }
    }

    {
        vv::detail::KugelAudioChunkPlan plan;
        std::string error;
        if (!vv::detail::split_kugelaudio_text_into_chunks(
                "Alpha one. Beta two. Gamma three.", 4, 1, vv::ChunkingStrategy::Heuristic, &plan, &error)) {
            std::fprintf(stderr, "FAIL: overlap chunking rejected valid input: %s\n", error.c_str());
            return 10;
        }
        if (plan.chunks.size() != 2 ||
            plan.chunks[0] != "Alpha one. Beta two." ||
            plan.chunks[1] != "two. Gamma three.") {
            std::fprintf(stderr, "FAIL: overlap chunking mismatch\n");
            return 11;
        }
        if (plan.overlap_prefixes.size() != 2 ||
            plan.overlap_prefixes[0] != "" ||
            plan.overlap_prefixes[1] != "two.") {
            std::fprintf(stderr, "FAIL: overlap prefixes mismatch\n");
            return 12;
        }
    }

    {
        vv::detail::KugelAudioChunkPlan plan;
        std::string error;
        const std::string text =
            "One two three four five six seven eight. "
            "Nine ten eleven twelve thirteen fourteen. "
            "Fifteen sixteen seventeen eighteen.";
        if (!vv::detail::split_kugelaudio_text_into_chunks(
                text, 12, 1, vv::ChunkingStrategy::Heuristic, &plan, &error)) {
            std::fprintf(stderr, "FAIL: budgeted overlap chunking rejected valid input: %s\n", error.c_str());
            return 13;
        }
        for (size_t i = 0; i < plan.chunks.size(); ++i) {
            auto count_words = [](const std::string& s) {
                int n = 0;
                bool in_word = false;
                for (char c : s) {
                    const bool is_word = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
                    if (is_word && !in_word) ++n;
                    in_word = is_word;
                }
                return n;
            };
            if (count_words(plan.chunks[i]) > 12) {
                std::fprintf(stderr, "FAIL: budgeted overlap chunk %zu has %d words > 12: %s\n",
                             i, count_words(plan.chunks[i]), plan.chunks[i].c_str());
                return 14;
            }
        }
    }

    {
        vv::detail::KugelAudioChunkPlan plan;
        std::string error;
        if (vv::detail::split_kugelaudio_text_into_chunks(
                "Hello world.", 2, -1, vv::ChunkingStrategy::Heuristic, &plan, &error)) {
            std::fprintf(stderr, "FAIL: negative overlap_sentences should be rejected\n");
            return 13;
        }
    }

    {
        vv::detail::KugelAudioChunkPlan plan;
        std::string error;
        if (!vv::detail::split_kugelaudio_text_into_chunks(
                "One two and three four and five six.", 3, 0,
                vv::ChunkingStrategy::SyntaxAware, &plan, &error)) {
            std::fprintf(stderr, "FAIL: syntax-aware chunking rejected valid input: %s\n", error.c_str());
            return 14;
        }
        const std::vector<std::string> want = {"One two", "and three four", "and five six."};
        if (plan.new_texts != want) {
            std::fprintf(stderr, "FAIL: syntax-aware chunking mismatch\n");
            return 15;
        }
        for (auto t : plan.boundary_types) {
            if (t != vv::detail::KugelAudioChunkBoundaryType::Clause) {
                std::fprintf(stderr, "FAIL: syntax-aware boundary typing mismatch\n");
                return 16;
            }
        }
    }

    {
        const int punct_ms = vv::detail::kugelaudio_pause_duration_ms_for_boundary(
            "Hello.", "Next", vv::ChunkPauseMode::Punctuation);
        if (punct_ms != 260) {
            std::fprintf(stderr, "FAIL: punctuation pause = %d want 260\n", punct_ms);
            return 14;
        }
        const int speaker_ms = vv::detail::kugelaudio_pause_duration_ms_for_boundary(
            "Hello.", "Speaker 1: Next", vv::ChunkPauseMode::SpeakerAware);
        if (speaker_ms != 360) {
            std::fprintf(stderr, "FAIL: speaker-aware pause = %d want 360\n", speaker_ms);
            return 15;
        }
    }

    {
        std::vector<std::vector<float>> chunks = {{1.0f, 1.0f, 1.0f}, {2.0f, 2.0f}};
        std::vector<float> stitched;
        std::string error;
        if (!vv::detail::stitch_kugelaudio_audio_chunks(
                chunks, {"left", "right"}, vv::ChunkPauseMode::None,
                0, 24000, &stitched, &error)) {
            std::fprintf(stderr, "FAIL: stitch without pause/crossfade failed: %s\n", error.c_str());
            return 16;
        }
        const std::vector<float> want = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f};
        if (stitched != want) {
            std::fprintf(stderr, "FAIL: stitch without pause/crossfade mismatch\n");
            return 17;
        }
    }

    {
        std::vector<std::vector<float>> chunks = {{0.0f, 0.0f}, {1.0f, 1.0f}};
        std::vector<float> stitched;
        std::string error;
        if (!vv::detail::stitch_kugelaudio_audio_chunks(
                chunks, {"Left.", "Right"}, vv::ChunkPauseMode::Punctuation,
                0, 1000, &stitched, &error)) {
            std::fprintf(stderr, "FAIL: stitch with pause failed: %s\n", error.c_str());
            return 18;
        }
        if (stitched.size() != 264) {
            std::fprintf(stderr, "FAIL: stitch with pause size=%zu want 264\n", stitched.size());
            return 19;
        }
    }

    {
        std::vector<std::vector<float>> chunks = {{1.0f, 1.0f, 1.0f}, {2.0f, 2.0f, 2.0f}};
        std::vector<float> stitched;
        std::string error;
        if (!vv::detail::stitch_kugelaudio_audio_chunks(
                chunks, {"Left", "Right"}, vv::ChunkPauseMode::None,
                2, 1000, &stitched, &error)) {
            std::fprintf(stderr, "FAIL: stitch with crossfade failed: %s\n", error.c_str());
            return 20;
        }
        if (stitched.size() != 4) {
            std::fprintf(stderr, "FAIL: stitch with crossfade size=%zu want 4\n", stitched.size());
            return 21;
        }
        if (stitched[0] != 1.0f || stitched[3] != 2.0f) {
            std::fprintf(stderr, "FAIL: stitch with crossfade boundary samples mismatch\n");
            return 22;
        }
    }

    std::printf("KugelAudio chunking helpers OK\n");
    return 0;
}
