#ifndef KUGELAUDIO_CHUNKING_HPP
#define KUGELAUDIO_CHUNKING_HPP

#include <string>
#include <vector>

namespace vv {

enum class ChunkPauseMode {
    None,
    Punctuation,
    SpeakerAware,
};

enum class ChunkingStrategy {
    Heuristic,
    SyntaxAware,
};

enum class ChunkContinuityMode {
    None,
    TailReference,
    LatentPrefix,
    PromptInstruction,
    CleanTailReference,
    SingleSequence,
    SegmentedState,
};

enum class TextEndPaddingMode {
    None,
    Ellipsis,
};

const char* chunk_pause_mode_name(ChunkPauseMode mode);
const char* chunking_strategy_name(ChunkingStrategy strategy);
const char* chunk_continuity_mode_name(ChunkContinuityMode mode);

namespace detail {

enum class KugelAudioChunkBoundaryType {
    Sentence,
    Clause,
    HardWrap,
};

struct KugelAudioChunkPlan {
    std::vector<std::string>                 chunks;
    std::vector<KugelAudioChunkBoundaryType> boundary_types;
    std::vector<std::string>                 overlap_prefixes;
    std::vector<std::string>                 new_texts;
};

const char* kugelaudio_chunk_boundary_type_name(KugelAudioChunkBoundaryType type);

bool split_kugelaudio_text_into_chunks(const std::string& text,
                                       int max_words_per_chunk,
                                       int overlap_sentences,
                                       ChunkingStrategy strategy,
                                       KugelAudioChunkPlan* out,
                                       std::string* error);

int kugelaudio_pause_duration_ms_for_boundary(const std::string& left_text,
                                              const std::string& right_text,
                                              ChunkPauseMode pause_mode);

bool stitch_kugelaudio_audio_chunks(const std::vector<std::vector<float>>& audio_chunks,
                                    const std::vector<std::string>& chunk_texts,
                                    ChunkPauseMode pause_mode,
                                    int crossfade_ms,
                                    int sample_rate,
                                    std::vector<float>* out,
                                    std::string* error);

}  // namespace detail
}  // namespace vv

#endif  // KUGELAUDIO_CHUNKING_HPP
