#include "kugelaudio_chunking.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <regex>
#include <string>
#include <vector>

namespace vv {

const char* chunk_pause_mode_name(ChunkPauseMode mode) {
    switch (mode) {
        case ChunkPauseMode::None:         return "none";
        case ChunkPauseMode::Punctuation:  return "punctuation";
        case ChunkPauseMode::SpeakerAware: return "speaker-aware";
    }
    return "unknown";
}

const char* chunking_strategy_name(ChunkingStrategy strategy) {
    switch (strategy) {
        case ChunkingStrategy::Heuristic:   return "heuristic";
        case ChunkingStrategy::SyntaxAware: return "syntax-aware";
    }
    return "unknown";
}

const char* chunk_continuity_mode_name(ChunkContinuityMode mode) {
    switch (mode) {
        case ChunkContinuityMode::None:               return "none";
        case ChunkContinuityMode::TailReference:      return "tail-reference";
        case ChunkContinuityMode::LatentPrefix:       return "latent-prefix";
        case ChunkContinuityMode::PromptInstruction:  return "prompt-instruction";
        case ChunkContinuityMode::CleanTailReference: return "clean-tail-reference";
        case ChunkContinuityMode::SingleSequence:     return "single-sequence";
        case ChunkContinuityMode::SegmentedState:     return "segmented-state";
    }
    return "unknown";
}

namespace detail {
namespace {

std::string trim_copy(const std::string& s) {
    size_t lo = 0;
    while (lo < s.size() && std::isspace(static_cast<unsigned char>(s[lo]))) ++lo;
    size_t hi = s.size();
    while (hi > lo && std::isspace(static_cast<unsigned char>(s[hi - 1]))) --hi;
    return s.substr(lo, hi - lo);
}

std::vector<std::string> split_words(const std::string& text) {
    std::vector<std::string> words;
    size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
        const size_t start = i;
        while (i < text.size() && !std::isspace(static_cast<unsigned char>(text[i]))) ++i;
        if (i > start) words.push_back(text.substr(start, i - start));
    }
    return words;
}

int word_count(const std::string& text) {
    return static_cast<int>(split_words(text).size());
}

std::string join_words(const std::vector<std::string>& words, size_t begin, size_t end) {
    std::string out;
    for (size_t i = begin; i < end; ++i) {
        if (!out.empty()) out += ' ';
        out += words[i];
    }
    return out;
}

std::vector<std::string> hard_wrap_words(const std::string& text, int max_words) {
    const std::vector<std::string> words = split_words(text);
    std::vector<std::string> out;
    if (max_words <= 0) return out;
    for (size_t i = 0; i < words.size(); i += static_cast<size_t>(max_words)) {
        out.push_back(join_words(words, i, std::min(words.size(), i + static_cast<size_t>(max_words))));
    }
    return out;
}

bool is_sentence_terminal(char c) {
    return c == '.' || c == '!' || c == '?';
}

std::vector<std::string> split_sentences_heuristic(const std::string& text) {
    std::vector<std::string> out;
    const std::string normalized = trim_copy(text);
    if (normalized.empty()) return out;

    size_t start = 0;
    size_t candidate_end = std::string::npos;
    for (size_t i = 0; i < normalized.size(); ++i) {
        if (is_sentence_terminal(normalized[i])) {
            candidate_end = i + 1;
            continue;
        }
        if (candidate_end != std::string::npos && std::isspace(static_cast<unsigned char>(normalized[i]))) {
            const std::string sentence = trim_copy(normalized.substr(start, candidate_end - start));
            if (!sentence.empty()) out.push_back(sentence);
            while (i < normalized.size() && std::isspace(static_cast<unsigned char>(normalized[i]))) ++i;
            start = i;
            if (i >= normalized.size()) break;
            --i;
            candidate_end = std::string::npos;
            continue;
        }
        if (!std::isspace(static_cast<unsigned char>(normalized[i]))) {
            candidate_end = std::string::npos;
        }
    }

    if (start < normalized.size()) {
        const std::string tail = trim_copy(normalized.substr(start));
        if (!tail.empty()) out.push_back(tail);
    }
    if (out.empty()) out.push_back(normalized);
    return out;
}

bool is_dash_like(char c) {
    return c == '-' || static_cast<unsigned char>(c) == 0x97 || static_cast<unsigned char>(c) == 0x96;
}

bool is_syntax_aware_conjunction(const std::string& word) {
    static const char* const kWords[] = {
        "and", "but", "or", "because", "which", "that",
        "while", "although", "however", "then"
    };
    for (const char* w : kWords) {
        if (word == w) return true;
    }
    return false;
}

std::vector<std::string> split_syntax_aware_phrase_parts(const std::string& text) {
    std::vector<std::string> out;
    const std::string normalized = trim_copy(text);
    if (normalized.empty()) return out;

    size_t start = 0;
    for (size_t i = 0; i < normalized.size(); ++i) {
        const char c = normalized[i];
        if ((c == ',' || c == ';' || c == ':') && i + 1 < normalized.size() &&
            std::isspace(static_cast<unsigned char>(normalized[i + 1]))) {
            const std::string piece = trim_copy(normalized.substr(start, i + 1 - start));
            if (!piece.empty()) out.push_back(piece);
            size_t next = i + 1;
            while (next < normalized.size() && std::isspace(static_cast<unsigned char>(normalized[next]))) ++next;
            start = next;
            i = next ? next - 1 : next;
            continue;
        }
        if (std::isspace(static_cast<unsigned char>(c))) {
            size_t j = i;
            while (j < normalized.size() && std::isspace(static_cast<unsigned char>(normalized[j]))) ++j;
            if (j < normalized.size() && is_dash_like(normalized[j])) {
                size_t k = j + 1;
                while (k < normalized.size() && std::isspace(static_cast<unsigned char>(normalized[k]))) ++k;
                if (k > j + 1) {
                    const std::string piece = trim_copy(normalized.substr(start, i - start));
                    if (!piece.empty()) out.push_back(piece);
                    start = k;
                    i = k ? k - 1 : k;
                    continue;
                }
            }
            if (j < normalized.size()) {
                size_t word_end = j;
                while (word_end < normalized.size() && std::isalpha(static_cast<unsigned char>(normalized[word_end]))) ++word_end;
                if (word_end > j) {
                    std::string word = normalized.substr(j, word_end - j);
                    for (char& ch : word) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
                    if (is_syntax_aware_conjunction(word)) {
                        const std::string piece = trim_copy(normalized.substr(start, i - start));
                        if (!piece.empty()) out.push_back(piece);
                        start = j;
                        i = j ? j - 1 : j;
                    }
                }
            }
        }
    }

    if (start < normalized.size()) {
        const std::string tail = trim_copy(normalized.substr(start));
        if (!tail.empty()) out.push_back(tail);
    }
    return out;
}

std::vector<std::string> split_clause_parts(const std::string& text) {
    std::vector<std::string> out;
    const std::string normalized = trim_copy(text);
    if (normalized.empty()) return out;

    size_t start = 0;
    for (size_t i = 0; i < normalized.size(); ++i) {
        const char c = normalized[i];
        if ((c == ',' || c == ';' || c == ':') && i + 1 < normalized.size() &&
            std::isspace(static_cast<unsigned char>(normalized[i + 1]))) {
            const std::string piece = trim_copy(normalized.substr(start, i + 1 - start));
            if (!piece.empty()) out.push_back(piece);
            size_t next = i + 1;
            while (next < normalized.size() && std::isspace(static_cast<unsigned char>(normalized[next]))) ++next;
            start = next;
            i = next ? next - 1 : next;
            continue;
        }
        if (std::isspace(static_cast<unsigned char>(c))) {
            size_t j = i;
            while (j < normalized.size() && std::isspace(static_cast<unsigned char>(normalized[j]))) ++j;
            if (j < normalized.size() && is_dash_like(normalized[j])) {
                size_t k = j + 1;
                while (k < normalized.size() && std::isspace(static_cast<unsigned char>(normalized[k]))) ++k;
                if (k > j + 1) {
                    const std::string piece = trim_copy(normalized.substr(start, i - start));
                    if (!piece.empty()) out.push_back(piece);
                    start = k;
                    i = k ? k - 1 : k;
                }
            }
        }
    }

    if (start < normalized.size()) {
        const std::string tail = trim_copy(normalized.substr(start));
        if (!tail.empty()) out.push_back(tail);
    }
    return out;
}

KugelAudioChunkBoundaryType merge_boundary_type(KugelAudioChunkBoundaryType current,
                                                KugelAudioChunkBoundaryType incoming) {
    const int cur = current == KugelAudioChunkBoundaryType::Sentence ? 0 :
                    current == KugelAudioChunkBoundaryType::Clause ? 1 : 2;
    const int in  = incoming == KugelAudioChunkBoundaryType::Sentence ? 0 :
                    incoming == KugelAudioChunkBoundaryType::Clause ? 1 : 2;
    return in > cur ? incoming : current;
}

std::string extract_overlap_prefix(const std::string& text, int overlap_sentences) {
    if (overlap_sentences <= 0) return "";
    const std::vector<std::string> sentences = split_sentences_heuristic(text);
    std::vector<std::string> completed;
    completed.reserve(sentences.size());
    for (const std::string& sentence : sentences) {
        const std::string trimmed = trim_copy(sentence);
        if (!trimmed.empty() && is_sentence_terminal(trimmed.back())) {
            completed.push_back(trimmed);
        }
    }
    if (completed.empty()) return "";
    const size_t keep = std::min<size_t>(completed.size(), static_cast<size_t>(overlap_sentences));
    return join_words(completed, completed.size() - keep, completed.size());
}

std::pair<std::vector<std::string>, KugelAudioChunkBoundaryType>
split_oversized_sentence_syntax_aware(const std::string& sentence, int max_words) {
    const std::vector<std::string> phrase_parts = split_syntax_aware_phrase_parts(sentence);
    if (phrase_parts.size() <= 1) {
        return {{}, KugelAudioChunkBoundaryType::Sentence};
    }

    std::vector<std::string> chunks;
    std::vector<std::string> current_parts;
    int current_count = 0;
    bool used_hard_wrap = false;
    for (const std::string& phrase : phrase_parts) {
        const int phrase_words = word_count(phrase);
        if (phrase_words > max_words) {
            if (!current_parts.empty()) {
                chunks.push_back(join_words(current_parts, 0, current_parts.size()));
                current_parts.clear();
                current_count = 0;
            }
            const std::vector<std::string> wrapped = hard_wrap_words(phrase, max_words);
            chunks.insert(chunks.end(), wrapped.begin(), wrapped.end());
            used_hard_wrap = true;
            continue;
        }
        if (!current_parts.empty() && current_count + phrase_words > max_words) {
            chunks.push_back(join_words(current_parts, 0, current_parts.size()));
            current_parts = {phrase};
            current_count = phrase_words;
        } else {
            current_parts.push_back(phrase);
            current_count += phrase_words;
        }
    }
    if (!current_parts.empty()) chunks.push_back(join_words(current_parts, 0, current_parts.size()));
    return {chunks, used_hard_wrap ? KugelAudioChunkBoundaryType::HardWrap
                                   : KugelAudioChunkBoundaryType::Clause};
}

std::pair<std::vector<std::string>, KugelAudioChunkBoundaryType>
split_oversized_sentence(const std::string& sentence,
                         int max_words,
                         ChunkingStrategy strategy) {
    if (word_count(sentence) <= max_words) {
        return {{sentence}, KugelAudioChunkBoundaryType::Sentence};
    }

    if (strategy == ChunkingStrategy::SyntaxAware) {
        auto syntax_aware = split_oversized_sentence_syntax_aware(sentence, max_words);
        if (!syntax_aware.first.empty()) {
            return syntax_aware;
        }
    }

    const std::vector<std::string> clause_parts = split_clause_parts(sentence);
    if (clause_parts.size() > 1) {
        std::vector<std::string> chunks;
        std::vector<std::string> current_parts;
        int current_count = 0;
        bool used_hard_wrap = false;
        for (const std::string& clause : clause_parts) {
            const int clause_words = word_count(clause);
            if (clause_words > max_words) {
                if (!current_parts.empty()) {
                    chunks.push_back(join_words(current_parts, 0, current_parts.size()));
                    current_parts.clear();
                    current_count = 0;
                }
                const std::vector<std::string> wrapped = hard_wrap_words(clause, max_words);
                chunks.insert(chunks.end(), wrapped.begin(), wrapped.end());
                used_hard_wrap = true;
                continue;
            }
            if (!current_parts.empty() && current_count + clause_words > max_words) {
                chunks.push_back(join_words(current_parts, 0, current_parts.size()));
                current_parts = {clause};
                current_count = clause_words;
            } else {
                current_parts.push_back(clause);
                current_count += clause_words;
            }
        }
        if (!current_parts.empty()) chunks.push_back(join_words(current_parts, 0, current_parts.size()));
        return {chunks, used_hard_wrap ? KugelAudioChunkBoundaryType::HardWrap
                                       : KugelAudioChunkBoundaryType::Clause};
    }

    return {hard_wrap_words(sentence, max_words), KugelAudioChunkBoundaryType::HardWrap};
}

bool is_speaker_boundary(const std::string& left_text, const std::string& right_text) {
    static const std::regex kSpeakerRe(R"(^\s*speaker\s+\d+\s*:)",
                                       std::regex_constants::icase);
    return std::regex_search(left_text, kSpeakerRe) || std::regex_search(right_text, kSpeakerRe);
}

int punctuation_pause_ms(const std::string& text) {
    const std::string trimmed = trim_copy(text);
    if (trimmed.empty()) return 0;
    switch (trimmed.back()) {
        case ',': return 120;
        case ';': return 180;
        case ':': return 180;
        case '.': return 260;
        case '!': return 260;
        case '?': return 260;
        default:  return 0;
    }
}

void append_with_crossfade(std::vector<float>* output,
                           const std::vector<float>& right,
                           int crossfade_ms,
                           int sample_rate) {
    if (!output) return;
    if (crossfade_ms <= 0 || output->empty() || right.empty()) {
        output->insert(output->end(), right.begin(), right.end());
        return;
    }

    const int requested = sample_rate > 0 ? (sample_rate * crossfade_ms) / 1000 : 0;
    const size_t overlap = std::min<size_t>(
        static_cast<size_t>(std::max(0, requested)),
        std::min(output->size(), right.size()));
    if (overlap == 0) {
        output->insert(output->end(), right.begin(), right.end());
        return;
    }

    const size_t left_keep = output->size() - overlap;
    std::vector<float> blended(overlap);
    for (size_t i = 0; i < overlap; ++i) {
        const float fade_out = overlap == 1 ? 1.0f : (1.0f - static_cast<float>(i) / static_cast<float>(overlap - 1));
        const float fade_in  = overlap == 1 ? 0.0f : (static_cast<float>(i) / static_cast<float>(overlap - 1));
        blended[i] = (*output)[left_keep + i] * fade_out + right[i] * fade_in;
    }

    output->resize(left_keep);
    output->insert(output->end(), blended.begin(), blended.end());
    output->insert(output->end(), right.begin() + static_cast<std::ptrdiff_t>(overlap), right.end());
}

}  // namespace

const char* kugelaudio_chunk_boundary_type_name(KugelAudioChunkBoundaryType type) {
    switch (type) {
        case KugelAudioChunkBoundaryType::Sentence: return "sentence";
        case KugelAudioChunkBoundaryType::Clause:   return "clause";
        case KugelAudioChunkBoundaryType::HardWrap: return "hard_wrap";
    }
    return "unknown";
}

std::string trim_overlap_to_word_budget(const std::string& prefix, int max_overlap_words) {
    if (prefix.empty() || max_overlap_words <= 0) return "";
    if (word_count(prefix) <= max_overlap_words) return prefix;

    std::vector<std::string> sentences = split_sentences_heuristic(prefix);
    while (sentences.size() > 1) {
        std::string candidate = join_words(sentences, 1, sentences.size());
        if (word_count(candidate) <= max_overlap_words) return candidate;
        sentences.erase(sentences.begin());
    }

    const std::vector<std::string> words = split_words(sentences.empty() ? prefix : sentences.front());
    if (static_cast<int>(words.size()) <= max_overlap_words) {
        return sentences.empty() ? prefix : sentences.front();
    }
    const size_t keep = static_cast<size_t>(std::max(0, max_overlap_words));
    return join_words(words, words.size() - keep, words.size());
}

int min_new_words_for_overlap_budget(int max_words_per_chunk) {
    if (max_words_per_chunk <= 1) return 1;
    return std::min(max_words_per_chunk - 1, std::max(8, max_words_per_chunk / 4));
}

bool split_kugelaudio_text_into_chunks(const std::string& text,
                                       int max_words_per_chunk,
                                       int overlap_sentences,
                                       ChunkingStrategy strategy,
                                       KugelAudioChunkPlan* out,
                                       std::string* error) {
    if (!out) {
        if (error) *error = "chunk plan output is required";
        return false;
    }
    out->chunks.clear();
    out->boundary_types.clear();
    out->overlap_prefixes.clear();
    out->new_texts.clear();

    const std::string normalized = trim_copy(text);
    if (normalized.empty()) {
        if (error) *error = "Text input is required";
        return false;
    }
    if (max_words_per_chunk <= 0) {
        if (error) *error = "max_words_per_chunk must be greater than 0";
        return false;
    }
    if (overlap_sentences < 0) {
        if (error) *error = "overlap_sentences must be greater than or equal to 0";
        return false;
    }

    struct Piece {
        std::string text;
        KugelAudioChunkBoundaryType boundary = KugelAudioChunkBoundaryType::Sentence;
    };
    std::vector<Piece> pieces;
    const std::vector<std::string> sentence_candidates = split_sentences_heuristic(normalized);
    for (const std::string& sentence : sentence_candidates) {
        auto split = split_oversized_sentence(sentence, max_words_per_chunk, strategy);
        for (const std::string& piece : split.first) {
            if (!trim_copy(piece).empty()) pieces.push_back({piece, split.second});
        }
    }

    size_t idx = 0;
    std::string previous_new_text;
    while (idx < pieces.size()) {
        std::string prefix;
        if (!previous_new_text.empty() && overlap_sentences > 0) {
            prefix = extract_overlap_prefix(previous_new_text, overlap_sentences);
            const int min_new = min_new_words_for_overlap_budget(max_words_per_chunk);
            const int max_overlap = std::max(0, max_words_per_chunk - min_new);
            prefix = trim_overlap_to_word_budget(prefix, max_overlap);
        }
        const int prefix_words = word_count(prefix);
        const int new_budget = std::max(1, max_words_per_chunk - prefix_words);

        std::vector<std::string> current_parts;
        int current_count = 0;
        KugelAudioChunkBoundaryType current_boundary = KugelAudioChunkBoundaryType::Sentence;

        while (idx < pieces.size()) {
            const int piece_words = word_count(pieces[idx].text);
            if (piece_words <= 0) {
                ++idx;
                continue;
            }
            if (current_parts.empty() && piece_words > new_budget) {
                const std::vector<std::string> words = split_words(pieces[idx].text);
                const size_t take = std::min(words.size(), static_cast<size_t>(new_budget));
                current_parts.push_back(join_words(words, 0, take));
                current_count += static_cast<int>(take);
                current_boundary = KugelAudioChunkBoundaryType::HardWrap;
                if (take < words.size()) {
                    pieces[idx].text = join_words(words, take, words.size());
                    pieces[idx].boundary = KugelAudioChunkBoundaryType::HardWrap;
                } else {
                    ++idx;
                }
                break;
            }
            if (!current_parts.empty() && current_count + piece_words > new_budget) {
                break;
            }
            current_parts.push_back(pieces[idx].text);
            current_count += piece_words;
            current_boundary = current_parts.size() == 1
                ? pieces[idx].boundary
                : merge_boundary_type(current_boundary, pieces[idx].boundary);
            ++idx;
        }

        if (current_parts.empty()) break;
        const std::string new_text = join_words(current_parts, 0, current_parts.size());
        const std::string chunk = prefix.empty() ? new_text : (prefix + " " + new_text);
        out->overlap_prefixes.push_back(prefix);
        out->new_texts.push_back(new_text);
        out->chunks.push_back(chunk);
        out->boundary_types.push_back(current_boundary);
        previous_new_text = new_text;
    }
    return true;
}

int kugelaudio_pause_duration_ms_for_boundary(const std::string& left_text,
                                              const std::string& right_text,
                                              ChunkPauseMode pause_mode) {
    if (pause_mode == ChunkPauseMode::None) return 0;
    int duration_ms = punctuation_pause_ms(left_text);
    if (pause_mode == ChunkPauseMode::SpeakerAware && is_speaker_boundary(left_text, right_text)) {
        duration_ms = std::max(duration_ms, 360);
    }
    return duration_ms;
}

bool stitch_kugelaudio_audio_chunks(const std::vector<std::vector<float>>& audio_chunks,
                                    const std::vector<std::string>& chunk_texts,
                                    ChunkPauseMode pause_mode,
                                    int crossfade_ms,
                                    int sample_rate,
                                    std::vector<float>* out,
                                    std::string* error) {
    if (!out) {
        if (error) *error = "stitched output is required";
        return false;
    }
    out->clear();
    if (audio_chunks.empty()) {
        if (error) *error = "audio_chunks must not be empty";
        return false;
    }
    if (crossfade_ms < 0) {
        if (error) *error = "crossfade_ms must be >= 0";
        return false;
    }
    if (!chunk_texts.empty() && chunk_texts.size() != audio_chunks.size()) {
        if (error) *error = "chunk_texts length must match audio_chunks length";
        return false;
    }

    *out = audio_chunks.front();
    if (audio_chunks.size() == 1) return true;

    const std::vector<std::string> texts = chunk_texts.empty()
        ? std::vector<std::string>(audio_chunks.size())
        : chunk_texts;

    for (size_t i = 1; i < audio_chunks.size(); ++i) {
        const int pause_ms = kugelaudio_pause_duration_ms_for_boundary(texts[i - 1], texts[i], pause_mode);
        const int pause_samples = sample_rate > 0 ? (sample_rate * pause_ms) / 1000 : 0;
        if (pause_samples > 0) {
            out->insert(out->end(), static_cast<size_t>(pause_samples), 0.0f);
        }
        append_with_crossfade(out, audio_chunks[i], crossfade_ms, sample_rate);
    }
    return true;
}

}  // namespace detail
}  // namespace vv
