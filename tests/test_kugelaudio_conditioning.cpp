#include "vibevoice_speech_helpers.hpp"

#include <cstdio>
#include <vector>

int main() {
    const int hidden = 2;
    const int T = 3;
    const std::vector<float> acoustic = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
    };
    const std::vector<float> semantic = {
        10.0f, 20.0f,
        30.0f, 40.0f,
        50.0f, 60.0f,
    };

    std::vector<float> fused;
    if (!vv::detail::fuse_conditioning_features(acoustic, semantic, hidden, T, &fused)) {
        std::fprintf(stderr, "FAIL: fuse_conditioning_features rejected valid inputs\n");
        return 1;
    }
    const std::vector<float> expected = {
        11.0f, 22.0f,
        33.0f, 44.0f,
        55.0f, 66.0f,
    };
    if (fused != expected) {
        std::fprintf(stderr, "FAIL: fused conditioning features mismatch\n");
        return 2;
    }

    std::vector<float> bad;
    if (vv::detail::fuse_conditioning_features(acoustic, {1.0f, 2.0f}, hidden, T, &bad)) {
        std::fprintf(stderr, "FAIL: fuse_conditioning_features accepted mismatched inputs\n");
        return 3;
    }

    std::printf("KugelAudio conditioning fusion OK\n");
    return 0;
}
