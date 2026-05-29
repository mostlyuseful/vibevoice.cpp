#include "vibevoice_tts.hpp"

#include <cstdio>

int main() {
    const auto def = vv::detail::kugelaudio_solver_config_for_test(0);
    if (def.num_train_timesteps != 1000 ||
        def.num_inference_steps != 20 ||
        def.solver_order != 2 ||
        !def.lower_order_final) {
        std::fprintf(stderr,
                     "FAIL: default KugelAudio solver config mismatch: train=%d infer=%d order=%d lower_final=%d\n",
                     def.num_train_timesteps,
                     def.num_inference_steps,
                     def.solver_order,
                     def.lower_order_final ? 1 : 0);
        return 1;
    }

    const auto custom = vv::detail::kugelaudio_solver_config_for_test(7);
    if (custom.num_train_timesteps != 1000 ||
        custom.num_inference_steps != 7 ||
        custom.solver_order != 2 ||
        !custom.lower_order_final) {
        std::fprintf(stderr,
                     "FAIL: custom KugelAudio solver config mismatch: train=%d infer=%d order=%d lower_final=%d\n",
                     custom.num_train_timesteps,
                     custom.num_inference_steps,
                     custom.solver_order,
                     custom.lower_order_final ? 1 : 0);
        return 2;
    }

    std::printf("KugelAudio reuse coverage wiring OK\n");
    return 0;
}
