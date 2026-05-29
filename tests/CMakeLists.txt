# Unit / integration tests. Each test_*.cpp becomes an executable + ctest entry.
# Large or weight-dependent tests are gated by VIBEVOICE_TEST_LARGE.

set(VV_FIXTURES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/fixtures)

function(vv_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE vibevoice)
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party
    )
    target_compile_definitions(${name} PRIVATE
        VV_FIXTURES_DIR="${VV_FIXTURES_DIR}"
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# Tests are added by later milestones; a placeholder smoke test
# proves the test machinery is wired up.
vv_add_test(test_smoke)
vv_add_test(test_audio_io)
vv_add_test(test_tokenizer)
vv_add_test(test_rope)
vv_add_test(test_qwen2_block)
vv_add_test(test_qwen2_resident)
vv_add_test(test_sconv1d)
vv_add_test(test_block1d)
vv_add_test(test_acoustic)
vv_add_test(test_acoustic_decoder_streaming)
vv_add_test(test_diffusion_head)
vv_add_test(test_dpm_solver)
vv_add_test(test_backend_selection)
vv_add_test(test_load_realtime)
vv_add_test(test_load_15b)
vv_add_test(test_kugelaudio_loader)
vv_add_test(test_kugelaudio_feature_gating)
vv_add_test(test_kugelaudio_prompt)
vv_add_test(test_kugelaudio_conditioning)
vv_add_test(test_kugelaudio_chunking)
vv_add_test(test_kugelaudio_capi)
vv_add_test(test_kugelaudio_cli_conditioning)
vv_add_test(test_kugelaudio_cli_chunking)
vv_add_test(test_kugelaudio_cli_chunking_e2e)
vv_add_test(test_kugelaudio_cli_chunking_seed)
vv_add_test(test_kugelaudio_cli_e2e)
vv_add_test(test_kugelaudio_cli_rejections)
vv_add_test(test_kugelaudio_cli_seed)
vv_add_test(test_kugelaudio_determinism)
vv_add_test(test_kugelaudio_generation_tokens)
vv_add_test(test_kugelaudio_reused_components)
vv_add_test(test_kugelaudio_reuse_coverage)
vv_add_test(test_kugelaudio_decode_smoke)
vv_add_test(test_kugelaudio_q8_0_smoke)
vv_add_test(test_kugelaudio_cuda_smoke)
vv_add_test(test_kugelaudio_cuda_conditioning)
vv_add_test(test_kugelaudio_vulkan_smoke)
vv_add_test(test_kugelaudio_vulkan_hybrid_decode)
vv_add_test(test_kugelaudio_vulkan_streamed_decode)
vv_add_test(test_kugelaudio_quantize_cli)
vv_add_test(test_e2e_tts)
# Large legacy / migration tests. Voice.gguf-dependent realtime tests are no
# longer part of the published surface, so they are not registered here.
if(VIBEVOICE_TEST_LARGE)
    vv_add_test(test_capi)
    vv_add_test(test_encoder_chunked_parity)
    vv_add_test(test_15b_smoke)
    vv_add_test(test_15b_closed_loop)
    vv_add_test(test_15b_multispeaker)
    set_tests_properties(test_capi
                         test_encoder_chunked_parity
                         test_15b_smoke test_15b_closed_loop test_15b_multispeaker
                         PROPERTIES SKIP_RETURN_CODE 77)
endif()
set_tests_properties(test_tokenizer test_rope test_qwen2_block test_qwen2_resident
                     test_sconv1d test_block1d test_acoustic test_acoustic_decoder_streaming test_diffusion_head
                     test_dpm_solver test_backend_selection test_load_realtime test_load_15b
                     test_kugelaudio_loader test_kugelaudio_feature_gating
                     test_kugelaudio_prompt test_kugelaudio_conditioning
                     test_kugelaudio_chunking test_kugelaudio_capi
                     test_kugelaudio_cli_conditioning test_kugelaudio_cli_chunking
                     test_kugelaudio_cli_chunking_e2e test_kugelaudio_cli_chunking_seed test_kugelaudio_cli_e2e test_kugelaudio_cli_rejections
                     test_kugelaudio_cli_seed test_kugelaudio_determinism
                     test_kugelaudio_generation_tokens
                     test_kugelaudio_reused_components
                     test_kugelaudio_reuse_coverage test_kugelaudio_decode_smoke
                     test_kugelaudio_q8_0_smoke test_kugelaudio_cuda_smoke
                     test_kugelaudio_cuda_conditioning test_kugelaudio_vulkan_smoke
                     test_kugelaudio_vulkan_hybrid_decode test_kugelaudio_vulkan_streamed_decode
                     test_kugelaudio_quantize_cli
                     test_e2e_tts
                     PROPERTIES SKIP_RETURN_CODE 77)
