add_executable(kugelaudio-quantize main.cpp)
target_link_libraries(kugelaudio-quantize PRIVATE vibevoice)
target_include_directories(kugelaudio-quantize PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_options(kugelaudio-quantize PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)
set_target_properties(kugelaudio-quantize PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Deprecated compatibility binary name for older scripts/tests.
add_custom_command(TARGET kugelaudio-quantize POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:kugelaudio-quantize>
            ${CMAKE_BINARY_DIR}/bin/vibevoice-quantize
)
add_custom_target(vibevoice-quantize DEPENDS kugelaudio-quantize)
