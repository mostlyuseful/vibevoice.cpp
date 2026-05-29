add_executable(kugelaudio-cli main.cpp)
target_link_libraries(kugelaudio-cli PRIVATE vibevoice)
target_include_directories(kugelaudio-cli PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_options(kugelaudio-cli PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)

set_target_properties(kugelaudio-cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Deprecated compatibility binary name for older scripts/tests.
add_custom_command(TARGET kugelaudio-cli POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:kugelaudio-cli>
            ${CMAKE_BINARY_DIR}/bin/vibevoice-cli
)
add_custom_target(vibevoice-cli DEPENDS kugelaudio-cli)
