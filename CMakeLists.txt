cmake_minimum_required(VERSION 3.18)
project(kugelaudio
        VERSION   0.0.1
        LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------- options ----------
option(KUGELAUDIO_BUILD_TESTS    "Build kugelaudio.cpp tests"    OFF)
option(KUGELAUDIO_BUILD_EXAMPLES "Build kugelaudio.cpp examples" ON)
option(KUGELAUDIO_BUILD_SERVER   "Build kugelaudio.cpp server"   OFF)
option(KUGELAUDIO_TEST_LARGE     "Run large/E2E tests"           OFF)
option(KUGELAUDIO_SHARED         "Build shared library"          OFF)

# Backend toggles — forwarded to the embedded ggml submodule. Off by
# default; enable with `-DKUGELAUDIO_GGML_CUDA=ON`, etc.
option(KUGELAUDIO_GGML_CUDA      "Build ggml with CUDA backend"     OFF)
option(KUGELAUDIO_GGML_METAL     "Build ggml with Metal backend"    OFF)
option(KUGELAUDIO_GGML_VULKAN    "Build ggml with Vulkan backend"   OFF)
option(KUGELAUDIO_GGML_HIPBLAS   "Build ggml with hipBLAS backend"  OFF)

# Deprecated option aliases retained for compatibility while the fork is renamed.
option(VIBEVOICE_BUILD_TESTS    "Deprecated alias for KUGELAUDIO_BUILD_TESTS"    ${KUGELAUDIO_BUILD_TESTS})
option(VIBEVOICE_BUILD_EXAMPLES "Deprecated alias for KUGELAUDIO_BUILD_EXAMPLES" ${KUGELAUDIO_BUILD_EXAMPLES})
option(VIBEVOICE_BUILD_SERVER   "Deprecated alias for KUGELAUDIO_BUILD_SERVER"   ${KUGELAUDIO_BUILD_SERVER})
option(VIBEVOICE_TEST_LARGE     "Deprecated alias for KUGELAUDIO_TEST_LARGE"     ${KUGELAUDIO_TEST_LARGE})
option(VIBEVOICE_SHARED         "Deprecated alias for KUGELAUDIO_SHARED"         ${KUGELAUDIO_SHARED})
option(VIBEVOICE_GGML_CUDA      "Deprecated alias for KUGELAUDIO_GGML_CUDA"      ${KUGELAUDIO_GGML_CUDA})
option(VIBEVOICE_GGML_METAL     "Deprecated alias for KUGELAUDIO_GGML_METAL"     ${KUGELAUDIO_GGML_METAL})
option(VIBEVOICE_GGML_VULKAN    "Deprecated alias for KUGELAUDIO_GGML_VULKAN"    ${KUGELAUDIO_GGML_VULKAN})
option(VIBEVOICE_GGML_HIPBLAS   "Deprecated alias for KUGELAUDIO_GGML_HIPBLAS"   ${KUGELAUDIO_GGML_HIPBLAS})
foreach(_opt BUILD_TESTS BUILD_EXAMPLES BUILD_SERVER TEST_LARGE SHARED GGML_CUDA GGML_METAL GGML_VULKAN GGML_HIPBLAS)
    if(VIBEVOICE_${_opt} AND NOT KUGELAUDIO_${_opt})
        set(KUGELAUDIO_${_opt} ON CACHE BOOL "Compatibility value from VIBEVOICE_${_opt}" FORCE)
    endif()
endforeach()

# ---------- ggml ----------
# We embed ggml as a submodule under third_party/ggml. We forward common
# GGML_* options so users can drive backends from the top-level cache.
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
if(KUGELAUDIO_GGML_CUDA)
    set(GGML_CUDA   ON CACHE BOOL "" FORCE)
endif()
if(KUGELAUDIO_GGML_METAL)
    set(GGML_METAL  ON CACHE BOOL "" FORCE)
endif()
if(KUGELAUDIO_GGML_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()
if(KUGELAUDIO_GGML_HIPBLAS)
    set(GGML_HIPBLAS ON CACHE BOOL "" FORCE)
endif()
add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)

# ---------- libvibevoice ----------
set(VIBEVOICE_SOURCES
    src/common.cpp
    src/backend.cpp
    src/vibevoice.cpp
    src/vibevoice_capi.cpp
    src/audio_io.cpp
    src/model_loader.cpp
    src/tokenizer.cpp
    src/kugelaudio_chunking.cpp
    src/qwen2.cpp
    src/conv1d.cpp
    src/acoustic_tokenizer.cpp
    src/diffusion_head.cpp
    src/dpm_solver.cpp
    src/vibevoice_tts.cpp
    src/vibevoice_asr.cpp
)

set(VIBEVOICE_PUBLIC_HEADERS
    include/vibevoice.h
    include/vibevoice_capi.h
)

if(KUGELAUDIO_SHARED)
    add_library(vibevoice SHARED ${VIBEVOICE_SOURCES})
    target_compile_definitions(vibevoice PUBLIC VIBEVOICE_SHARED)
    target_compile_definitions(vibevoice PRIVATE VIBEVOICE_BUILD)
else()
    add_library(vibevoice STATIC ${VIBEVOICE_SOURCES})
endif()

set_target_properties(vibevoice PROPERTIES OUTPUT_NAME kugelaudio)

target_include_directories(vibevoice
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_link_libraries(vibevoice PUBLIC ggml)

# When a ggml backend (CUDA / Metal / Vulkan / hipBLAS / blas) is enabled,
# explicitly link its target so the static library carries the symbols
# (mirrors the pattern in LocalAI's qwen3-tts-cpp wrapper).
foreach(_be cuda metal vulkan hipblas blas)
    if(TARGET ggml-${_be})
        target_link_libraries(vibevoice PUBLIC ggml-${_be})
        string(TOUPPER ${_be} _BE)
        target_compile_definitions(vibevoice PUBLIC VIBEVOICE_HAVE_${_BE})
    endif()
endforeach()
if(KUGELAUDIO_GGML_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        target_link_libraries(vibevoice PUBLIC CUDA::cudart)
    endif()
endif()

target_compile_options(vibevoice PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic -Wno-unused-parameter>
)

# ---------- subdirs ----------
if(KUGELAUDIO_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(KUGELAUDIO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---------- install ----------
include(GNUInstallDirs)

install(TARGETS vibevoice
        EXPORT  kugelaudioTargets
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES ${VIBEVOICE_PUBLIC_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
