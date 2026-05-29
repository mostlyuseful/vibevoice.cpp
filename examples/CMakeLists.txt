add_subdirectory(cli)
add_subdirectory(quantize)

if(KUGELAUDIO_BUILD_SERVER)
    add_subdirectory(server)
endif()
