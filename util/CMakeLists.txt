ADD_CUSTOM_COMMAND(
    OUTPUT  "${CMAKE_CURRENT_SOURCE_DIR}/libpcm.a"
    COMMAND make -C intel-pcm
    COMMAND ${CMAKE_AR} -rs libpcm.a intel-pcm/*.o
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Compile Intel PCM library into util/libpcm.a")

add_custom_target(pcm ALL DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/libpcm.a")
        

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/intel-pcm")
file(GLOB files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
add_library(utility STATIC ${files})

add_dependencies(utility pcm)
target_link_libraries(utility "${CMAKE_CURRENT_SOURCE_DIR}/libpcm.a")

