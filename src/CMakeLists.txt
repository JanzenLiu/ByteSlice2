file(GLOB byteslice_src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
add_library(byteslice-core STATIC ${byteslice_src})
