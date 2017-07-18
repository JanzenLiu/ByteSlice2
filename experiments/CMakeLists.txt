include_directories("${CMAKE_SOURCE_DIR}/util" "${CMAKE_CURRENT_SOURCE_DIR}")
file(GLOB experiment_src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
foreach(srcfile ${experiment_src})
    string(REPLACE ".cpp" "" exp ${srcfile})
    add_executable(${exp} ${srcfile})
    target_link_libraries(${exp} byteslice-core utility)
endforeach()

