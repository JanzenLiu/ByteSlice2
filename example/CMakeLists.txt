file(GLOB examples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
foreach(file ${examples})
    string(REPLACE ".cpp" "" tg ${file})
    add_executable(${tg} ${file})
    target_link_libraries(${tg} byteslice-core)
endforeach()

