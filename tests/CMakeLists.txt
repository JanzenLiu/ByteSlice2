include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
file(GLOB test_src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
add_executable(all-test ${test_src})
target_link_libraries(all-test byteslice-core gtest gtest_main)
if(autoplay)
message(STATUS "Autoplay tests after build.")
add_custom_command(TARGET all-test
                   POST_BUILD
                   COMMAND all-test
                   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                   COMMENT "Running all test case using google-test.")
endif()
