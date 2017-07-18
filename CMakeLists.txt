cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project ("ByteSlice")


set(warnings "-Wall")
set(archs "-mavx2 -m64 -std=c++11 -fopenmp")

# Set default build type as debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Default build type: Debug." FORCE)
endif()

if(NOT CONFIGURED_ONCE)
    set(CMAKE_CXX_FLAGS "${warnings} ${archs}"
        CACHE STRING "Flags used by the compiler during all build types." FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "-ggdb3 -O1 -D NPREFETCH" 
        CACHE STRING "Flags used by the complier during debug build type." FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -D NDEBUG -D NPREFETCH" 
        CACHE STRING "Flags used by the compiler during release build type." FORCE)
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -ggdb3 -D NDEBUG -D NPREFETCH" 
        CACHE STRING "Flags used by the compiler during relwithdebinfo." FORCE)
endif()


add_definitions(-DNEARLYSTOP)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}"  "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory(src)
add_subdirectory(util)
add_subdirectory(experiments)
add_subdirectory(example)


#############################
##  Set up test
############################
option(test "Build all tests." ON)
option(autoplay "Auto play test immediately after build." ON)
if(test)
    message(STATUS "Tests will be built.")
    enable_testing()
    add_subdirectory(gtest-1.7.0)
    add_subdirectory(tests)
    add_test(NAME all-test COMMAND all-test)
endif()

set(CONFIGURED_ONCE TRUE CACHE INTERNAL "A flag showing that CMake has configured at least once.")


