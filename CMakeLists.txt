cmake_minimum_required(VERSION 3.27)
project(memorypool LANGUAGES C DESCRIPTION "Memory Pool Allocator for C")

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

option(BUILD_TESTS "Build tests" ON)

set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0")

add_subdirectory(cpu-concept)
add_subdirectory(malloc-tracker)
add_subdirectory(macros)
add_subdirectory(linalg)
add_subdirectory(tests)
