cmake_minimum_required(VERSION 3.27)
project(memorypool LANGUAGES C DESCRIPTION "Memory Pool Allocator for C")

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(BUILD_TESTS "Build tests" ON)
option(TESTS_USE_POOL_MALLOC "Builds tests with poolmalloc" ON)

set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0")

add_library(poolalloc-cpu STATIC cpu-concept/poolalloc.c cpu-concept/poolalloc.h)
target_include_directories(poolalloc-cpu PUBLIC cpu-concept)

add_library(malloc-tracker STATIC malloc-tracker/trackedmalloc.c malloc-tracker/trackedmalloc.h)
target_include_directories(malloc-tracker PUBLIC malloc-tracker)

