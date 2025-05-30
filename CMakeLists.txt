cmake_minimum_required(VERSION 3.27)
project(memorypool LANGUAGES C DESCRIPTION "Memory Pool Allocator for C")

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

option(BUILD_TESTS "Build tests" ON)

set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0")

add_library(poolalloc-cpu STATIC cpu-concept/poolalloc.c cpu-concept/poolalloc.h)
target_include_directories(poolalloc-cpu PUBLIC cpu-concept)

add_executable(fuzzy-test cpu-concept/fuzzy-test.c)
target_link_libraries(fuzzy-test PRIVATE poolalloc-cpu)

add_library(malloc-tracker STATIC malloc-tracker/trackedmalloc.c malloc-tracker/trackedmalloc.h)
target_include_directories(malloc-tracker PUBLIC malloc-tracker)

add_library(linalg-ref STATIC linalg/leastsq_kkt.c linalg/leastsq.c linalg/lubksb.c linalg/ludcmp.c linalg/luminv.c linalg/mmdot.c linalg/mvdot.c linalg/vvdot.c linalg/linalg.h)
target_include_directories(linalg-ref PUBLIC linalg)

add_library(linalg-pool STATIC linalg/leastsq_kkt.c linalg/leastsq.c linalg/lubksb.c linalg/ludcmp.c linalg/luminv.c linalg/mmdot.c linalg/mvdot.c linalg/vvdot.c linalg/linalg.h)
target_include_directories(linalg-pool PUBLIC linalg)
target_include_directories(linalg-pool PUBLIC cpu-concept)
target_compile_definitions(linalg-pool PUBLIC USE_POOL)

add_executable(linalg-ref-test linalg/main.c linalg/linalg.h)
target_include_directories(linalg-ref-test PUBLIC linalg)
target_link_libraries(linalg-ref-test PRIVATE linalg-ref)

add_executable(linalg-pool-test linalg/main.c linalg/linalg.h)
target_compile_definitions(linalg-pool-test PRIVATE USE_POOL)
target_include_directories(linalg-pool-test PUBLIC linalg)
target_include_directories(linalg-pool-test PUBLIC cpu-concept)
target_link_libraries(linalg-pool-test PRIVATE poolalloc-cpu)
target_link_libraries(linalg-pool-test PRIVATE linalg-pool)


