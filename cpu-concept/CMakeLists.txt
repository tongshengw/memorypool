add_library(poolalloc-cpu STATIC poolalloc.c poolalloc.h)
target_include_directories(poolalloc-cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(poolalloc-cpu PUBLIC m)

add_executable(fuzzy-test fuzzy-test.c)
target_link_libraries(fuzzy-test PRIVATE poolalloc-cpu)