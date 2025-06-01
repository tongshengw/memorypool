add_library(malloc-tracker STATIC 
    trackedmalloc.c 
    trackedmalloc.h
)
target_include_directories(malloc-tracker PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(malloc-tracker-test test-malloc-tracker.c)
target_include_directories(malloc-tracker-test PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(malloc-tracker-test PUBLIC malloc-tracker)

enable_testing()
add_test(NAME basic-malloc-tracker COMMAND malloc-tracker-test)
set_tests_properties(basic-malloc-tracker PROPERTIES
    PASS_REGULAR_EXPRESSION ".*900"
)