add_library(poolalloc-cpu STATIC poolalloc.c poolalloc.h)
target_include_directories(poolalloc-cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(poolalloc-cpu PUBLIC m)

add_executable(fuzzy-cpu fuzzy-test.c)
target_link_libraries(fuzzy-cpu PRIVATE poolalloc-cpu)

add_custom_command(TARGET fuzzy-cpu POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/fuzzy.sh
    $<TARGET_FILE_DIR:fuzzy-cpu>/fuzzy.sh
)