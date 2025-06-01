add_library(malloc-tracker STATIC 
    trackedmalloc.c 
    trackedmalloc.h
)
target_include_directories(malloc-tracker PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR} PUBLIC 
) 