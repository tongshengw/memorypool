set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "86")

add_library(poolalloc-gpu STATIC poolalloc.cu)
target_link_libraries(poolalloc-gpu PUBLIC ${CUDA_LIBRARIES})
target_include_directories(poolalloc-gpu 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
set_target_properties(poolalloc-gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(gpu-concept test.cu)
target_link_libraries(gpu-concept ${CUDA_LIBRARIES} poolalloc-gpu)
target_include_directories(gpu-concept PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(gpu-concept PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(fuzzy-gpu fuzzy-test.cu)
target_link_libraries(fuzzy-gpu ${CUDA_LIBRARIES} poolalloc-gpu)
target_include_directories(fuzzy-gpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(fuzzy-gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_custom_command(TARGET fuzzy-gpu POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/fuzzy-script.py
    $<TARGET_FILE_DIR:fuzzy-gpu>/fuzzy-script.py
)