set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "61;75;86")

add_executable(gpu-concept test.cu poolalloc.cu)
target_link_libraries(gpu-concept ${CUDA_LIBRARIES})
target_include_directories(gpu-concept PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(gpu-concept PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)