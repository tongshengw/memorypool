set(CMAKE_CUDA_STANDARD 17)

add_executable(gpu-concept test.cu)
target_link_libraries(gpu-concept ${CUDA_LIBRARIES})
target_include_directories(gpu-concept PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})