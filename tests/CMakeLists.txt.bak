add_executable(linalg-ref-test linalg-test.c)
target_include_directories(linalg-ref-test PUBLIC ${CMAKE_SOURCE_DIR}/linalg)
target_link_libraries(linalg-ref-test PRIVATE linalg-ref)

add_executable(linalg-pool-test linalg-test.c)
target_include_directories(linalg-pool-test PUBLIC ${CMAKE_SOURCE_DIR}/linalg PUBLIC ${CMAKE_SOURCE_DIR}/cpu-concept)
target_compile_definitions(linalg-pool-test PRIVATE USE_POOL)
target_link_libraries(linalg-pool-test PRIVATE linalg-pool PRIVATE poolalloc-cpu)

add_executable(linalg-gpu-test linalg-gpu.cu)
target_include_directories(linalg-gpu-test PUBLIC ${CMAKE_SOURCE_DIR}/linalg PUBLIC ${CMAKE_SOURCE_DIR}/gpu-concept ${CMAKE_SOURCE_DIR}/macros ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(linalg-gpu-test PRIVATE linalg-gpu PRIVATE poolalloc-gpu PUBLIC ${CUDA_LIBRARIES})
set_target_properties(linalg-gpu-test PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)

foreach(target IN ITEMS linalg-ref-test linalg-pool-test)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/X.txt
        $<TARGET_FILE_DIR:${target}>
    )
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/Y.txt
        $<TARGET_FILE_DIR:${target}>
    )
endforeach()

enable_testing()
add_test(NAME linalg-test-compare
    COMMAND /bin/sh -c 
    "./linalg-pool-test > out1.txt && ./linalg-ref-test > out2.txt && diff out1.txt out2.txt"
)