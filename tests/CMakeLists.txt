add_executable(linalg-ref-test linalg-test.c)
target_include_directories(linalg-ref-test PUBLIC ${CMAKE_SOURCE_DIR}/linalg)
target_link_libraries(linalg-ref-test PRIVATE linalg-ref)

add_executable(linalg-pool-test linalg-test.c)
target_include_directories(linalg-pool-test PUBLIC ${CMAKE_SOURCE_DIR}/linalg PUBLIC ${CMAKE_SOURCE_DIR}/cpu-concept)
target_compile_definitions(linalg-pool-test PRIVATE USE_POOL)
target_link_libraries(linalg-pool-test PRIVATE linalg-pool)