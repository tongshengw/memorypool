add_library(linalg-ref STATIC mmdot.c mvdot.c vvdot.c)
# add_library(linalg-ref STATIC leastsq_kkt.c leastsq.c lubksb.c ludcmp.c luminv.c mmdot.c mvdot.c vvdot.c)
target_include_directories(linalg-ref PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_SOURCE_DIR}/macros
)
target_link_libraries(linalg-ref PUBLIC macros)

add_library(linalg-pool STATIC mmdot.c mvdot.c vvdot.c)
# add_library(linalg-pool STATIC leastsq_kkt.c leastsq.c lubksb.c ludcmp.c luminv.c mmdot.c mvdot.c vvdot.c)
target_include_directories(linalg-pool PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_SOURCE_DIR}/macros
    PUBLIC ${CMAKE_SOURCE_DIR}/cpu-concept
)


target_compile_definitions(linalg-pool PUBLIC USE_POOL)
target_link_libraries(linalg-pool PUBLIC macros)

add_library(linalg-gpu STATIC mmdot.c mvdot.c vvdot.c)
target_include_directories(linalg-gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_SOURCE_DIR}/macros
    PUBLIC ${CMAKE_SOURCE_DIR}/gpu-concept
)

target_compile_definitions(linalg-gpu PUBLIC USE_GPU)
target_link_libraries(linalg-gpu PUBLIC macros poolalloc-gpu)