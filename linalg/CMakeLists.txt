add_library(linalg-ref STATIC leastsq_kkt.c leastsq.c lubksb.c ludcmp.c luminv.c mmdot.c mvdot.c vvdot.c)
target_include_directories(linalg-ref PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(linalg-ref PUBLIC macros)

add_library(linalg-pool STATIC leastsq_kkt.c leastsq.c lubksb.c ludcmp.c luminv.c mmdot.c mvdot.c vvdot.c)
target_include_directories(linalg-pool PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(linalg-pool PUBLIC USE_POOL)
target_link_libraries(linalg-pool PUBLIC macros)