# ============================
# A collection of unit tests
# ============================

enable_testing()

# C++ tests
setup_test(demo)
#setup_test(fuzzy-test)
setup_test(linalg-test)
setup_test(volume-test)

# cuda tests
setup_cuda_test(test-cuda)
setup_cuda_test(linalg-gpu)
setup_cuda_test(fuzzy-test-gpu)
setup_cuda_test(performance-test-gpu)

setup_python_test_script(performance-test-py)

# copy data file
file(GLOB txtfiles *.txt)
foreach(txtfile ${txtfiles})
  get_filename_component(name ${txtfile} NAME)
  message(STATUS "Copying ${txtfile} to ${name}")
  configure_file(${txtfile} ${CMAKE_CURRENT_BINARY_DIR}/${name} @ONLY)
endforeach()
