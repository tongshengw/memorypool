# ============================
# A collection of unit tests
# ============================

enable_testing()

# C++ tests
setup_test(demo)
#setup_test(fuzzy-test)
setup_test(linalg-test)
setup_test(volume-test)

# copy data file
file(GLOB txtfiles *.txt)
foreach(txtfile ${txtfiles})
  get_filename_component(name ${txtfile} NAME)
  message(STATUS "Copying ${txtfile} to ${name}")
  configure_file(${txtfile} ${CMAKE_CURRENT_BINARY_DIR}/${name} @ONLY)
endforeach()
