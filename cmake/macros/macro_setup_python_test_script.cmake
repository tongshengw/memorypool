# A small macro used for setting up Python test scripts.

string(TOLOWER ${CMAKE_BUILD_TYPE} buildl)
string(TOUPPER ${CMAKE_BUILD_TYPE} buildu)
string(TOUPPER ${PROJECT_NAME} projectu)

macro(setup_python_test_script namel)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/${namel}.py
    ${CMAKE_CURRENT_BINARY_DIR}/${namel}.py
    COPYONLY
  )
endmacro()