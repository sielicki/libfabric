# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
if(TARGET fi_info)
  add_test(NAME cmake.info COMMAND fi_info --version)
  if(HAVE_TCP)
    add_test(NAME cmake.tcp COMMAND fi_info -p tcp)
    set_tests_properties(
      cmake.tcp
      PROPERTIES ENVIRONMENT
                 "FI_PROVIDER_PATH=${libfabric_BINARY_DIR}/libfabric/$<CONFIG>"
    )
  endif()
endif()
if(LIBFABRIC_INSTALL AND NOT CMAKE_CROSSCOMPILING)
  if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    find_program(LIBFABRIC_TEST_CXX_COMPILER NAMES clang++ REQUIRED)
  elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    find_program(LIBFABRIC_TEST_CXX_COMPILER NAMES g++ REQUIRED)
  else()
    set(LIBFABRIC_TEST_CXX_COMPILER "${CMAKE_C_COMPILER}")
  endif()
  add_test(
    NAME cmake.install
    COMMAND
      "${CMAKE_COMMAND}" "-DSOURCE_DIR=${libfabric_SOURCE_DIR}"
      "-DBINARY_DIR=${libfabric_BINARY_DIR}" "-DGENERATOR=${CMAKE_GENERATOR}"
      "-DCONFIG=$<CONFIG>" "-DSHARED=${BUILD_SHARED_LIBS}"
      "-DC_COMPILER=${CMAKE_C_COMPILER}"
      "-DCXX_COMPILER=${LIBFABRIC_TEST_CXX_COMPILER}"
      "-DDEPENDENCY_PREFIX_PATH=${CMAKE_PREFIX_PATH}" -P
      "${CMAKE_CURRENT_SOURCE_DIR}/install.cmake"
  )
  set_tests_properties(cmake.install PROPERTIES TIMEOUT 120)
endif()
find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_FOUND
   AND TARGET fi_pingpong
   AND HAVE_TCP
)
  add_test(
    NAME cmake.loopback
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/loopback.py"
            "$<TARGET_FILE:fi_pingpong>"
  )
  set_tests_properties(
    cmake.loopback
    PROPERTIES TIMEOUT 45 ENVIRONMENT
               "FI_PROVIDER_PATH=${libfabric_BINARY_DIR}/libfabric/$<CONFIG>"
  )
endif()
if(NOT CMAKE_CROSSCOMPILING)
  add_test(
    NAME cmake.configure
    COMMAND
      "${CMAKE_COMMAND}" "-DSOURCE_DIR=${libfabric_SOURCE_DIR}"
      "-DBINARY_DIR=${libfabric_BINARY_DIR}" "-DGENERATOR=${CMAKE_GENERATOR}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/configure.cmake"
  )
  set_tests_properties(cmake.configure PROPERTIES TIMEOUT 120)
endif()
