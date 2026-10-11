# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
cmake_minimum_required(VERSION 3.21)
function(run)
  execute_process(
    COMMAND ${ARGV}
    RESULT_VARIABLE _status
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
  )
  if(NOT _status EQUAL 0)
    message(FATAL_ERROR "${ARGV}\n${_stdout}\n${_stderr}")
  endif()
endfunction()
set(_root "${BINARY_DIR}/install-test/${CONFIG}")
file(REMOVE_RECURSE "${_root}")
run("${CMAKE_COMMAND}"
    --install
    "${BINARY_DIR}"
    --prefix
    "${_root}/stage"
    --config
    "${CONFIG}"
)
file(RENAME "${_root}/stage" "${_root}/relocated")
file(
  WRITE "${_root}/dependencies.cmake"
  "set(CMAKE_PREFIX_PATH [==[${_root}/relocated;${DEPENDENCY_PREFIX_PATH}]==] CACHE STRING \"\")\n"
)
foreach(_language C CXX)
  run("${CMAKE_COMMAND}"
      -S
      "${SOURCE_DIR}/cmake/tests/consumer"
      -B
      "${_root}/${_language}"
      -G
      "${GENERATOR}"
      "-DCONSUMER_LANGUAGE=${_language}"
      "-DCONSUMER_RUNTIME_DIR=${_root}/relocated/bin"
      "-DCMAKE_${_language}_COMPILER=${${_language}_COMPILER}"
      -C
      "${_root}/dependencies.cmake"
      "-DCMAKE_BUILD_TYPE=${CONFIG}"
  )
  run("${CMAKE_COMMAND}" --build "${_root}/${_language}" --config "${CONFIG}")
  run("${CMAKE_CTEST_COMMAND}" --test-dir "${_root}/${_language}" -C
      "${CONFIG}" --output-on-failure
  )
endforeach()

# Exercise the other installed package interface, including private static
# links.
find_program(_pkg_config NAMES pkg-config)
if(_pkg_config AND UNIX)
  file(GLOB_RECURSE _pc "${_root}/relocated/libfabric.pc")
  get_filename_component(_pc_dir "${_pc}" DIRECTORY)
  set(ENV{PKG_CONFIG_PATH} "${_pc_dir}:$ENV{PKG_CONFIG_PATH}")
  set(_static "")
  if(NOT SHARED)
    set(_static --static)
  endif()
  execute_process(
    COMMAND "${_pkg_config}" --cflags --libs ${_static} libfabric
    OUTPUT_VARIABLE _flags
    RESULT_VARIABLE _status
    ERROR_VARIABLE _error
  )
  if(NOT _status EQUAL 0)
    message(FATAL_ERROR "pkg-config failed: ${_error}")
  endif()
  separate_arguments(_flags UNIX_COMMAND "${_flags}")
  run("${C_COMPILER}" "${SOURCE_DIR}/cmake/tests/consumer/main.c" ${_flags}
      "-o" "${_root}/pkg-consumer"
  )
  get_filename_component(_lib_dir "${_pc_dir}" DIRECTORY)
  # Private SDKs need the same runtime search path as their link-time -L paths.
  foreach(_flag IN LISTS _flags)
    if(_flag MATCHES "^-L(.+)")
      string(APPEND _lib_dir ":${CMAKE_MATCH_1}")
    endif()
  endforeach()
  if(APPLE)
    set(_loader DYLD_LIBRARY_PATH)
  else()
    set(_loader LD_LIBRARY_PATH)
  endif()
  run("${CMAKE_COMMAND}" -E env "${_loader}=${_lib_dir}:$ENV{${_loader}}"
      "${_root}/pkg-consumer"
  )
endif()
