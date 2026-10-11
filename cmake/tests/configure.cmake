# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
cmake_minimum_required(VERSION 3.21)
function(configure expected)
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" -S "${SOURCE_DIR}" -B "${BINARY_DIR}/configure-test" -G
      "${GENERATOR}" -DLIBFABRIC_BUILD_TESTING=OFF ${ARGN}
    RESULT_VARIABLE _status
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
  )
  if(expected STREQUAL "success" AND NOT _status EQUAL 0)
    message(FATAL_ERROR "Configuration failed:\n${_stdout}\n${_stderr}")
  elseif(expected STREQUAL "failure" AND _status EQUAL 0)
    message(FATAL_ERROR "Invalid configuration was accepted: ${ARGN}")
  endif()
endfunction()
file(REMOVE_RECURSE "${BINARY_DIR}/configure-test")
# This must fail even on machines with RDMA development packages installed.
configure(failure -DLIBFABRIC_PROVIDER_VERBS=ON
          -DCMAKE_DISABLE_FIND_PACKAGE_IBVerbs=TRUE
)
configure(success -DLIBFABRIC_PROVIDER_VERBS=OFF -DLIBFABRIC_PROVIDER_TCP=ON)
configure(success -DLIBFABRIC_PROVIDER_TCP=OFF)
file(READ "${BINARY_DIR}/configure-test/config.h" _config)
if(NOT _config MATCHES "#define HAVE_TCP 0")
  message(
    FATAL_ERROR "Disabling TCP retained the previous feature configuration"
  )
endif()
configure(failure -DLIBFABRIC_PROVIDER_TCP=invalid)
configure(failure -DLIBFABRIC_PROVIDER_TCP=ON -DBUILD_SHARED_LIBS=OFF
          -DLIBFABRIC_TCP_DLOPEN=ON
)
