# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(CheckCSourceCompiles)
include(CheckSymbolExists)
include(CMakePushCheckState)

if(RdmaCM_FOUND)
  include(CMakePushCheckState)
  cmake_push_check_state()
  set(CMAKE_REQUIRED_INCLUDES ${RdmaCM_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES RdmaCM::RdmaCM)

  # Check for rdma_establish (modern rdma-core)
  check_symbol_exists(
    rdma_establish "rdma/rdma_cma.h" _RdmaCM_HAVE_RDMA_ESTABLISH
  )
  if(_RdmaCM_HAVE_RDMA_ESTABLISH)
    set(RdmaCM_HAVE_RDMA_ESTABLISH TRUE)
  else()
    set(RdmaCM_HAVE_RDMA_ESTABLISH FALSE)
  endif()

  # Check for rdma_create_qp_ex (required for XRC support)
  check_symbol_exists(
    rdma_create_qp_ex "rdma/rdma_cma.h" _RdmaCM_HAVE_CREATE_QP_EX
  )
  if(_RdmaCM_HAVE_CREATE_QP_EX)
    set(RdmaCM_HAVE_CREATE_QP_EX TRUE)
  else()
    set(RdmaCM_HAVE_CREATE_QP_EX FALSE)
  endif()

  cmake_pop_check_state()
endif()
