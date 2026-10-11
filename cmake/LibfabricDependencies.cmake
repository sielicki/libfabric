# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only Do not inherit
# availability for backends the caller has disabled.
foreach(
  _backend
  CUDAToolkit
  GDRCopy
  LevelZero
  ROCm
  Neuron
  SynapseAI
  XPMEM
  Kdreg2
  UFD
  LibUring
  LTTng
  Valgrind
)
  set(${_backend}_FOUND FALSE)
endforeach()
if(LIBFABRIC_ENABLE_GDRCOPY AND NOT LIBFABRIC_ENABLE_CUDA)
  message(FATAL_ERROR "GDRCopy requires LIBFABRIC_ENABLE_CUDA=ON")
endif()

# Required
find_package(Threads REQUIRED)
find_package(Atomics REQUIRED)

# Platform-specific required
if(NOT WIN32)
  find_package(DL REQUIRED)
  find_package(RT)
endif()

# Optional features (auto-detected)
find_package(Epoll)
find_package(Kqueue)
find_package(Ethtool)
find_package(CMA)

# Userfaultfd
if(LIBFABRIC_UFFD_MONITOR)
  find_package(UFD)
endif()

# XPMEM
if(LIBFABRIC_ENABLE_XPMEM)
  find_package(XPMEM)
endif()

# kdreg2
if(LIBFABRIC_KDREG2_MONITOR)
  find_package(Kdreg2)
endif()

# io_uring
if(LIBFABRIC_ENABLE_LIBURING AND LIBFABRIC_LINUX)
  find_package(LibUring)
  if(LibUring_FOUND)
    cmake_push_check_state()
    set(CMAKE_REQUIRED_LIBRARIES LibUring::LibUring)
    check_c_source_compiles(
      "
      #include <liburing.h>
      int main(void) {
        struct io_uring_sqe sqe = {0};
        io_uring_prep_poll_multishot(&sqe, 0, 0);
        return IORING_CQE_F_MORE;
      }"
      LIBFABRIC_LIBURING_MULTISHOT
    )
    cmake_pop_check_state()
    if(NOT LIBFABRIC_LIBURING_MULTISHOT)
      set(LibUring_FOUND FALSE)
    endif()
  endif()
endif()

# LTTng
if(LIBFABRIC_ENABLE_LTTNG)
  find_package(LTTng REQUIRED)
endif()

# Valgrind
if(LIBFABRIC_ENABLE_VALGRIND)
  find_package(Valgrind REQUIRED)
endif()

# GPU/Accelerator support
if(LIBFABRIC_ENABLE_CUDA)
  if(LIBFABRIC_CUDA_DLOPEN)
    # No CUDA compiler or runtime library is required for runtime loading.
    find_path(
      LIBFABRIC_CUDA_INCLUDE_DIR cuda.h
      HINTS ${CUDAToolkit_ROOT} ENV CUDA_PATH
      PATH_SUFFIXES include REQUIRED
    )
    find_path(
      LIBFABRIC_CUDART_INCLUDE_DIR cuda_runtime.h
      HINTS ${LIBFABRIC_CUDA_INCLUDE_DIR} ${CUDAToolkit_ROOT} ENV CUDA_PATH
      PATH_SUFFIXES include REQUIRED
    )
    find_path(
      LIBFABRIC_NVML_INCLUDE_DIR nvml.h
      HINTS ${LIBFABRIC_CUDA_INCLUDE_DIR} ${CUDAToolkit_ROOT} ENV CUDA_PATH
      PATH_SUFFIXES include REQUIRED
    )
    set(CUDAToolkit_INCLUDE_DIRS
        ${LIBFABRIC_CUDA_INCLUDE_DIR} ${LIBFABRIC_CUDART_INCLUDE_DIR}
        ${LIBFABRIC_NVML_INCLUDE_DIR}
    )
    set(CUDAToolkit_FOUND TRUE)
  else()
    find_package(CUDAToolkit REQUIRED)
    if(NOT TARGET CUDA::nvml)
      message(FATAL_ERROR "CUDA support requires the NVML development library")
    endif()
  endif()
  if(LIBFABRIC_ENABLE_GDRCOPY AND CUDAToolkit_FOUND)
    find_package(GDRCopy REQUIRED)
  endif()
endif()

if(LIBFABRIC_ENABLE_ZE)
  find_package(LevelZero REQUIRED)
endif()

if(LIBFABRIC_ENABLE_ROCR)
  find_package(ROCm REQUIRED)
  cmake_push_check_state()
  set(CMAKE_REQUIRED_LIBRARIES ROCm::ROCm)
  check_c_source_compiles(
    "
    #include <hsa/hsa_ext_amd.h>
    int main(void) {
      (void) &hsa_amd_portable_export_dmabuf;
      return HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED;
    }"
    ROCm_HAS_DMABUF
  )
  cmake_pop_check_state()
endif()

if(LIBFABRIC_ENABLE_NEURON)
  find_package(Neuron REQUIRED)
endif()

if(LIBFABRIC_ENABLE_SYNAPSEAI)
  find_package(SynapseAI REQUIRED)
endif()

# Search only for dependencies used by enabled provider candidates.
if(LIBFABRIC_PROVIDER_VERBS
   OR LIBFABRIC_PROVIDER_EFA
   OR LIBFABRIC_PROVIDER_USNIC
   OR (LIBFABRIC_PROVIDER_PSM3 AND LIBFABRIC_PSM3_VERBS)
)
  find_package(IBVerbs)
  include(CheckIBVerbs)
endif()
if(LIBFABRIC_PROVIDER_VERBS)
  find_package(RdmaCM)
  include(CheckRdmaCM)
endif()
if(LIBFABRIC_PROVIDER_EFA)
  find_package(Efadv)
  include(CheckEfadv)
endif()
if(LIBFABRIC_PROVIDER_OPX
   OR LIBFABRIC_PROVIDER_PSM3
   OR LIBFABRIC_PROVIDER_SHM
)
  find_package(Numa)
  find_package(UUID)
  find_package(DSA)
endif()
if(LIBFABRIC_PROVIDER_PSM3 AND LIBFABRIC_PSM3_HWLOC)
  find_package(Hwloc)
endif()
if(LIBFABRIC_PROVIDER_PSM2)
  find_package(PSM2)
endif()
if(LIBFABRIC_PROVIDER_UCX)
  find_package(UCX)
  if(UCX_FOUND)
    # =============================================================================
    # Feature Detection
    # =============================================================================
    include(CheckCSourceCompiles)
    include(CMakePushCheckState)

    cmake_push_check_state()
    set(CMAKE_REQUIRED_INCLUDES ${UCX_INCLUDE_DIRS})

    # Check for UCP_WORKER_FLAG_IGNORE_REQUEST_LEAK
    check_c_source_compiles(
      "
    #include <ucp/api/ucp.h>
    int main() {
      int x = UCP_WORKER_FLAG_IGNORE_REQUEST_LEAK;
      return x;
    }
  "
      UCX_HAVE_WORKER_FLAG_IGNORE_REQUEST_LEAK
    )

    set(CMAKE_REQUIRED_LIBRARIES UCX::UCX)
    check_c_source_compiles(
      "
      #include <ucs/memory/memtype_cache.h>
      int main(void) {
        ucs_memtype_cache_update(0, 0, UCS_MEMORY_TYPE_UNKNOWN, UCS_SYS_DEVICE_ID_UNKNOWN);
        return 0;
      }"
      LIBFABRIC_UCX_MEMTYPE_API
    )
    if(NOT LIBFABRIC_UCX_MEMTYPE_API)
      message(
        STATUS "UCX provider requires the four-argument memtype cache API"
      )
      set(UCX_FOUND FALSE)
    endif()
    cmake_pop_check_state()
  endif()
endif()
if(LIBFABRIC_PROVIDER_CXI)
  find_package(LibCxi)
  find_package(CXIDriverHeaders)
  if(LibCxi_FOUND AND CXIDriverHeaders_FOUND)
    cmake_push_check_state()
    set(CMAKE_REQUIRED_LIBRARIES LibCxi::LibCxi
                                 CXIDriverHeaders::CXIDriverHeaders
    )
    foreach(_feature MODIFY_CP SVC_GET_VNI_RANGE ALLOC_TRIG_CP SVC_GET_NETNS)
      string(TOLOWER "${_feature}" _symbol)
      check_symbol_exists(
        cxil_${_symbol} "string.h;libcxi/libcxi.h" LibCxi_HAVE_${_feature}
      )
    endforeach()
    cmake_pop_check_state()
  endif()
  find_package(JsonC)
  find_package(LibCurl)
endif()
if(LIBFABRIC_PROVIDER_USNIC)
  find_package(LibNL)
endif()
