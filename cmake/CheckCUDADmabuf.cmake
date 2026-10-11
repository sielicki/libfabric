# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(CheckCSourceCompiles)
include(CMakePushCheckState)
cmake_push_check_state()
set(CMAKE_REQUIRED_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})
# Compile only: runtime loading needs declarations, not link-time CUDA symbols.
set(_saved_try_compile_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
check_c_source_compiles(
  "
  #include <cuda.h>
  int main(void) {
    (void)&cuMemGetHandleForAddressRange;
    return CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED + CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD;
  }"
  CUDAToolkit_HAS_DMABUF
)
check_c_source_compiles(
  "
  #include <cuda.h>
  int main(void) {
    (void)&cuCtxGetFlags;
    (void)&cuCtxSetFlags;
    return CU_CTX_SYNC_MEMOPS;
  }"
  CUDAToolkit_HAS_CTX_SYNC_MEMOPS
)
set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_saved_try_compile_type}")
cmake_pop_check_state()
