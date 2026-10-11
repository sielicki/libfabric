# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(CheckCSourceCompiles)
include(CheckSymbolExists)
include(CheckCCompilerFlag)
include(CMakePushCheckState)

function(libfabric_check_c11_atomic_least_types RESULT_VAR)
  check_c_source_compiles(
    "
    #include <stdatomic.h>
    int main() {
      atomic_int_least32_t a;
      atomic_int_least64_t b;
      return 0;
    }
  "
    ${RESULT_VAR}
  )
  set(${RESULT_VAR}
      ${${RESULT_VAR}}
      PARENT_SCOPE
  )
endfunction()

function(libfabric_check_int128_atomics RESULT_VAR)
  check_c_source_compiles(
    "
    #include <stdint.h>
    int main() {
      __int128 d = 0;
      __int128 s = 1;
      __int128 c = 0;
      __int128 r;
      r = __atomic_fetch_add(&d, s, __ATOMIC_SEQ_CST);
      __atomic_load(&d, &r, __ATOMIC_SEQ_CST);
      __atomic_exchange(&d, &s, &r, __ATOMIC_SEQ_CST);
      __atomic_compare_exchange(&d, &c, &s, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
      return !__atomic_is_lock_free(sizeof(d), 0);
    }
  "
    ${RESULT_VAR}
  )
  set(${RESULT_VAR}
      ${${RESULT_VAR}}
      PARENT_SCOPE
  )
endfunction()

function(libfabric_check_cpuid RESULT_VAR)
  check_c_source_compiles(
    "
    #include <stddef.h>
    #include <cpuid.h>
    int main() {
      int a, b, c, d;
      __cpuid_count(0, 0, a, b, c, d);
      return 0;
    }
  "
    ${RESULT_VAR}
  )
  set(${RESULT_VAR}
      ${${RESULT_VAR}}
      PARENT_SCOPE
  )
endfunction()

function(libfabric_check_symver_support RESULT_VAR)
  check_c_source_compiles(
    "
    __asm__(\".symver main_, main@ABIVER_1.0\");
    int main() { return 0; }
  "
    ${RESULT_VAR}
  )
  set(${RESULT_VAR}
      ${${RESULT_VAR}}
      PARENT_SCOPE
  )
endfunction()

function(libfabric_check_alias_attribute RESULT_VAR)
  check_c_source_compiles(
    "
    int foo(int arg);
    int foo(int arg) { return arg + 3; }
    int foo2(int arg) __attribute__ (( __alias__(\"foo\")));
    int main() { foo2(1); return 0; }
  "
    ${RESULT_VAR}
  )
  set(${RESULT_VAR}
      ${${RESULT_VAR}}
      PARENT_SCOPE
  )
endfunction()

function(libfabric_check_linux_perf_rdpmc RESULT_VAR)
  check_c_source_compiles(
    "
    #include <linux/perf_event.h>
    int main() {
      __builtin_ia32_rdpmc(0);
      return 0;
    }
  "
    ${RESULT_VAR}
  )
  set(${RESULT_VAR}
      ${${RESULT_VAR}}
      PARENT_SCOPE
  )
endfunction()

function(libfabric_check_pthread_spin RESULT_VAR)
  cmake_push_check_state()
  set(CMAKE_REQUIRED_LIBRARIES Threads::Threads)
  check_symbol_exists(pthread_spin_init "pthread.h" ${RESULT_VAR})
  cmake_pop_check_state()
  set(${RESULT_VAR}
      ${${RESULT_VAR}}
      PARENT_SCOPE
  )
endfunction()

function(libfabric_check_getifaddrs RESULT_VAR)
  check_symbol_exists(getifaddrs "sys/types.h;ifaddrs.h" _have_getifaddrs)
  if(_have_getifaddrs)
    set(${RESULT_VAR}
        1
        PARENT_SCOPE
    )
  else()
    set(${RESULT_VAR}
        0
        PARENT_SCOPE
    )
  endif()
endfunction()

# Apply private requirements without exporting internal build targets.
function(libfabric_configure_target target)
  target_link_libraries(
    ${target} PRIVATE $<BUILD_INTERFACE:libfabric_build_settings>
  )
  set_target_properties(
    ${target}
    PROPERTIES POSITION_INDEPENDENT_CODE ON
               C_VISIBILITY_PRESET hidden
               C_STANDARD 11
               C_STANDARD_REQUIRED ON
  )
  if(LIBFABRIC_ENABLE_LTO)
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
  endif()
endfunction()

function(libfabric_configure_instrumentation target)
  if(LIBFABRIC_ENABLE_ASAN AND LIBFABRIC_ENABLE_TSAN)
    message(
      FATAL_ERROR "AddressSanitizer and ThreadSanitizer cannot be combined"
    )
  endif()
  foreach(_kind address thread undefined leak)
    if((_kind STREQUAL "address" AND LIBFABRIC_ENABLE_ASAN)
       OR (_kind STREQUAL "thread" AND LIBFABRIC_ENABLE_TSAN)
       OR (_kind STREQUAL "undefined" AND LIBFABRIC_ENABLE_UBSAN)
       OR (_kind STREQUAL "leak" AND LIBFABRIC_ENABLE_LSAN)
    )
      cmake_push_check_state()
      set(CMAKE_REQUIRED_LIBRARIES "-fsanitize=${_kind}")
      check_c_compiler_flag(
        "-fsanitize=${_kind}" LIBFABRIC_HAS_${_kind}_SANITIZER
      )
      cmake_pop_check_state()
      if(NOT LIBFABRIC_HAS_${_kind}_SANITIZER)
        message(FATAL_ERROR "The compiler does not support -fsanitize=${_kind}")
      endif()
      target_compile_options(
        ${target} INTERFACE "-fsanitize=${_kind}" -fno-omit-frame-pointer
      )
      target_link_options(${target} INTERFACE "-fsanitize=${_kind}")
    endif()
  endforeach()
  if(LIBFABRIC_PICKY AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
      ${target}
      INTERFACE -Wall
                -Wextra
                -Wundef
                -Wpointer-arith
                -Wno-unused-parameter
                -Wno-sign-compare
                -Wno-missing-field-initializers
    )
  endif()
endfunction()
