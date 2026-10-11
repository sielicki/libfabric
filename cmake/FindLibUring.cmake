# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(_LIBURING QUIET IMPORTED_TARGET liburing>=2.1)
endif()
if(_LIBURING_FOUND)
  set(LibUring_FOUND TRUE)
  set(LibUring_VERSION "${_LIBURING_VERSION}")
  set(LibUring_INCLUDE_DIRS "${_LIBURING_INCLUDE_DIRS}")
  if(NOT TARGET LibUring::LibUring)
    add_library(LibUring::LibUring ALIAS PkgConfig::_LIBURING)
  endif()
else()
  find_path(LibUring_INCLUDE_DIR liburing.h PATH_SUFFIXES include)
  find_library(LibUring_LIBRARY uring PATH_SUFFIXES lib lib64)
  find_package_handle_standard_args(
    LibUring REQUIRED_VARS LibUring_INCLUDE_DIR LibUring_LIBRARY
  )
  if(LibUring_FOUND)
    set(LibUring_INCLUDE_DIRS "${LibUring_INCLUDE_DIR}")
    if(NOT TARGET LibUring::LibUring)
      add_library(LibUring::LibUring UNKNOWN IMPORTED)
      set_target_properties(
        LibUring::LibUring
        PROPERTIES IMPORTED_LOCATION "${LibUring_LIBRARY}"
                   INTERFACE_INCLUDE_DIRECTORIES "${LibUring_INCLUDE_DIR}"
      )
    endif()
  endif()
endif()
mark_as_advanced(LibUring_INCLUDE_DIR LibUring_LIBRARY)
