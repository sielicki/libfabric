# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(_LIBNL3 QUIET libnl-3.0)
  pkg_check_modules(_LIBNL3_ROUTE QUIET libnl-route-3.0)
endif()
find_path(
  LibNL_INCLUDE_DIR
  NAMES netlink/netlink.h
  HINTS ${LibNL_ROOT} ENV LibNL_ROOT ${_LIBNL3_INCLUDE_DIRS}
  PATH_SUFFIXES include/libnl3 include
)
find_library(
  LibNL_LIBRARY
  NAMES nl-3 nl
  HINTS ${LibNL_ROOT} ENV LibNL_ROOT ${_LIBNL3_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64
)
set(LibNL_VERSION_MAJOR 1)
if(EXISTS "${LibNL_INCLUDE_DIR}/netlink/version.h")
  file(STRINGS "${LibNL_INCLUDE_DIR}/netlink/version.h" _version
       REGEX "^#define[ \t]+LIBNL_VER_MAJ[ \t]+[0-9]+"
  )
  if(_version MATCHES "LIBNL_VER_MAJ[ \t]+([0-9]+)")
    set(LibNL_VERSION_MAJOR "${CMAKE_MATCH_1}")
  endif()
endif()
set(_required LibNL_LIBRARY LibNL_INCLUDE_DIR)
set(LibNL_LIBRARIES "${LibNL_LIBRARY}")
set(LibNL_IS_V3 FALSE)
if(LibNL_VERSION_MAJOR EQUAL 3)
  set(LibNL_IS_V3 TRUE)
  find_library(
    LibNL_ROUTE_LIBRARY
    NAMES nl-route-3
    HINTS ${LibNL_ROOT} ENV LibNL_ROOT ${_LIBNL3_ROUTE_LIBRARY_DIRS}
    PATH_SUFFIXES lib lib64
  )
  list(APPEND _required LibNL_ROUTE_LIBRARY)
  list(APPEND LibNL_LIBRARIES "${LibNL_ROUTE_LIBRARY}")
else()
  list(APPEND LibNL_LIBRARIES m)
endif()
find_package_handle_standard_args(
  LibNL
  REQUIRED_VARS ${_required}
  VERSION_VAR LibNL_VERSION_MAJOR
)
if(LibNL_FOUND)
  set(LibNL_INCLUDE_DIRS "${LibNL_INCLUDE_DIR}")
  if(NOT TARGET LibNL::LibNL)
    add_library(LibNL::LibNL INTERFACE IMPORTED)
    set_target_properties(
      LibNL::LibNL
      PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${LibNL_INCLUDE_DIR}"
                 INTERFACE_LINK_LIBRARIES "${LibNL_LIBRARIES}"
    )
  endif()
endif()
mark_as_advanced(LibNL_INCLUDE_DIR LibNL_LIBRARY LibNL_ROUTE_LIBRARY)
