# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(_EFADV QUIET IMPORTED_TARGET libefa)
endif()

if(_EFADV_FOUND)
  set(Efadv_FOUND TRUE)
  set(Efadv_VERSION "${_EFADV_VERSION}")
  set(Efadv_INCLUDE_DIRS "${_EFADV_INCLUDE_DIRS}")
  set(Efadv_LIBRARIES "${_EFADV_LIBRARIES}")

  # Create our aliased target from the pkg-config IMPORTED_TARGET
  if(NOT TARGET Efadv::Efadv)
    add_library(Efadv::Efadv ALIAS PkgConfig::_EFADV)
  endif()
else()
  find_path(
    Efadv_INCLUDE_DIR
    NAMES infiniband/efadv.h
    HINTS ${Efadv_ROOT} ENV Efadv_ROOT
    PATH_SUFFIXES include
  )

  find_library(
    Efadv_LIBRARY
    NAMES efa
    HINTS ${Efadv_ROOT} ENV Efadv_ROOT
    PATH_SUFFIXES lib lib64
  )

  find_package_handle_standard_args(
    Efadv REQUIRED_VARS Efadv_LIBRARY Efadv_INCLUDE_DIR
  )

  if(Efadv_FOUND)
    set(Efadv_INCLUDE_DIRS "${Efadv_INCLUDE_DIR}")
    set(Efadv_LIBRARIES "${Efadv_LIBRARY}")

    if(NOT TARGET Efadv::Efadv)
      add_library(Efadv::Efadv UNKNOWN IMPORTED)
      set_target_properties(
        Efadv::Efadv
        PROPERTIES IMPORTED_LOCATION "${Efadv_LIBRARY}"
                   INTERFACE_INCLUDE_DIRECTORIES "${Efadv_INCLUDE_DIR}"
      )
    endif()
  endif()
endif()

mark_as_advanced(Efadv_INCLUDE_DIR Efadv_LIBRARY)
