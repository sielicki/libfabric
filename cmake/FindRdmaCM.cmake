# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(_RDMACM QUIET IMPORTED_TARGET librdmacm)
endif()

if(_RDMACM_FOUND)
  set(RdmaCM_FOUND TRUE)
  set(RdmaCM_VERSION "${_RDMACM_VERSION}")
  set(RdmaCM_INCLUDE_DIRS "${_RDMACM_INCLUDE_DIRS}")
  set(RdmaCM_LIBRARIES "${_RDMACM_LIBRARIES}")
  set(RdmaCM_LIBRARY_DIRS "${_RDMACM_LIBRARY_DIRS}")

  # Create our aliased target from the pkg-config IMPORTED_TARGET
  if(NOT TARGET RdmaCM::RdmaCM)
    add_library(RdmaCM::RdmaCM ALIAS PkgConfig::_RDMACM)
  endif()
else()
  find_path(
    RdmaCM_INCLUDE_DIR
    NAMES rdma/rdma_cma.h
    HINTS ${RdmaCM_ROOT} ENV RdmaCM_ROOT
    PATH_SUFFIXES include
  )

  find_library(
    RdmaCM_LIBRARY
    NAMES rdmacm
    HINTS ${RdmaCM_ROOT} ENV RdmaCM_ROOT
    PATH_SUFFIXES lib lib64
  )

  find_package_handle_standard_args(
    RdmaCM REQUIRED_VARS RdmaCM_LIBRARY RdmaCM_INCLUDE_DIR
  )

  if(RdmaCM_FOUND)
    set(RdmaCM_INCLUDE_DIRS "${RdmaCM_INCLUDE_DIR}")
    set(RdmaCM_LIBRARIES "${RdmaCM_LIBRARY}")

    if(NOT TARGET RdmaCM::RdmaCM)
      add_library(RdmaCM::RdmaCM UNKNOWN IMPORTED)
      set_target_properties(
        RdmaCM::RdmaCM
        PROPERTIES IMPORTED_LOCATION "${RdmaCM_LIBRARY}"
                   INTERFACE_INCLUDE_DIRECTORIES "${RdmaCM_INCLUDE_DIR}"
      )
    endif()
  endif()
endif()

mark_as_advanced(RdmaCM_INCLUDE_DIR RdmaCM_LIBRARY)
