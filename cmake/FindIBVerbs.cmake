# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)

# Try pkg-config first (preferred for rdma-core)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(_IBVERBS QUIET IMPORTED_TARGET libibverbs)
endif()

if(_IBVERBS_FOUND)
  set(IBVerbs_FOUND TRUE)
  set(IBVerbs_VERSION "${_IBVERBS_VERSION}")
  set(IBVerbs_INCLUDE_DIRS "${_IBVERBS_INCLUDE_DIRS}")
  set(IBVerbs_LIBRARIES "${_IBVERBS_LIBRARIES}")
  set(IBVerbs_LIBRARY_DIRS "${_IBVERBS_LIBRARY_DIRS}")

  # Create our aliased target from the pkg-config IMPORTED_TARGET
  if(NOT TARGET IBVerbs::IBVerbs)
    add_library(IBVerbs::IBVerbs ALIAS PkgConfig::_IBVERBS)
  endif()
else()
  # Fallback: manual search
  find_path(
    IBVerbs_INCLUDE_DIR
    NAMES infiniband/verbs.h
    HINTS ${IBVerbs_ROOT} ENV IBVerbs_ROOT
    PATH_SUFFIXES include
  )

  find_library(
    IBVerbs_LIBRARY
    NAMES ibverbs
    HINTS ${IBVerbs_ROOT} ENV IBVerbs_ROOT
    PATH_SUFFIXES lib lib64
  )

  find_package_handle_standard_args(
    IBVerbs REQUIRED_VARS IBVerbs_LIBRARY IBVerbs_INCLUDE_DIR
  )

  if(IBVerbs_FOUND)
    set(IBVerbs_INCLUDE_DIRS "${IBVerbs_INCLUDE_DIR}")
    set(IBVerbs_LIBRARIES "${IBVerbs_LIBRARY}")

    if(NOT TARGET IBVerbs::IBVerbs)
      add_library(IBVerbs::IBVerbs UNKNOWN IMPORTED)
      set_target_properties(
        IBVerbs::IBVerbs
        PROPERTIES IMPORTED_LOCATION "${IBVerbs_LIBRARY}"
                   INTERFACE_INCLUDE_DIRECTORIES "${IBVerbs_INCLUDE_DIR}"
      )
    endif()
  endif()
endif()

mark_as_advanced(IBVerbs_INCLUDE_DIR IBVerbs_LIBRARY)
