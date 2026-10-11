# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
install(
  TARGETS fabric
  EXPORT LibfabricTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(
  DIRECTORY include/rdma
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.h"
)

install(FILES ${LIBFABRIC_PROVIDER_HEADERS}
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rdma"
)
if(WIN32)
  install(FILES include/windows/sys/uio.h
          DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/sys"
  )
  install(FILES include/windows/poll.h
          DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  )
endif()

# A shared-library consumer needs only the public headers. Static consumers must
# rediscover the private libraries without embedding build-machine paths.
get_target_property(_links fabric LINK_LIBRARIES)
set(LIBFABRIC_DEPENDENCIES "")
foreach(_link IN LISTS _links)
  if(_link MATCHES "^([A-Za-z0-9]+)::")
    set(_package "${CMAKE_MATCH_1}")
    if(_package STREQUAL "CUDA")
      set(_package CUDAToolkit)
    endif()
    list(APPEND LIBFABRIC_DEPENDENCIES "${_package}")
  endif()
endforeach()
list(REMOVE_DUPLICATES LIBFABRIC_DEPENDENCIES)
set(_config_dir "${CMAKE_INSTALL_LIBDIR}/cmake/Libfabric")
configure_package_config_file(
  cmake/LibfabricConfig.cmake.in "${libfabric_BINARY_DIR}/LibfabricConfig.cmake"
  INSTALL_DESTINATION "${_config_dir}"
)
write_basic_package_version_file(
  "${libfabric_BINARY_DIR}/LibfabricConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion
)
install(
  EXPORT LibfabricTargets
  NAMESPACE Libfabric::
  DESTINATION "${_config_dir}"
)
install(FILES "${libfabric_BINARY_DIR}/LibfabricConfig.cmake"
              "${libfabric_BINARY_DIR}/LibfabricConfigVersion.cmake"
        DESTINATION "${_config_dir}"
)
# Only install modules needed by the exported static dependency graph.
foreach(_package IN LISTS LIBFABRIC_DEPENDENCIES)
  if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/Find${_package}.cmake")
    install(FILES "${CMAKE_CURRENT_LIST_DIR}/Find${_package}.cmake"
            DESTINATION "${_config_dir}/modules"
    )
  endif()
endforeach()

# Flatten imported dependency targets for pkg-config's static link flags.
function(libfabric_pkgconfig_links output)
  set(_flags "")
  foreach(_link IN LISTS ARGN)
    if(_link MATCHES "^\\$<|^::@")
      continue()
    elseif(TARGET "${_link}")
      get_target_property(_location "${_link}" IMPORTED_LOCATION)
      if(_location)
        libfabric_pkgconfig_links(_location_flags "${_location}")
        list(APPEND _flags ${_location_flags})
      endif()
      get_target_property(_interface "${_link}" INTERFACE_LINK_LIBRARIES)
      if(_interface)
        libfabric_pkgconfig_links(_interface_flags ${_interface})
        list(APPEND _flags ${_interface_flags})
      endif()
      get_target_property(_options "${_link}" INTERFACE_LINK_OPTIONS)
      if(_options)
        list(APPEND _flags ${_options})
      endif()
    elseif(IS_ABSOLUTE "${_link}")
      get_filename_component(_dir "${_link}" DIRECTORY)
      get_filename_component(_name "${_link}" NAME)
      if(_name MATCHES "^lib(.+)\\.(so(\\.[0-9.]+)?|a|dylib)$")
        list(APPEND _flags "-L${_dir}" "-l${CMAKE_MATCH_1}")
      else()
        list(APPEND _flags "${_link}")
      endif()
    elseif(_link MATCHES "^-")
      list(APPEND _flags "${_link}")
    else()
      list(APPEND _flags "-l${_link}")
    endif()
  endforeach()
  set(${output}
      "${_flags}"
      PARENT_SCOPE
  )
endfunction()
libfabric_pkgconfig_links(_private_links ${_links})
list(JOIN _private_links " " LIBS_PRIVATE)
# pcfiledir makes a staged installation relocatable for relative install dirs.
file(RELATIVE_PATH _pc_prefix "${CMAKE_INSTALL_FULL_LIBDIR}/pkgconfig"
     "${CMAKE_INSTALL_PREFIX}"
)
set(prefix "\${pcfiledir}/${_pc_prefix}")
set(exec_prefix "\${prefix}")
foreach(_dir lib include)
  if(_dir STREQUAL "lib")
    set(_install_dir "${CMAKE_INSTALL_LIBDIR}")
  else()
    set(_install_dir "${CMAKE_INSTALL_INCLUDEDIR}")
  endif()
  if(IS_ABSOLUTE "${_install_dir}")
    set(${_dir}dir "${_install_dir}")
  else()
    set(${_dir}dir "\${prefix}/${_install_dir}")
  endif()
endforeach()
set(PACKAGE_VERSION "${LIBFABRIC_VERSION_FULL}")
set(LIBFABRIC_PKGCONFIG_LINK_OPTIONS "")
if(LIBFABRIC_SANITIZER_LINK_OPTIONS)
  list(JOIN LIBFABRIC_SANITIZER_LINK_OPTIONS " "
       LIBFABRIC_PKGCONFIG_LINK_OPTIONS
  )
endif()
configure_file(cmake/libfabric.pc.in libfabric.pc @ONLY)
install(FILES "${libfabric_BINARY_DIR}/libfabric.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
)

# Documentation is shipped pre-generated; source compilation needs no renderer.
foreach(_section 1 3 7)
  file(GLOB _pages "${libfabric_SOURCE_DIR}/man/man${_section}/*.${_section}")
  install(FILES ${_pages} DESTINATION "${CMAKE_INSTALL_MANDIR}/man${_section}")
endforeach()
