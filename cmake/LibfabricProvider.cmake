# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

# Each provider owns an object target so its private includes, ISA flags, and
# feature macros do not leak into the core or into other providers.
function(libfabric_add_provider)
  cmake_parse_arguments(
    P
    "LINUX_ONLY;FREEBSD_ONLY;MACOS_ONLY;X86_64_ONLY;AARCH64_ONLY"
    "NAME"
    "SOURCES;HEADERS;INCLUDE_DIRS;COMPILE_DEFINITIONS;COMPILE_OPTIONS;LINK_LIBRARIES;DEPENDS;CONDITION"
    ${ARGN}
  )
  if(P_UNPARSED_ARGUMENTS
     OR NOT P_NAME
     OR NOT P_SOURCES
  )
    message(FATAL_ERROR "Invalid provider definition: ${ARGN}")
  endif()
  string(TOUPPER "${P_NAME}" _name)
  foreach(_platform LINUX FREEBSD MACOS X86_64 AARCH64)
    if(P_${_platform}_ONLY AND NOT LIBFABRIC_${_platform})
      message(STATUS "Provider ${P_NAME}: requires ${_platform}")
      return()
    endif()
  endforeach()
  foreach(_dependency IN LISTS P_DEPENDS)
    if(NOT ${_dependency}_FOUND)
      message(STATUS "Provider ${P_NAME}: missing ${_dependency}")
      return()
    endif()
  endforeach()
  if(P_CONDITION)
    if(NOT (${P_CONDITION}))
      return()
    endif()
  endif()

  set(_target libfabric_provider_${P_NAME})
  if(LIBFABRIC_${_name}_DLOPEN)
    add_library(
      ${_target} MODULE ${P_SOURCES} ${P_HEADERS}
                        $<TARGET_OBJECTS:libfabric_common>
    )
    target_link_libraries(${_target} PRIVATE fabric ${LIBFABRIC_CORE_LIBRARIES})
    set_target_properties(
      ${_target}
      PROPERTIES OUTPUT_NAME "${P_NAME}-fi"
                 PREFIX "lib"
                 LIBRARY_OUTPUT_DIRECTORY
                 "${libfabric_BINARY_DIR}/libfabric/$<CONFIG>"
                 LIBFABRIC_PLUGIN TRUE
    )
    if(LIBFABRIC_LINUX AND NOT LIBFABRIC_SANITIZER_LINK_OPTIONS)
      target_link_options(${_target} PRIVATE "LINKER:--no-undefined")
    endif()
    if(LIBFABRIC_INSTALL)
      install(TARGETS ${_target}
              LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/libfabric"
      )
    endif()
  else()
    add_library(${_target} OBJECT ${P_SOURCES} ${P_HEADERS})
    target_sources(fabric PRIVATE $<TARGET_OBJECTS:${_target}>)
    target_link_libraries(fabric PRIVATE ${P_LINK_LIBRARIES})
  endif()
  libfabric_configure_target(${_target})
  target_include_directories(${_target} PRIVATE ${P_INCLUDE_DIRS})
  target_compile_definitions(${_target} PRIVATE ${P_COMPILE_DEFINITIONS})
  target_compile_options(${_target} PRIVATE ${P_COMPILE_OPTIONS})
  target_link_libraries(${_target} PRIVATE ${P_LINK_LIBRARIES})
  set(HAVE_${_name}
      TRUE
      PARENT_SCOPE
  )
endfunction()

function(libfabric_add_hook_provider)
  cmake_parse_arguments(
    H
    "LINUX_ONLY"
    "NAME"
    "SOURCES;HEADERS;INCLUDE_DIRS;COMPILE_DEFINITIONS;LINK_LIBRARIES;DEPENDS;CONDITION"
    ${ARGN}
  )
  string(TOUPPER "${H_NAME}" _name)
  if(NOT LIBFABRIC_HOOK_${_name})
    return()
  endif()
  libfabric_add_provider(${ARGN})
  if(TARGET libfabric_provider_${H_NAME})
    target_include_directories(
      libfabric_build_settings INTERFACE ${H_INCLUDE_DIRS}
    )
  endif()
endfunction()
