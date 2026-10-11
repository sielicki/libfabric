# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
cmake_minimum_required(VERSION 3.21)
project(libfabric_parent LANGUAGES C)
set(BUILD_SHARED_LIBS OFF)
set(LIBFABRIC_INSTALL
    OFF
    CACHE BOOL "" FORCE
)
set(LIBFABRIC_BUILD_TESTING
    OFF
    CACHE BOOL "" FORCE
)
set(LIBFABRIC_BUILD_UTILITIES
    OFF
    CACHE BOOL "" FORCE
)
add_subdirectory("${LIBFABRIC_SOURCE}" libfabric)
add_executable(parent ../consumer/main.c)
target_link_libraries(parent PRIVATE Libfabric::fabric)
enable_testing()
add_test(NAME parent.consume COMMAND parent)
