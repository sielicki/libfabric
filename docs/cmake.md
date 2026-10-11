# Building with CMake

CMake 3.21 or newer and a C11 compiler are required. A Git checkout can be
built directly; `autogen.sh`, Autoconf, Automake, and Libtool are not used.
Ninja and Makefile generators are supported. For example:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --install build
```

The default builds a shared library, utilities, and available providers.
Use `-DBUILD_SHARED_LIBS=OFF` for a static library. Both configurations export
one CMake target, `Libfabric::fabric`, and install `libfabric.pc`. The library
version and SONAME follow the existing release and ABI version definitions.

The `debug`, `release`, `static`, `minimal`, `asan`, and `ubsan` presets provide
starting configurations. For example, `cmake --preset minimal` selects TCP
and sockets. Presets put their output under `build/<preset>`.

With a multi-configuration generator, choose the configuration when building
and testing, for example `cmake --build build --config Debug` and
`ctest --test-dir build -C Debug --output-on-failure`.

## Provider selection

`LIBFABRIC_PROVIDER_<NAME>` accepts three values:

| Value | Behavior |
| --- | --- |
| `AUTO` (default) | Build if the platform and dependencies allow it. |
| `ON` | Require the provider; fail configuration if it cannot be built. |
| `OFF` | Omit the provider and its dependency checks. |

Names are uppercase, for example `LIBFABRIC_PROVIDER_VERBS=ON` or
`LIBFABRIC_PROVIDER_PSM3=OFF`. The configuration output lists built-in and
loadable providers. Dependencies can be found through `CMAKE_PREFIX_PATH`,
pkg-config, or the corresponding `<Package>_ROOT` variable.

`LIBFABRIC_TCP_DLOPEN=ON` builds TCP as a separate provider plugin. The same
switch is available for other transport providers. Plugins require a shared
libfabric build. In the build tree, point `FI_PROVIDER_PATH` at
`build/libfabric/<configuration>`, for example `build/libfabric/Release`.
Installed plugins go in `<libdir>/libfabric`. If an installation is moved,
set `FI_PROVIDER_PATH` to that directory.

Libraries installed outside the system loader's search path may require
`CMAKE_INSTALL_RPATH` or `CMAKE_INSTALL_RPATH_USE_LINK_PATH=ON`. These are
standard CMake settings; dependency search paths are not forced into the
installed runtime search path.

## Optional features

GPU backends are opt-in: `LIBFABRIC_ENABLE_CUDA`, `LIBFABRIC_ENABLE_ZE`,
`LIBFABRIC_ENABLE_ROCR`, `LIBFABRIC_ENABLE_NEURON`, and
`LIBFABRIC_ENABLE_SYNAPSEAI`. An enabled backend requires its development
files. CUDA, Level Zero, and ROCr default to runtime loading; their
`LIBFABRIC_<BACKEND>_DLOPEN` options select direct linking instead when OFF.

A CUDA build using runtime loading requires headers, including NVML, but no
CUDA compiler or installed driver. Use `CUDAToolkit_ROOT` for a combined SDK,
or set `LIBFABRIC_CUDA_INCLUDE_DIR`, `LIBFABRIC_CUDART_INCLUDE_DIR`, and
`LIBFABRIC_NVML_INCLUDE_DIR` for split header packages. Direct linking uses
CMake's `FindCUDAToolkit` and requires the CUDA driver, runtime, and NVML
libraries.

Memory monitors, XPMEM, and io_uring are enabled when their dependencies are
available. They can be disabled with the corresponding `LIBFABRIC_*` cache
options. `LIBFABRIC_DEFAULT_MONITOR` selects `memhooks`, `uffd`, `kdreg2`, or
`disabled`; an unavailable selection is rejected.

`LIBFABRIC_BUILD_FABTESTS=ON` builds fabtests. `LIBFABRIC_BUILD_UTILITIES=OFF`
omits the command-line utilities. `LIBFABRIC_ENABLE_LTO`,
`LIBFABRIC_ENABLE_ASAN`, `LIBFABRIC_ENABLE_TSAN`, `LIBFABRIC_ENABLE_UBSAN`,
and `LIBFABRIC_ENABLE_LSAN` control compiler instrumentation.

Direct provider builds, external PSM3, the EFA provider's unit-test suite,
and the EFA GDA test are not implemented by this CMake build. The existing
Autotools and native Windows build files remain available.

## Using libfabric from CMake

After installation:

```cmake
find_package(Libfabric CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE Libfabric::fabric)
```

Set `CMAKE_PREFIX_PATH` when installing outside a standard prefix. A static
installation rediscovers its private dependencies on the consuming machine.
Private compiler definitions and build-tree paths are not exported.

A parent project can also use `add_subdirectory(path/to/libfabric)` and link
the same target. Installation, utilities, and CMake integration tests default
to OFF when libfabric is a subproject. Explicit cache options override those
defaults.

## Validation and maintenance

CTest covers provider discovery, TCP payload verification, relocated C and
C++ consumers, and provider reconfiguration and error handling. The TCP test
uses Python 3 when available. Install tests require a C++ compiler in addition
to the library's C compiler. Disable the integration tests with
`LIBFABRIC_BUILD_TESTING=OFF` when those tools are unavailable.

Source lists are explicit. When adding or removing a source, update both its
`Makefile.include` and `CMakeLists.txt`. Package discovery belongs in
`cmake/Find*.cmake`; library feature probes belong in the build configuration,
so installed consumers do not repeat libfabric's compiler checks. Each
provider owns its include paths, feature macros, and architecture flags.

Compilation does not establish hardware correctness. Provider execution and
GPU transfers still require their respective hardware test environments.
