cmake_minimum_required(VERSION 3.15...3.28)

# --------------------------
# Project
# --------------------------
project(OpenLPT VERSION 0.1.0 LANGUAGES C CXX)

# Global C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default build type for single-config generators
if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Toggle: build Python bindings via pybind11
option(PYOPENLPT "Build Python bindings (pybind11)" OFF)
option(OPENLPT_GPU_EXACT_MORPHOLOGY
       "Offload exact Boolean morphology propagation to CUDA" OFF)

if (OPENLPT_GPU_EXACT_MORPHOLOGY)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Optional: OpenMP detection (link per-target inside module cmake files)
find_package(OpenMP QUIET COMPONENTS CXX)
if (OpenMP_CXX_FOUND)
  message(STATUS "OpenMP CXX found")
endif()

# --------------------------
# Core C++ libraries
# --------------------------
# Expect your existing library definitions here
if (EXISTS "${PROJECT_SOURCE_DIR}/cmake/openLPT.cmake")
  include("${PROJECT_SOURCE_DIR}/cmake/openLPT.cmake")
else()
  message(FATAL_ERROR "Missing cmake/openLPT.cmake")
endif()

# --------------------------
# Python bindings (pybind11)
# --------------------------
if (PYOPENLPT)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  if (NOT DEFINED pybind11_DIR)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -c "import pybind11,sys; sys.stdout.write(pybind11.get_cmake_dir())"
      OUTPUT_VARIABLE _pybind11_cmakedir
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(pybind11_DIR "${_pybind11_cmakedir}" CACHE PATH "pybind11 CMake config dir" FORCE)
  endif()
  find_package(pybind11 CONFIG REQUIRED)


  # Build the Python module (defined in cmake/bindings.cmake)
  if (EXISTS "${PROJECT_SOURCE_DIR}/cmake/bindings.cmake")
    include("${PROJECT_SOURCE_DIR}/cmake/bindings.cmake")
  else()
    message(FATAL_ERROR "Missing cmake/bindings.cmake")
  endif()

# --------------------------
# Native C++ install/export
# --------------------------
else()
  include(GNUInstallDirs)

  # Install public headers
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/inc/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  # Ensure interface include target is part of export set if other libs link to it
  if (TARGET openlpt_includes)
    install(TARGETS openlpt_includes EXPORT OpenLPTTargets)
  endif()

  # (Optional) tests
  option(OPENLPT_BUILD_TESTS "Build OpenLPT tests" OFF)
  if (OPENLPT_BUILD_TESTS AND EXISTS "${PROJECT_SOURCE_DIR}/cmake/ctest.cmake")
    include("${PROJECT_SOURCE_DIR}/cmake/ctest.cmake")
  endif()

  # Package config export
  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/OpenLPTConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
  )

  # If you have a template config, keep it; otherwise a minimal one can be added later.
  if (EXISTS "${PROJECT_SOURCE_DIR}/cmake/OpenLPTConfig.cmake.in")
    configure_package_config_file(
      "${PROJECT_SOURCE_DIR}/cmake/OpenLPTConfig.cmake.in"
      "${CMAKE_CURRENT_BINARY_DIR}/OpenLPTConfig.cmake"
      INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenLPT
    )
    install(FILES
      "${CMAKE_CURRENT_BINARY_DIR}/OpenLPTConfig.cmake"
      "${CMAKE_CURRENT_BINARY_DIR}/OpenLPTConfigVersion.cmake"
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenLPT
    )
  else()
    # If you don't maintain a config template, at least install the version file
    install(FILES
      "${CMAKE_CURRENT_BINARY_DIR}/OpenLPTConfigVersion.cmake"
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenLPT
    )
  endif()

  # Export targets created in openLPT.cmake
  install(EXPORT OpenLPTTargets
    NAMESPACE OpenLPT::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenLPT
  )
endif()
