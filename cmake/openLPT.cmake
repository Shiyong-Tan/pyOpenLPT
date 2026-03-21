# cmake/openLPT.cmake — modern, per-target CMake for OpenLPT
# Included by the top-level CMakeLists.txt when PYOPENLPT is OFF.

if (TARGET OpenLPT)
  return()
endif()

include(GNUInstallDirs)

# ---- Global defaults -------------------------------------------------
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Common include roots used during build
set(OPENLPT_INC_ROOTS
  "${PROJECT_SOURCE_DIR}/inc"
  "${PROJECT_SOURCE_DIR}/inc/libMath"
  "${PROJECT_SOURCE_DIR}/src/srcMath"
  "${PROJECT_SOURCE_DIR}/inc/libObject"
  "${PROJECT_SOURCE_DIR}/inc/libObject/BubbleCenterAndSizeByCircle"
  "${PROJECT_SOURCE_DIR}/inc/libObject/BubbleResize"
  "${PROJECT_SOURCE_DIR}/src/srcObject"
  "${PROJECT_SOURCE_DIR}/src/srcObject/BubbleCenterAndSizeByCircle"
  "${PROJECT_SOURCE_DIR}/src/srcObject/BubbleResize"
  "${PROJECT_SOURCE_DIR}/inc/libSTB"
  "${PROJECT_SOURCE_DIR}/src/srcSTB"
  "${PROJECT_SOURCE_DIR}/src"  # main.cpp lives here
)

if (MSVC)
  # Keep runtime consistent across all targets/configurations:
  # Debug  -> /MDd
  # Release-> /MD
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

# If you generate a config header from a template, enable:
# configure_file(${PROJECT_SOURCE_DIR}/config.h.in ${PROJECT_BINARY_DIR}/config.h @ONLY)
# list(APPEND OPENLPT_INC_ROOTS "${PROJECT_BINARY_DIR}")

# Helper: warnings per compiler
function(openlpt_apply_warnings tgt)
  if (MSVC)
    target_compile_options(${tgt} PRIVATE /W4 /permissive-)
  else()
    target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endfunction()

# Helper: attach include dirs with proper build/install interfaces
function(openlpt_public_includes tgt)
  # 基本包含
  target_include_directories(${tgt} PUBLIC
    $<BUILD_INTERFACE:${OPENLPT_INC_ROOTS}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
  # 如果使用内置 libtiff，把它的头路径作为“构建期 PUBLIC”传播给所有目标
  if (EXISTS "${PROJECT_SOURCE_DIR}/inc/libtiff/CMakeLists.txt")
    target_include_directories(${tgt} PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/inc/libtiff>
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/inc/libtiff/libtiff>
    )
  endif()
endfunction()


function(openlpt_interface_includes tgt)
  target_include_directories(${tgt} INTERFACE
    $<BUILD_INTERFACE:${OPENLPT_INC_ROOTS}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
endfunction()

# ---- Third-party deps -----------------------------------------------

# nanoflann (header-only): real target + namespaced alias
add_library(nanoflann INTERFACE)
target_include_directories(nanoflann INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/inc/nanoflann>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
add_library(OpenLPT::nanoflann ALIAS nanoflann)
install(TARGETS nanoflann EXPORT OpenLPTTargets)

# TIFF: prefer in-tree subproject; otherwise find_package
if (EXISTS "${PROJECT_SOURCE_DIR}/inc/libtiff/CMakeLists.txt")
  add_subdirectory("${PROJECT_SOURCE_DIR}/inc/libtiff" "${CMAKE_BINARY_DIR}/_deps/tiff")
  if (TARGET tiff AND NOT TARGET OpenLPT::tiff)
    add_library(OpenLPT::tiff ALIAS tiff)
  endif()
  if (TARGET tiff)
    install(TARGETS tiff EXPORT OpenLPTTargets)
  endif()
else()
  find_package(TIFF REQUIRED)
  if (NOT TARGET OpenLPT::tiff)
    add_library(OpenLPT::tiff INTERFACE)
    target_link_libraries(OpenLPT::tiff INTERFACE TIFF::TIFF)
  endif()
endif()

# OpenMP found at top-level; link per-target where needed

# ---- Core math / io / camera ----------------------------------------

# Matrix (header-only)
add_library(Matrix INTERFACE)
openlpt_interface_includes(Matrix)

# myMath
add_library(myMath STATIC "${PROJECT_SOURCE_DIR}/src/srcMath/myMATH.cpp")
openlpt_interface_includes(myMath)
target_link_libraries(myMath PUBLIC Matrix)
openlpt_apply_warnings(myMath)

# ImageIO
add_library(ImageIO STATIC "${PROJECT_SOURCE_DIR}/src/srcMath/ImageIO.cpp")
openlpt_public_includes(ImageIO)

# 如果使用内置 libtiff，则在构建期提供它的头路径（私有，不导出）
if (EXISTS "${PROJECT_SOURCE_DIR}/inc/libtiff/CMakeLists.txt")
  target_include_directories(ImageIO PRIVATE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/inc/libtiff>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/inc/libtiff/libtiff>
  )
endif()

target_link_libraries(ImageIO
  PUBLIC Matrix
  PRIVATE OpenLPT::tiff
)

openlpt_apply_warnings(ImageIO)

# Camera
add_library(Camera STATIC
  "${PROJECT_SOURCE_DIR}/src/srcMath/Camera.cpp"
)
openlpt_public_includes(Camera)
target_link_libraries(Camera PUBLIC myMath Matrix)
openlpt_apply_warnings(Camera)

# ---- Object modules --------------------------------------------------

# ObjectInfo
add_library(ObjectInfo STATIC "${PROJECT_SOURCE_DIR}/src/srcObject/ObjectInfo.cpp")
openlpt_public_includes(ObjectInfo)
target_link_libraries(ObjectInfo PUBLIC Matrix myMath)
openlpt_apply_warnings(ObjectInfo)

# CircleIdentifier (+ all circle detectors)
file(GLOB CIRCLE_SRCS "${PROJECT_SOURCE_DIR}/src/srcObject/BubbleCenterAndSizeByCircle/*.cpp")
add_library(CircleIdentifier STATIC
  "${PROJECT_SOURCE_DIR}/src/srcObject/CircleIdentifier.cpp"
  ${CIRCLE_SRCS}
)
openlpt_public_includes(CircleIdentifier)
target_link_libraries(CircleIdentifier 
  PUBLIC ObjectInfo myMath Matrix
  PUBLIC OpenMP::OpenMP_CXX)
openlpt_apply_warnings(CircleIdentifier)

# BubbleResize
file(GLOB BBRESIZE_SRCS "${PROJECT_SOURCE_DIR}/src/srcObject/BubbleResize/*.cpp")
add_library(BubbleResize STATIC ${BBRESIZE_SRCS})
openlpt_public_includes(BubbleResize)
target_link_libraries(BubbleResize PUBLIC ObjectInfo myMath OpenMP::OpenMP_CXX)
openlpt_apply_warnings(BubbleResize)

# BubbleRefImg (headers in inc/libObject, source in src/srcObject)
add_library(BubbleRefImg STATIC "${PROJECT_SOURCE_DIR}/src/srcObject/BubbleRefImg.cpp")
openlpt_public_includes(BubbleRefImg)
target_link_libraries(BubbleRefImg PUBLIC
  BubbleResize
  ObjectInfo
  myMath       # Matrix is header-only; include via PUBLIC includes
  OpenMP::OpenMP_CXX
)
openlpt_apply_warnings(BubbleRefImg)

# ObjectFinder (.cpp) — depends privately on nanoflann to avoid exporting it
add_library(ObjectFinder STATIC "${PROJECT_SOURCE_DIR}/src/srcObject/ObjectFinder.cpp")
openlpt_public_includes(ObjectFinder)
target_link_libraries(ObjectFinder
  PUBLIC Matrix myMath CircleIdentifier
  PUBLIC OpenMP::OpenMP_CXX
  PRIVATE OpenLPT::nanoflann
)
openlpt_apply_warnings(ObjectFinder)

# ---- STB family ------------------------------------------------------

# StereoMatch (.cpp)
add_library(StereoMatch STATIC "${PROJECT_SOURCE_DIR}/src/srcSTB/StereoMatch.cpp")
openlpt_public_includes(StereoMatch)
target_link_libraries(StereoMatch PUBLIC
  ObjectFinder Camera myMath Matrix OpenMP::OpenMP_CXX
)
openlpt_apply_warnings(StereoMatch)

# OTF
add_library(OTF STATIC "${PROJECT_SOURCE_DIR}/src/srcSTB/OTF.cpp")
openlpt_public_includes(OTF)
target_link_libraries(OTF PUBLIC Camera myMath Matrix OpenMP::OpenMP_CXX)
openlpt_apply_warnings(OTF)

# VSC (Volume Self Calibration)
add_library(VSC STATIC "${PROJECT_SOURCE_DIR}/src/srcSTB/VSC.cpp")
openlpt_public_includes(VSC)
target_link_libraries(VSC PUBLIC
  OTF Track ObjectFinder Camera myMath Matrix
  OpenMP::OpenMP_CXX
  PRIVATE OpenLPT::nanoflann
)
openlpt_apply_warnings(VSC)

# Shake (PUBLIC depend on BubbleRefImg)
add_library(Shake STATIC "${PROJECT_SOURCE_DIR}/src/srcSTB/Shake.cpp")
openlpt_public_includes(Shake)
target_link_libraries(Shake PUBLIC
  BubbleRefImg CircleIdentifier
  OTF ObjectInfo Camera myMath Matrix
  OpenMP::OpenMP_CXX
  OpenLPT::nanoflann            
)
openlpt_apply_warnings(Shake)


# IPR (.cpp) — PUBLIC depend on BubbleRefImg
add_library(IPR STATIC "${PROJECT_SOURCE_DIR}/src/srcSTB/IPR.cpp")
openlpt_public_includes(IPR)
target_link_libraries(IPR PUBLIC
  BubbleRefImg StereoMatch ObjectFinder Shake
  Camera myMath Matrix OpenMP::OpenMP_CXX
)
openlpt_apply_warnings(IPR)

# PredField (.cpp)
add_library(PredField STATIC "${PROJECT_SOURCE_DIR}/src/srcSTB/PredField.cpp")
openlpt_public_includes(PredField)
target_link_libraries(PredField PUBLIC myMath Matrix OpenMP::OpenMP_CXX)
openlpt_apply_warnings(PredField)

# ---- Config module ---------------------------------------------------
add_library(Config STATIC "${PROJECT_SOURCE_DIR}/src/srcSTB/Config.cpp")
openlpt_public_includes(Config)
# Also include plain inc/ so headers like error.hpp remain visible post-install
target_include_directories(Config PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/inc>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(Config PUBLIC
  ImageIO Camera OTF ObjectInfo BubbleResize myMath Matrix OpenMP::OpenMP_CXX
)
openlpt_apply_warnings(Config)

# Track (.cpp)
add_library(Track STATIC "${PROJECT_SOURCE_DIR}/src/srcSTB/Track.cpp")
openlpt_public_includes(Track)
target_link_libraries(Track PUBLIC 
  Config ObjectInfo myMath Matrix OpenMP::OpenMP_CXX
)
openlpt_apply_warnings(Track)

# STB (orchestrator)
add_library(STB STATIC "${PROJECT_SOURCE_DIR}/src/srcSTB/STB.cpp")
openlpt_public_includes(STB)
target_link_libraries(STB PUBLIC
  IPR Shake PredField VSC Track
  ObjectInfo ObjectFinder Camera myMath Matrix
  OpenMP::OpenMP_CXX
)
openlpt_apply_warnings(STB)



# ---- Final executable ------------------------------------------------
add_executable(OpenLPT "${PROJECT_SOURCE_DIR}/src/main.cpp")
target_compile_definitions(OpenLPT PRIVATE OPENLPT_BUILD_CLI)
# set_target_properties(OpenLPT PROPERTIES
#   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
#   RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_BINARY_DIR}/Debug"
#   RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release"
# )
set_target_properties(OpenLPT PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build/$<CONFIG>"
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build/$<CONFIG>"
  ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build/$<CONFIG>"
)
add_custom_command(TARGET OpenLPT POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E echo "OpenLPT exe: $<TARGET_FILE:OpenLPT>"
)
target_include_directories(OpenLPT PRIVATE ${OPENLPT_INC_ROOTS})
target_link_libraries(OpenLPT PRIVATE
  STB Config
  OpenLPT::nanoflann OpenMP::OpenMP_CXX
)
openlpt_apply_warnings(OpenLPT)

# ---- Install rules ---------------------------------------------------
# Install all public headers (so installed targets can #include after install)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/inc/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Install our targets & archives
install(TARGETS
  Matrix myMath ImageIO Camera ObjectInfo
  ObjectFinder CircleIdentifier BubbleResize BubbleRefImg
  StereoMatch OTF VSC Shake IPR PredField Track
  STB Config
  OpenLPT
  EXPORT OpenLPTTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
