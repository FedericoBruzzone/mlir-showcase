cmake_minimum_required(VERSION 3.21)
project(iree_mobilenet_runner C)

option(MOBILENET_RUNNER_STATIC_LINK
  "Attempt full static linking for the mobilenet_runner executable (Linux only)."
  OFF)

if(NOT DEFINED IREE_SOURCE_DIR)
  set(IREE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/iree")
endif()

# Accept relative paths from both:
# - native_runner/ (CMAKE_CURRENT_SOURCE_DIR)
# - repo root (parent of native_runner)
if(NOT IS_ABSOLUTE "${IREE_SOURCE_DIR}")
  set(_iree_candidate_from_native_runner
      "${CMAKE_CURRENT_SOURCE_DIR}/${IREE_SOURCE_DIR}")
  set(_iree_candidate_from_repo_root
      "${CMAKE_CURRENT_SOURCE_DIR}/../${IREE_SOURCE_DIR}")

  if(EXISTS "${_iree_candidate_from_native_runner}/CMakeLists.txt")
    set(IREE_SOURCE_DIR "${_iree_candidate_from_native_runner}")
  elseif(EXISTS "${_iree_candidate_from_repo_root}/CMakeLists.txt")
    set(IREE_SOURCE_DIR "${_iree_candidate_from_repo_root}")
  endif()
endif()

if(NOT EXISTS "${IREE_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "IREE source not found. Set -DIREE_SOURCE_DIR=/path/to/iree")
endif()

# Runtime-only build.
set(IREE_BUILD_COMPILER OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# CPU local-task runtime and loaders.
set(IREE_HAL_DRIVER_DEFAULTS OFF CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_LOCAL_TASK ON CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_LOCAL_SYNC ON CACHE BOOL "" FORCE)

set(IREE_HAL_EXECUTABLE_LOADER_DEFAULTS OFF CACHE BOOL "" FORCE)
set(IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF ON CACHE BOOL "" FORCE)
set(IREE_HAL_EXECUTABLE_LOADER_SYSTEM_LIBRARY ON CACHE BOOL "" FORCE)

add_subdirectory("${IREE_SOURCE_DIR}" "iree-build" EXCLUDE_FROM_ALL)

add_executable(mobilenet_runner runner.c)
target_compile_options(mobilenet_runner PRIVATE ${IREE_DEFAULT_COPTS})
target_link_libraries(mobilenet_runner PRIVATE iree_runtime_runtime)

if(MOBILENET_RUNNER_STATIC_LINK)
  if(APPLE)
    message(WARNING
      "MOBILENET_RUNNER_STATIC_LINK was requested, but fully static linking is "
      "not supported on macOS. Building with normal system dynamic libraries.")
  elseif(UNIX)
    target_link_options(mobilenet_runner PRIVATE -static)
  endif()
endif()
