cmake_minimum_required(VERSION 3.21)
project(iree_mobilenet_runner C)

option(MOBILENET_RUNNER_STATIC_LINK
  "Attempt full static linking for the mobilenet_runner executable (Linux only)."
  OFF)
option(MOBILENET_BUILD_EMBEDDED_RUNNER
  "Build mobilenet_runner_embedded that embeds VMFB + dylib generated from .ll."
  ON)

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

if(MOBILENET_BUILD_EMBEDDED_RUNNER)
  set(MOBILENET_LL_PATH
      "${CMAKE_CURRENT_SOURCE_DIR}/../dump_plugin/module___linked_system_dylib_arm_64.linked.ll"
      CACHE FILEPATH "Input LLVM .ll used to build embedded system dylib")
  set(MOBILENET_VMFB_PATH
      "${CMAKE_CURRENT_SOURCE_DIR}/../mobilenet_v2_plugin.vmfb"
      CACHE FILEPATH "Input VMFB embedded into mobilenet_runner_embedded")
  set(MOBILENET_LL_TARGET_TRIPLE "arm64-apple-macosx14.0.0"
      CACHE STRING "Target triple used when compiling MOBILENET_LL_PATH")

  if(NOT EXISTS "${MOBILENET_LL_PATH}")
    message(FATAL_ERROR "MOBILENET_LL_PATH not found: ${MOBILENET_LL_PATH}")
  endif()
  if(NOT EXISTS "${MOBILENET_VMFB_PATH}")
    message(FATAL_ERROR "MOBILENET_VMFB_PATH not found: ${MOBILENET_VMFB_PATH}")
  endif()

  find_program(LLVM_LLC_EXECUTABLE NAMES llc REQUIRED)
  find_program(XXD_EXECUTABLE NAMES xxd REQUIRED)

  set(MOBILENET_EMBEDDED_DIR "${CMAKE_CURRENT_BINARY_DIR}/embedded_assets")
  set(MOBILENET_EMBEDDED_DYLIB "${MOBILENET_EMBEDDED_DIR}/__linked_system_dylib_arm_64.dylib")
  set(MOBILENET_EMBEDDED_DYLIB_OBJ "${MOBILENET_EMBEDDED_DIR}/__linked_system_dylib_arm_64.o")
  set(MOBILENET_EMBEDDED_VMFB_INC "${MOBILENET_EMBEDDED_DIR}/embedded_vmfb.inc")
  set(MOBILENET_EMBEDDED_DYLIB_INC "${MOBILENET_EMBEDDED_DIR}/embedded_dylib.inc")

  add_custom_command(
    OUTPUT "${MOBILENET_EMBEDDED_DYLIB}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${MOBILENET_EMBEDDED_DIR}"
    COMMAND "${LLVM_LLC_EXECUTABLE}" "${MOBILENET_LL_PATH}"
            "-mtriple=${MOBILENET_LL_TARGET_TRIPLE}"
            -filetype=obj
            -o "${MOBILENET_EMBEDDED_DYLIB_OBJ}"
    COMMAND "${CMAKE_C_COMPILER}" -shared "${MOBILENET_EMBEDDED_DYLIB_OBJ}"
            -o "${MOBILENET_EMBEDDED_DYLIB}"
    DEPENDS "${MOBILENET_LL_PATH}"
    COMMENT "Compiling embedded system dylib from LLVM IR"
    VERBATIM
  )

  add_custom_command(
    OUTPUT "${MOBILENET_EMBEDDED_VMFB_INC}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${MOBILENET_EMBEDDED_DIR}"
    COMMAND "${XXD_EXECUTABLE}" -i -n mobilenet_embedded_vmfb
            "${MOBILENET_VMFB_PATH}" "${MOBILENET_EMBEDDED_VMFB_INC}"
    DEPENDS "${MOBILENET_VMFB_PATH}"
    COMMENT "Embedding VMFB payload as C array"
    VERBATIM
  )

  add_custom_command(
    OUTPUT "${MOBILENET_EMBEDDED_DYLIB_INC}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${MOBILENET_EMBEDDED_DIR}"
    COMMAND "${XXD_EXECUTABLE}" -i -n mobilenet_embedded_dylib
            "${MOBILENET_EMBEDDED_DYLIB}" "${MOBILENET_EMBEDDED_DYLIB_INC}"
    DEPENDS "${MOBILENET_EMBEDDED_DYLIB}"
    COMMENT "Embedding compiled dylib payload as C array"
    VERBATIM
  )

  add_custom_target(mobilenet_runner_embedded_assets
    DEPENDS
      "${MOBILENET_EMBEDDED_VMFB_INC}"
      "${MOBILENET_EMBEDDED_DYLIB_INC}"
  )

  add_executable(mobilenet_runner_embedded runner_embedded.c)
  add_dependencies(mobilenet_runner_embedded mobilenet_runner_embedded_assets)
  target_compile_options(mobilenet_runner_embedded PRIVATE ${IREE_DEFAULT_COPTS})
  target_include_directories(mobilenet_runner_embedded PRIVATE "${MOBILENET_EMBEDDED_DIR}")
  target_link_libraries(mobilenet_runner_embedded PRIVATE iree_runtime_runtime)

  if(MOBILENET_RUNNER_STATIC_LINK)
    if(APPLE)
      message(WARNING
        "MOBILENET_RUNNER_STATIC_LINK was requested, but fully static linking is "
        "not supported on macOS. Building with normal system dynamic libraries.")
    elseif(UNIX)
      target_link_options(mobilenet_runner_embedded PRIVATE -static)
    endif()
  endif()
endif()
