# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)
include(FetchContent)

# NMSIS-NN configuration with dynamic path detection
set(NMSIS_VERSION
    "1.5.0"
    CACHE STRING "NMSIS-NN version to download"
)
set(NMSIS_LOCAL_PATH
    ""
    CACHE PATH "Path to existing local NMSIS-NN installation"
)

# Try to find existing / local NMSIS-NN installation. This is useful for
# debugging and testing with local changes. This is not common, as the NMSIS-NN
# library is downloaded via FetchContent in the default/regular case.
if(NMSIS_LOCAL_PATH AND EXISTS "${NMSIS_LOCAL_PATH}")
  message(STATUS "Using NMSIS-NN from specified path: ${NMSIS_LOCAL_PATH}/NN/Source")
  add_subdirectory(${NMSIS_LOCAL_PATH}/NN/Source _deps/nmsis_nn-build)
else()
  # Use FetchContent with automatic fallback
  message(STATUS "Using NMSIS-NN via FetchContent")

  FetchContent_Declare(
    nmsis_nn
    GIT_REPOSITORY https://github.com/Nuclei-Software/NMSIS.git
    GIT_TAG ${NMSIS_VERSION}
    GIT_SHALLOW TRUE
  )

  FetchContent_GetProperties(nmsis_nn)
  
  if(NOT nmsis_nn_POPULATED)
    FetchContent_Populate(nmsis_nn)
    add_subdirectory(${nmsis_nn_SOURCE_DIR}/NMSIS/NN/Source ${nmsis_nn_BINARY_DIR})
  endif()
endif()

# nuclei ops kernel sources
set(_nuclei_kernels__srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantize_per_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_dequantize_per_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantized_add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantized_avg_pool2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantized_batch_matmul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantized_conv2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantized_depthwise_conv2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantized_linear.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantized_max_pool2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantized_mul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantized_transpose_conv2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_pad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_minimum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_maximum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_transpose.cpp
)

# Generate C++ bindings to register kernels into Executorch
set(_yaml_file ${CMAKE_CURRENT_LIST_DIR}/ops/operators.yaml)
gen_selected_ops(LIB_NAME "nuclei_ops_lib" OPS_SCHEMA_YAML "${_yaml_file}")
generate_bindings_for_kernels(
  LIB_NAME "nuclei_ops_lib" CUSTOM_OPS_YAML "${_yaml_file}"
)

# Build library for nuclei_kernels
add_library(nuclei_kernels ${_nuclei_kernels__srcs})

# Use PRIVATE for implementation dependencies to avoid INTERFACE pollution
target_link_libraries(
  nuclei_kernels
  PRIVATE NMSISNN
  PRIVATE executorch
  PRIVATE kernels_util_all_deps
)

# Include directories for nuclei_kernels
target_include_directories(
  nuclei_kernels PRIVATE ${EXECUTORCH_ROOT}/..
                           ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
)

# nuclei_ops_lib: Register Nuclei ops kernels into Executorch runtime
gen_operators_lib(
  LIB_NAME "nuclei_ops_lib" KERNEL_LIBS nuclei_kernels DEPS executorch
)

install(
  TARGETS nuclei_kernels nuclei_ops_lib NMSISNN
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/backends/nuclei/ops/
)
