cmake_minimum_required(VERSION 3.16)
project(hip-analyzer LANGUAGES C CXX)


set(CMAKE_CXX_STANDARD 20)
list(APPEND CMAKE_CXX_FLAGS "-Wall -O3 -fPIE -fno-rtti")

# ---- Paths setup ----- #

string(REPLACE ":" ";" CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH}")
string(REPLACE ":" ";" CMAKE_LIBRARY_PATH "$ENV{CMAKE_LIBRARY_PATH}")


if(NOT DEFINED ROCM_LLVM)
    set(ROCM_LLVM ${ROCM_PATH}/llvm)
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_LLVM};${ROCM_PATH};${ROCM_PATH}/hip")
set(CLANG_INSTALL_PREFIX ${ROCM_LLVM})
list(APPEND CMAKE_LIBRARY_PATH "${ROCM_LLVM}/lib")
list(APPEND CMAKE_MODULE_PATH "${ROCM_LLVM}/lib/cmake/llvm")

message(${CMAKE_PREFIX_PATH})

# ----- Packages ----- #

# HIP

find_package(hip)

# LLVM & Clang

link_directories(BEFORE ${ROCM_LLVM}/lib ${ROCM_PATH}/lib)

set(LLVM_LINK_COMPONENTS support)
find_package(Clang REQUIRED)
find_package(LLVM CONFIG REQUIRED)
message(${LLVM_INSTALL_PREFIX})

include_directories(
    include
    ${ROCM_LLVM}/include
    ${ROCM_PATH}/roctracer/include
    ${ROCM_PATH}/include/hsa
)

# jsoncpp

include_directories(/usr/include/jsoncpp) # quick hack but I'm tired of fighting CMake

# ----- Targets definitions ----- #

# Instrumentation library

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/gpu_hip_instrumentation.o
    COMMAND ${ROCM_PATH}/hip/bin/hipcc -fPIE -c -I${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src/gpu_functions.cpp -std=c++20 -o ${CMAKE_BINARY_DIR}/gpu_hip_instrumentation.o
    DEPENDS 
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_instrumentation.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/reduction_kernels.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_utils.hpp
        ${CMAKE_SOURCE_DIR}/src/gpu_functions.cpp
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/gpu_queue.o
    COMMAND ${ROCM_PATH}/hip/bin/hipcc -fPIE -c -I${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src/gpu_queue.cpp -std=c++20 -o ${CMAKE_BINARY_DIR}/gpu_queue.o
    DEPENDS 
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_instrumentation.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/gpu_queue.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/reduction_kernels.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_utils.hpp
        ${CMAKE_SOURCE_DIR}/src/gpu_queue.cpp
    VERBATIM
)

# LLVM Pass

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/gpu_pass_instr.ll
    COMMAND ${ROCM_PATH}/hip/bin/hipcc -c -S -emit-llvm ${CMAKE_SOURCE_DIR}/src/pass/gpu_pass_instr.cpp --cuda-device-only -O0 -o gpu_pass_instr.ll -Wno-unused-command-line-argument -I${CMAKE_SOURCE_DIR}/include -std=c++20
    DEPENDS
        ${CMAKE_SOURCE_DIR}/src/pass/gpu_pass_instr.cpp
    VERBATIM
)

add_custom_target(gpu_pass_instr DEPENDS ${CMAKE_BINARY_DIR}/gpu_pass_instr.ll)

add_library(
    hip-analyzer-pass SHARED
    src/pass/hip_analyzer_pass.cpp
    src/pass/hip_analyzer_device_pass.cpp
    src/pass/hip_analyzer_host_pass.cpp
    src/pass/ir_codegen.cpp
    src/pass/trace_type.cpp
    src/llvm_instr_counters.cpp
    src/basic_block.cpp
)

add_library(
    hip-kernel-timer-pass SHARED
    src/pass/kernel_timer_pass.cpp
    src/pass/ir_codegen.cpp
    src/llvm_instr_counters.cpp
    src/basic_block.cpp
)

add_dependencies(hip-analyzer-pass gpu_pass_instr)

target_link_libraries(
    hip-analyzer-pass
    LLVMPasses
    LLVMCore
    LLVMSupport
    LLVMAnalysis
    jsoncpp
)

target_link_libraries(
    hip-kernel-timer-pass
    LLVMPasses
    LLVMCore
    LLVMSupport
    LLVMAnalysis
    jsoncpp
)

add_library(hip_instrumentation
    src/hip_instrumentation.cpp
    src/basic_block.cpp
    src/gpu_info.cpp
    src/state_recoverer.cpp
    src/hip_instrumentation_cbindings.cpp
    ${CMAKE_BINARY_DIR}/gpu_hip_instrumentation.o
    ${CMAKE_BINARY_DIR}/gpu_queue.o
)

target_link_libraries(hip_instrumentation
    hip::host
    jsoncpp
    roctracer64
    rocprofiler64
    dl
    pthread
    Terminfo::terminfo
)


# ----- Testing ----- #

add_subdirectory(test)
