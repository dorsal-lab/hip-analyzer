cmake_minimum_required(VERSION 3.16)
project(hip-analyzer LANGUAGES C CXX)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -fPIE -fno-rtti")

option(ENABLE_TRACEPOINTS "Enable LTTng-UST Tracepoints" OFF)

# ---- Paths setup ----- #

string(REPLACE ":" ";" CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH}")
string(REPLACE ":" ";" CMAKE_LIBRARY_PATH "$ENV{CMAKE_LIBRARY_PATH}")

if(NOT DEFINED ROCM_LLVM)
    set(ROCM_LLVM ${ROCM_PATH})
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_LLVM};${ROCM_PATH};${ROCM_PATH}/hip")
set(CLANG_INSTALL_PREFIX ${ROCM_LLVM})
list(APPEND CMAKE_LIBRARY_PATH "${ROCM_LLVM}/lib")
list(APPEND CMAKE_MODULE_PATH "${ROCM_LLVM}/lib/cmake/llvm")

message(${CMAKE_PREFIX_PATH})

# ----- Packages ----- #

# HIP

find_package(hip REQUIRED)

# LLVM & Clang

link_directories(BEFORE ${ROCM_LLVM}/lib ${ROCM_PATH}/lib)

set(LLVM_LINK_COMPONENTS support)
find_package(Clang REQUIRED)
find_package(LLVM CONFIG REQUIRED)

include_directories(
    include
    ${ROCM_LLVM}/include
    ${ROCM_PATH}/roctracer/include
    ${ROCM_PATH}/include/hsa
)

# jsoncpp

include_directories(/usr/include/jsoncpp) # quick hack but I'm tired of fighting CMake

# Tracepoints

if(ENABLE_TRACEPOINTS)
    find_package(LTTngUST REQUIRED)
    add_definitions(-DENABLE_TRACEPOINTS)
    set(ENABLE_TRACEPOINTS_FLAGS -DENABLE_TRACEPOINTS)

    # CheckGitSetup (from https://jonathanhamberg.com/post/cmake-embedding-git-hash/)

    include(TracepointProvider.cmake)
    CheckGitSetup()
endif()

# ----- Targets definitions ----- #

# Instrumentation library

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/gpu_hip_instrumentation.o
    COMMAND ${ROCM_PATH}/bin/hipcc -fPIE -c -I${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src/gpu_functions.cpp -std=c++20 -o ${CMAKE_BINARY_DIR}/gpu_hip_instrumentation.o
    DEPENDS 
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_instrumentation.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/reduction_kernels.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_utils.hpp
        ${CMAKE_SOURCE_DIR}/src/gpu_functions.cpp
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/queue_info.o
    COMMAND ${ROCM_PATH}/bin/hipcc -fPIE -c -I${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src/queue_info.cpp -std=c++20 -o ${CMAKE_BINARY_DIR}/queue_info.o ${ENABLE_TRACEPOINTS_FLAGS}
    DEPENDS 
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_instrumentation.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/queue_info.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/reduction_kernels.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_utils.hpp
        ${CMAKE_SOURCE_DIR}/src/queue_info.cpp
    VERBATIM
)

# LLVM Pass

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/gpu_pass_instr.ll
    COMMAND ${ROCM_PATH}/bin/hipcc -c -S -emit-llvm ${CMAKE_SOURCE_DIR}/src/pass/gpu_pass_instr.hip --cuda-device-only -O0 -o gpu_pass_instr.ll -Wno-unused-command-line-argument -I${CMAKE_SOURCE_DIR}/include -std=c++20
    DEPENDS
        ${CMAKE_SOURCE_DIR}/src/pass/gpu_pass_instr.hip
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

target_link_libraries(hip-analyzer-pass
    LLVMPasses
    LLVMCore
    LLVMSupport
    LLVMAnalysis
    jsoncpp
)

target_link_libraries(hip-kernel-timer-pass
    LLVMPasses
    LLVMCore
    LLVMSupport
    LLVMAnalysis
    jsoncpp
)

# add_definitions(-DHIP_INSTRUMENTATION_VERBOSE)
add_library(hip_instrumentation
)

target_link_libraries(hip_instrumentation
    hip::host
    jsoncpp
    rocprofiler-sdk
    dl
    pthread
)

if(ENABLE_TRACEPOINTS)
    target_link_libraries(hip_instrumentation LTTng::UST)
    add_custom_target(hip_instrumentation_tracepoints_o
        COMMAND ar -x $<TARGET_FILE:hip_instrumentation_tracepoints>
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DEPENDS hip_instrumentation_tracepoints
    )
    add_dependencies(hip_instrumentation hip_instrumentation_tracepoints_o)

    target_sources(hip_instrumentation PRIVATE
        src/hip_instrumentation.cpp
        src/hip_trace_manager.cpp
        src/basic_block.cpp
        src/gpu_info.cpp
        src/state_recoverer.cpp
        src/hip_instrumentation_cbindings.cpp
        src/managed_queue_info.cpp
        ${CMAKE_BINARY_DIR}/hip_instrumentation_tracepoint_provider.cpp.o
        ${CMAKE_BINARY_DIR}/gpu_hip_instrumentation.o
        ${CMAKE_BINARY_DIR}/queue_info.o
    )
else()
    target_sources(hip_instrumentation PRIVATE
        src/hip_instrumentation.cpp
        src/hip_trace_manager.cpp
        src/basic_block.cpp
        src/gpu_info.cpp
        src/state_recoverer.cpp
        src/hip_instrumentation_cbindings.cpp
        src/managed_queue_info.cpp
        ${CMAKE_BINARY_DIR}/gpu_hip_instrumentation.o
        ${CMAKE_BINARY_DIR}/queue_info.o
    )
endif()

# ----- Testing ----- #

add_subdirectory(test)
