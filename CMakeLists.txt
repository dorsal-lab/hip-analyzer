cmake_minimum_required(VERSION 3.16)
project(hip-analyzer LANGUAGES C CXX)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-fno-rtti")

# ---- Paths setup ----- #

set(ROCM_LLVM ${ROCM_PATH}/llvm/)
set(CMAKE_PREFIX_PATH ${ROCM_LLVM} ${CMAKE_PREFIX_PATH})
set(CLANG_INSTALL_PREFIX ${ROCM_LLVM})
set(CMAKE_LIBRARY_PATH ${ROCM_LLVM}/lib)

# ----- Packages ----- #

# HIP

list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)
find_package(hip)

# LLVM & Clang

link_directories(BEFORE ${ROCM_LLVM}/lib ${ROCM_PATH}/lib)

set(LLVM_LINK_COMPONENTS support)
find_package(Clang REQUIRED)
find_package(LLVM CONFIG REQUIRED)

include_directories(
    include
    ${ROCM_PATH}/llvm/include
    ${ROCM_PATH}/roctracer/include
    ${ROCM_PATH}/include/hsa
)

# jsoncpp

include_directories(/usr/include/jsoncpp) # quick hack but I'm tired of fighting CMake

# ----- Targets definitions ----- #

# Main tool

add_executable(hip-analyzer
    src/main.cpp
    src/hip_runtime_matcher.cpp
    src/cfg_instrumentation.cpp
    src/instr_generator.cpp
    src/cfg_inner_matchers.cpp
    src/basic_block.cpp
    src/llvm_ir_consumer.cpp
    src/llvm_instr_counters.cpp
)

target_link_libraries(hip-analyzer
    clangTooling
    clangFrontend
    clangAST
    clangASTMatchers
    clangBasic
    clangSerialization
    clangCodeGen
    LLVMAnalysis
    LLVMCore
    jsoncpp
)

# Instrumentation library

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/gpu_hip_instrumentation.o
    COMMAND ${ROCM_PATH}/hip/bin/hipcc -c -I${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src/gpu_functions.cpp -std=c++20 -o ${CMAKE_BINARY_DIR}/gpu_hip_instrumentation.o
    DEPENDS 
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_instrumentation.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/reduction_kernels.hpp
        ${CMAKE_SOURCE_DIR}/include/hip_instrumentation/hip_utils.hpp
        ${CMAKE_SOURCE_DIR}/src/gpu_functions.cpp
    VERBATIM
)

add_library(hip_instrumentation
    src/hip_instrumentation.cpp
    src/basic_block.cpp
    src/gpu_info.cpp
    src/state_recoverer.cpp
    ${CMAKE_BINARY_DIR}/gpu_hip_instrumentation.o
)

target_link_libraries(hip_instrumentation
    hip::host
    jsoncpp
    roctracer64
    rocprofiler64
)


# ----- Testing ----- #

add_subdirectory(test)
