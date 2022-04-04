cmake_minimum_required(VERSION 3.16)
project(hip-analyzer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-fno-rtti")

# ---- Paths setup ----- #

set(ROCM_LLVM ${ROCM_PATH}/llvm/)
set(CMAKE_PREFIX_PATH ${ROCM_LLVM} ${CMAKE_PREFIX_PATH})
set(CLANG_INSTALL_PREFIX ${ROCM_LLVM})
set(CMAKE_LIBRARY_PATH ${ROCM_LLVM}/lib)

# ----- Packages ----- #
set(LLVM_LINK_COMPONENTS support)
find_package(Clang)

include_directories(
    include
    ${ROCM_PATH}/llvm/include
)

link_directories(BEFORE ${ROCM_LLVM}/lib)


# ----- Targets definitions ----- #

add_executable(hip-analyzer
    src/main.cpp
    src/flop_counter.cpp
    src/hip_runtime_matcher.cpp
)

target_link_libraries(hip-analyzer
    clangTooling
    clangFrontend
    clangAST
    clangASTMatchers
    clangBasic
    clangSerialization
)
