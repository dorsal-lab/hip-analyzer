cmake_minimum_required(VERSION 3.16)
project(hip-analyzer)

# ---- Paths setup ----- #

set(ROCM_LLVM ${ROCM_PATH}/llvm/)
set(CMAKE_PREFIX_PATH ${ROCM_LLVM} ${CMAKE_PREFIX_PATH})


# ----- Packages ----- #
set(LLVM_LINK_COMPONENTS support)
find_package(Clang)

include_directories(
    include
    ${ROCM_PATH}/llvm/include
)


# ----- Targets definitions ----- #

add_executable(hip-analyzer
    src/main.cpp
)

target_link_libraries(hip-analyzer
    clangAST
    clangASTMatchers
    clangBasic
    clangFrontend
    clangSerialization
    clangTooling
)
