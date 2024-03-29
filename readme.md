# HIP-analyzer

The HIP-analyzer tool provides static analysis for runtime instrumentation (tracing & software counters) for heterogeneous applications using the AMD HIP programming framework.

This tool relies on the LLVM project, the pass framework, and its C++ frontend, Clang.

## Features

The current objective is to generate counters for each basic block in the compute kernel, in order to identify where divergence occurs.

Currently, the goal is to allow the user to add custom software counters and tracepoints during the execution for a fine-grained analysis of the kernel execution. We have successfully implemented various events as well as a tracing pass to this effect.

## Building the tool

Using `hip-analyzer` requires, until further improvements, compiling [ROCm-LLVM](https://github.com/RadeonOpenCompute/llvm-project) as a standalone component with `BUILD_SHARED_LIBS` and `ENABLE_LLVM_SHARED`
enabled. Use the tag corresponding to your ROCm install.

Run the following command :

```bash
mkdir build && cd build
cmake -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -DROCM_PATH=/opt/rocm -DROCM_LLVM=<path to llvm install directory>..
```

## Running

The `HIP_CLANG_PATH` environment variable must be set to the binary dir of the custom LLVM build to ensure 
that `hipcc` invokes the correct compiler.

The tool now relies on a set of LLVM IR passes which can be found in the `libhip-analyzer-pass.so` shared library. It can be run using the `opt` tool on LLVM IR modules, or directly in the `hipcc` compiler, using a few compiler flags.

`hipcc` (prefered way, one-shot compilation):
```bash
hipcc <input> -o <output> -fpass-plugin=libhip-analyzer-passes.so
```


`opt` :

```bash 
opt -load=libhip-analyzer-passes.so <module> [--hip-analyzer-counters | --hip-analyzer-trace | --hip-analyzer-host]
```
The flags to use depends on the module you're using (whether it be host or device)
