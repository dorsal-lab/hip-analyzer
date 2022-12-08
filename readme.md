# HIP-analyzer

The HIP-analyzer tool provides static analysis for runtime instrumentation (tracing & software counters) for heterogeneous applications using the AMD HIP programming framework.

This tool relies on the LLVM project, the pass framework, and its C++ frontend, Clang.

## Features

The current objective is to generate counters for each basic block in the compute kernel, in order to identify where divergence occurs.

Currently, the goal is to allow the user to add custom software counters and tracepoints during the execution for a fine-grained analysis of the kernel execution. We have successfully implemented various events as well as a tracing pass to this effect.

## Building the tool

Run the following command :

```bash
mkdir build && cd build
cmake -DCMAKE_C_COMPILER=/opt/rocm-5.0.0/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm-5.0.0/llvm/bin/clang++ -DROCM_PATH=/opt/rocm ..
```

## Running

### Currently supported way

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




### Legacy front-end plugin
```bash
build/hip-analyzer -p <path to compilation database> <input file> -k <kernel name> -o <output file>
```

The compilation database can be obtained using CMake (`-DCMAKE_EXPORT_COMPILE_COMMANDS=On`) or the [`bear` tool](https://github.com/rizsotto/Bear).

The output file has to be linked with `libhip_instrumentation.a`, generated during compilation. It provides runtime utilities for the instrumentation as well as GPU reductions for the instrumentation data (e.g. sum the total count for a basic block).
