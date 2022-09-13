# HIP-analyzer

The HIP-analyzer tool provides static analysis for runtime instrumentation (tracing & software counters) for heterogeneous applications using the AMD HIP programming framework.

This tool relies on the LLVM project and its C++ frontend, Clang.

## Features

The current objective is to generate counters for each basic block in the compute kernel, in order to identify where divergence occurs.

Ultimately, the goal is to allow the user to add custom software counters and tracepoints during the execution for a fine-grained analysis of the kernel execution.

So far, the tool only supports single-source programs.

## Building the tool

Run the following command :

```bash
mkdir build && cd build
cmake -DCMAKE_C_COMPILER=/opt/rocm-5.0.0/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm-5.0.0/llvm/bin/clang++ -DROCM_PATH=/opt/rocm ..
```

If you encounter some building problems with `Terminfo::terminfo` (related to LLVM 14), installing libtinfo-dev seems to fix the problem.

## Running

```bash
build/hip-analyzer -p <path to compilation database> <input file> -k <kernel name> -o <output file>
```

The compilation database can be obtained using CMake (`-DCMAKE_EXPORT_COMPILE_COMMANDS=On`) or the [`bear` tool](https://github.com/rizsotto/Bear).

The output file has to be linked with `libhip_instrumentation.a`, generated during compilation. It provides runtime utilities for the instrumentation as well as GPU reductions for the instrumentation data (e.g. sum the total count for a basic block).
