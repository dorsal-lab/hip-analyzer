# HIP-analyzer

Static analysis for HIP application tracing using the clang AST.

## Building the tool

Run the following command :

```bash
mkdir build && cd build
cmake -DCMAKE_C_COMPILER=/opt/rocm-5.0.0/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm-5.0.0/llvm/bin/clang++ -DROCM_PATH=/opt/rocm ..
```

