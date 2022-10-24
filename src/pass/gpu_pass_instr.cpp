/** \file gpu_pass_instr.cpp
 * \brief Utility functions
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip/hip_runtime.h"

extern "C" {

/** \fn _hip_store_ctr
 * \brief Store the counters in the provided _instr_ptr
 */
[[clang::optnone]] __device__ void _hip_store_ctr(const uint8_t counters[],
                                                  size_t _bb_count,
                                                  uint8_t* _instr_ptr) {
#pragma unroll
    for (auto i = 0u; i < _bb_count; ++i) {
        _instr_ptr[blockIdx.x * blockDim.x * _bb_count +
                   threadIdx.x * _bb_count + i] = counters[i];
    }
}
}

namespace {

// Disable external linkage
[[clang::optnone]] __global__ void dummy_kernel_noopt(size_t _bb_count,
                                                      uint8_t* _instr_ptr) {
    _hip_store_ctr(nullptr, 0, nullptr);
}

constexpr size_t X = 512, Y = 512;
__global__ void dummy_kernel_opt(float* data, uint8_t* _instr_ptr) {
    uint8_t _bb_counters[2];
    size_t _bb_count = 2;
#pragma unroll
    for (auto i = 0u; i < _bb_count; ++i) {
        _bb_counters[i] = 0;
    }

    _bb_counters[1] += 1;
    size_t offset;
    size_t stride, limit;

    offset = (blockIdx.x * blockDim.x + threadIdx.x);
    stride = blockDim.x * gridDim.x;
    limit = X * Y;

    for (auto i = offset; i < limit; i += stride) {
        if (i < X * Y) {
            _bb_counters[0] += 1;
            data[i] = data[i] * data[i];
        }
    }

    _hip_store_ctr(_bb_counters, _bb_count, _instr_ptr);
}

} // namespace
