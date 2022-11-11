/** \file gpu_pass_instr.cpp
 * \brief Device-side utility functions
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip/hip_runtime.h"

extern "C" {

/** \fn _hip_store_ctr
 * \brief Store the counters in the provided _instr_ptr
 */
__device__ void _hip_store_ctr(const uint8_t counters[], size_t _bb_count,
                               uint8_t* _instr_ptr) {
    size_t base = blockIdx.x * blockDim.x * _bb_count + threadIdx.x * _bb_count;

    for (auto i = 0u; i < _bb_count; ++i) {
        _instr_ptr[base + i] = counters[i];
    }
}
}

// Disable external linkage
namespace {

// Dummy kernel calling all the utils, with no optimization to force no-inlining
// of device function
[[clang::optnone]] __global__ void dummy_kernel_noopt(size_t _bb_count,
                                                      uint8_t* _instr_ptr) {
    _hip_store_ctr(nullptr, 0, nullptr);
}

} // namespace
