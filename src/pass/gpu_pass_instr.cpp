/** \file gpu_pass_instr.cpp
 * \brief Device-side utility functions
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip/hip_runtime.h"

#include "hip_instrumentation/gpu_queue.hpp"

extern "C" {

// ----- Counters instrumentation ----- //

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

// ----- Tracing instrumentation ----- //

__device__ void* _hip_get_trace_offset(void* storage, size_t* offsets,
                                       size_t event_size) {
    auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    auto offset = offsets[thread_id];

    // We are going to calculate a byte offset, cast to byte-sized (nice pun)
    // array
    auto* bitcast = reinterpret_cast<uint8_t*>(storage);

    return &bitcast[offset * event_size];
}

// Generic function pointer to : void event_ctor(void* storage, size_t bb);
using event_ctor = void (*)(void*, size_t);

__device__ void _hip_create_event(void* storage, size_t* idx, size_t event_size,
                                  event_ctor ctor, size_t bb) {
    auto* bitcast = reinterpret_cast<uint8_t*>(storage);

    // Postfix increment global index
    auto curr_index = (*idx)++;
    ctor(&bitcast[curr_index * event_size], bb);
}

// ----- Events constructors ----- //

__device__ void _hip_event_ctor(void* buf, size_t bb) {
    new (buf) hip::Event{bb};
}

__device__ void _hip_tagged_event_ctor(void* buf, size_t bb) {
    new (buf) hip::TaggedEvent{bb};
}

__device__ void _hip_wavestate_ctor(void* buf, size_t bb) {
    new (buf) hip::WaveState{bb};
}
}

// Disable external linkage
namespace {

// Dummy kernel calling all the utils, with no optimization to force no-inlining
// of device function
[[clang::optnone]] __global__ void dummy_kernel_noopt(size_t _bb_count,
                                                      uint8_t* _instr_ptr) {
    _hip_store_ctr(nullptr, 0, nullptr);
    _hip_get_trace_offset(nullptr, nullptr, 0);
    _hip_create_event(nullptr, nullptr, 0, nullptr, 0);
}

} // namespace
