/** \file gpu_pass_instr.cpp
 * \brief Device-side utility functions
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip/hip_runtime.h"

#include "hip_instrumentation/gpu_queue.hpp"

extern "C" {

// ----- Counters instrumentation ----- //

/** \fn _hip_inc_wave_ctr
 * \brief Increment a wave counter
 */
__device__ void _hip_inc_wave_ctr(uint8_t* counter) {
    uint64_t thread = __lastbit_u32_u64(hip::gcnasm::get_exec());
    if (threadIdx.x % warpSize == thread) {
        ++(*counter);
    }
}

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

__device__ void* _hip_get_wave_trace_offset(void* storage, size_t* offsets,
                                            size_t event_size) {
    size_t waves_per_block;
    if (blockDim.x % warpSize == 0) {
        waves_per_block = blockDim.x / warpSize;
    } else {
        waves_per_block = blockDim.x / warpSize + 1;
    }

    auto wave_in_block = threadIdx.x / warpSize;

    size_t wavefront_id = blockIdx.x * waves_per_block + wave_in_block;

    auto offset = offsets[wavefront_id];

    // We are going to calculate a byte offset, cast to byte-sized (nice pun)
    // array
    auto* bitcast = reinterpret_cast<uint8_t*>(storage);

    return &bitcast[offset * event_size];
}

// Generic function pointer to : void event_ctor(void* storage, size_t bb);
using event_ctor = void (*)(void*, size_t);

__device__ void _hip_create_event(void* storage, uint32_t* idx,
                                  size_t event_size, event_ctor ctor,
                                  size_t bb) {
    auto* bitcast = reinterpret_cast<uint8_t*>(storage);

    ctor(&bitcast[*idx * event_size], bb);
}

__device__ size_t* _hip_wave_get_index_in_block(size_t* idx_array) {
    auto wave_in_block = threadIdx.x / warpSize;
    return &idx_array[wave_in_block];
}

__device__ void _hip_create_wave_event(void* storage, uint32_t* idx,
                                       size_t event_size, event_ctor ctor,
                                       size_t bb) {
    uint64_t thread = __lastbit_u32_u64(hip::gcnasm::get_exec());
    hip::WaveState state;
    ctor(&state, bb);
    if (threadIdx.x % warpSize == thread) {
        auto* bitcast = reinterpret_cast<hip::WaveState*>(storage);
        bitcast[*idx] = state;
        // Incrementing idx is now left to the instrumenter as it may be stored
        // in a SGPR
    }
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
    uint32_t* private_ptr = nullptr;
    _hip_store_ctr(nullptr, 0, nullptr);
    _hip_get_trace_offset(nullptr, nullptr, 0);
    _hip_create_event(nullptr, private_ptr, 0, nullptr, 0);
    _hip_create_wave_event(nullptr, private_ptr, 0, nullptr, 0);
    _hip_event_ctor(nullptr, 0);
    _hip_tagged_event_ctor(nullptr, 0);
    _hip_wavestate_ctor(nullptr, 0);
}

} // namespace
