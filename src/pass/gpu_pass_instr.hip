/** \file gpu_pass_instr.cpp
 * \brief Device-side utility functions
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip/hip_runtime.h"

#include "hip_instrumentation/gpu_queue.hpp"
#include "hip_instrumentation/managed_queue_info.hpp"

extern "C" {

// ----- Utils ----- //

__device__ uint32_t _hip_flat_size() {
    return blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z;
}

__device__ uint32_t _hip_flat_block_id() {
    return blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
}

__device__ uint32_t _hip_flat_thread_id() {
    return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
}

__device__ uint32_t _hip_flat_block_size() {
    return blockDim.x * blockDim.y * blockDim.z;
}

__device__ uint32_t _hip_waves_per_block() {
    auto block_size = _hip_flat_block_size();
    uint32_t waves_per_block = block_size / warpSize;
    if (block_size % warpSize != 0) {
        ++waves_per_block;
    }

    return waves_per_block;
}

__device__ uint32_t _hip_wave_id_1d() {
    auto block_id = _hip_flat_block_id();
    auto waves_per_block = _hip_waves_per_block();
    auto wave_in_block = _hip_flat_thread_id() / warpSize;

    return block_id * waves_per_block + wave_in_block;
}

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

/** \fn _hip_wave_ctr_get_offset
 * \brief Returns the counter base ptr for the wave
 */
__device__ void* _hip_wave_ctr_get_offset(uint32_t* base, uint32_t bb_count) {
    return &base[_hip_wave_id_1d()];
}

// ----- Tracing instrumentation ----- //

using offset_getter = void*(*)(void*, size_t*, size_t);

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
    uint32_t wavefront_id = _hip_wave_id_1d();

    auto offset = offsets[wavefront_id];

    // We are going to calculate a byte offset, cast to byte-sized (nice pun)
    // array
    auto* bitcast = reinterpret_cast<uint8_t*>(storage);

    return &bitcast[offset * event_size];
}

__device__ void* _hip_get_global_memory_trace_ptr(
    hip::GlobalMemoryQueueInfo::Registry* device_ptr) {
    return device_ptr;
}

__device__ hip::ChunkAllocator::Registry* _hip_get_cache_aligned_registry(hip::CUChunkAllocator::CacheAlignedRegistry* reg) {
    auto cu_id = hip::gcnasm::get_cu_id();

    return &(reg[cu_id].reg);
}

// Generic function pointer to : void event_ctor(void* storage, size_t bb);
using event_ctor = void (*)(void*, uint32_t);

__device__ void _hip_create_event(void* storage, uint32_t* idx,
                                  size_t event_size, event_ctor ctor,
                                  uint32_t bb) {
    auto* bitcast = reinterpret_cast<uint8_t*>(storage);

    ctor(&bitcast[*idx * event_size], bb);
}

__device__ size_t* _hip_wave_get_index_in_block(size_t* idx_array) {
    auto wave_in_block = threadIdx.x / warpSize;
    return &idx_array[wave_in_block];
}

__device__ void _hip_create_wave_event(void* storage, uint32_t* idx,
                                       size_t event_size, event_ctor ctor,
                                       uint32_t bb) {
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

 __attribute__((naked)) __device__ void _hip_chunk_allocator_alloc(){
    // Input parameters :
    // - s[44:45] : pointer to registry
    // - s40 : producer_id
    // - s41 : event_size - 1
    // - s[46:47] : return address
    asm volatile(
        // New allocation (ChunkAllocator::Registry::alloc)
        //// Atomic add, load values
        "s_mov_b64 s[28:29], 1\n"
        "s_atomic_add_x2 s[28:29], s[44:45], 24 glc\n"
        "s_load_dwordx2 s[22:23], s[44:45], 0\n"  // Load buffer_count
        "s_load_dwordx2 s[24:25], s[44:45], 8\n"  // Load buffer_size
        "s_load_dwordx2 s[26:27], s[44:45], 16\n" // Load begin
        "s_waitcnt lgkmcnt(0)\n"
        //// Compute new ptr, ptr_end
        ///// next %= buffer_count. Assume buffer_count is a power of two, so
        /// just retrieve the `buffer_count` lsb from s[28:29]
        "s_ff1_i32_b64 s22, s[22:23]\n" // log2(s[28:29])
        "s_add_u32 s22, s22, 1\n"
        "s_bfm_b64 s[22:23], s22, 0\n"
        "s_and_b64 s[28:29], s[28:29], s[22:23]\n"

        ///// next *= buffer_size (multiply s[28:29] * s[24:25], intermediate
        /// result in s[22:23])
        "s_mul_hi_u32 s22, s29, s24\n"
        "s_mul_hi_u32 s23, s28, s25\n"
        "s_mul_i32 s28, s28, s24\n"
        "s_add_u32 s29, s22, s23\n"
        ///// ptr = begin + next
        "s_add_u32 s22, s26, s28\n"
        "s_addc_u32 s23, s27, s29\n"
        // Store producer id
        "s_mov_b32 s30, s40\n"
        "s_sub_u32 s24, s24, s41\n" // Reasonable to expect event_size <<< 2^32, so no carry
        "s_store_dwordx2 s[30:31], s[22:23]\n"
        "s_add_u32 s42, s22, s24\n"  // ptr_end_lo
        "s_addc_u32 s43, s23, s25\n" // ptr_end_hi
        "s_add_u32 s40, s22, 8\n"    // ptr_lo
        "s_addc_u32 s41, s23, 0\n"   // ptr_hi
        "s_setpc_b64 s[46:47]\n"
    );
}

// ----- Events constructors ----- //

__device__ void _hip_event_ctor(void* buf, uint32_t bb) {
    new (buf) hip::Event{bb};
}

__device__ void _hip_tagged_event_ctor(void* buf, uint32_t bb) {
    new (buf) hip::TaggedEvent{bb};
}

__device__ void _hip_wavestate_ctor(void* buf, uint32_t bb) {
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
