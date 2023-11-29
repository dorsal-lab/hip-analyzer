/** \file global_atomics_tracing.cpp
 * \brief Test single buffer tracing
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip/hip_runtime.h"
#include "hip_instrumentation/hip_utils.hpp"

#include "hip_instrumentation/gpu_queue.hpp"
#include "hip_instrumentation/managed_queue_info.hpp"

#include "../src/pass/gpu_pass_instr.hip"

#include <iostream>

__global__ void
create_n_events(hip::GlobalMemoryQueueInfo::GlobalMemoryTrace* buffer,
                size_t n) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    auto* trace_pointer = _hip_get_global_memory_trace_ptr(buffer);

    uint64_t trace_ptr_u64 = reinterpret_cast<uint64_t>(trace_pointer);
    uint32_t lsb = trace_ptr_u64 & 0xffffffff;
    uint32_t msb = trace_ptr_u64 >> 32u;

    asm volatile("v_readfirstlane_b32 s40, %0" : : "v"(lsb));
    asm volatile("v_readfirstlane_b32 s41, %0" : : "v"(msb));

    for (uint32_t i = 0u; i < n; ++i) {
        asm volatile(                                  // Prepare payload
            "s_atomic_add_x2 s[22:23], s[40:41], %0\n" // Atomically increment
                                                       // the global trace
                                                       // pointer
            "s_memrealtime s[24:25]\n"                 // timestamp
            "s_mov_b64 s[26:27], exec\n"               // exec mask
            "s_getreg_b32 s28, hwreg(HW_REG_HW_ID)\n"  // hw_id
            "s_mov_b32 s29, %1\n"                      // bb
            "v_readfirstlane_b32 s30, %2\n" // s31 will be stored as well but
                                            // that's not an issue (just ignore
                                            // in the trace). Not clobbered
            "s_waitcnt lgkmcnt(0)\n"
            // Write to mem
            "s_store_dwordx4 s[24:27], s[22:23], 0\n"
            "s_store_dwordx4 s[28:31], s[22:23], 16\n"
            "s_waitcnt lgkmcnt(0)\n"
            :
            : "i"(28), "s"(i), "s"(tid)
            : "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s30");
    }
}

bool test_n_threads(size_t n) {
    std::cout << "Testing n_threads : " << n << '\n';

    hip::GlobalMemoryQueueInfo queue(sizeof(hip::GlobalWaveState));
    auto* device_ptr = queue.toDevice();

    constexpr auto n_iter = 8u;
    create_n_events<<<1, n>>>(device_ptr, n_iter);

    hip::check(hipDeviceSynchronize());

    queue.fromDevice(device_ptr);

    const hip::GlobalWaveState* cpu_queue =
        reinterpret_cast<const hip::GlobalWaveState*>(queue.buffer().data());

    for (auto i = 0; i < queue.queueLength(); ++i) {
        const hip::GlobalWaveState& e = cpu_queue[i];
        std::cout << e.producer << " : " << e.bb << '\n';
    }

    return true;
}

namespace hip {
inline void assertC(bool cond, const std::string& message) {
    if (!cond) {
        throw std::runtime_error("Failed assertion : " + message);
    }
}
} // namespace hip

int main() {
    hip::assertC(test_n_threads(1), "1 thread");
    hip::assertC(test_n_threads(2), "2 threads");
    hip::assertC(test_n_threads(6), "6 threads");
    hip::assertC(test_n_threads(64), "64 threads");

    return 0;
}
