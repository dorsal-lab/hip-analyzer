/** \file wavestate_asm.cpp
 * \brief Test wavestate handwritten asm event creation
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/gpu_queue.hpp"

#include "../src/pass/gpu_pass_instr.hip"

#include <cstdint>
#include <iostream>

__global__ void create_one_event(void* buffer, uint32_t bb) {
    uint64_t buff_u64 = reinterpret_cast<uint64_t>(buffer);
    uint32_t lsb = buff_u64 & 0xffffffff;
    uint32_t msb = buff_u64 >> 32u;

    asm volatile("v_readfirstlane_b32 s20, %0\n" // lsb ptr
                 "v_readfirstlane_b32 s21, %1\n" // msb ptr

                 "s_memrealtime s[24:25]\n"                // timestamp
                 "s_mov_b64 s[26:27], exec\n"              // exec mask
                 "s_getreg_b32 s28, hwreg(HW_REG_HW_ID)\n" // hw_id
                 "v_readfirstlane_b32 s29, %2\n"           // bb
                 // Write to mem
                 "s_store_dwordx4 s[24:27], s[20:21]\n"
                 "s_store_dwordx2 s[28:29], s[20:21], 16\n"

                 "s_waitcnt lgkmcnt(0)\n"
                 "s_dcache_wb\n"
                 :
                 : "v"(lsb), "v"(msb), "v"(bb)
                 : "s24", "s25", "s26", "s27", "s28", "s29");
}

bool test_n_threads(size_t n, uint32_t bb) {
    std::cout << "Testing n_threads : " << n << ", " << bb << '\n';

    hip::WaveState* wavestate;
    hip::check(hipMalloc(&wavestate, sizeof(hip::WaveState)));
    hip::check(hipMemset(wavestate, 0x00, sizeof(hip::WaveState)));

    create_one_event<<<1, n>>>(wavestate, bb);
    hip::check(hipDeviceSynchronize());

    uint8_t buffer[sizeof(hip::WaveState)];
    hip::check(hipMemcpy(buffer, wavestate, sizeof(hip::WaveState),
                         hipMemcpyDeviceToHost));

    auto* ptr = reinterpret_cast<hip::WaveState*>(buffer);

    std::cout << "Wavestate :\n\tbb : " << ptr->bb
              << "\n\tStamp : " << ptr->stamp << "\n\tExec : " << ptr->exec
              << "\n\thw_id : " << ptr->hw_id << "\n\n";

    hip::check(hipFree(wavestate));

    return std::popcount(ptr->exec) == n && ptr->bb == bb;
}

namespace hip {
inline void assertC(bool cond, const std::string& message) {
    if (!cond) {
        throw std::runtime_error("Failed assertion : " + message);
    }
}
} // namespace hip

int main() {
    hip::assertC(test_n_threads(1, 0), "1 thread");
    hip::assertC(test_n_threads(2, 0), "2 threads");
    hip::assertC(test_n_threads(6, 0), "6 threads");
    hip::assertC(test_n_threads(64, 0), "64 threads");

    return 0;
}
