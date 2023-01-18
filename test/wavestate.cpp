/** \file wavestate.cpp
 * \brief Test wavestate event creation
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/gpu_queue.hpp"

// It's bad, I know, but no LTO for the device otherwise and time is getting
// tight
#include "../src/pass/gpu_pass_instr.cpp"

#include <iostream>

__global__ void create_one_event(void* buffer, size_t bb) {
    size_t idx = 0;

    _hip_create_wave_event(buffer, &idx, sizeof(hip::WaveState),
                           _hip_wavestate_ctor, bb);
}

bool test_n_threads(size_t n, size_t bb) {
    std::cout << "Testing n_threads : " << n << ", " << bb << '\n';

    hip::WaveState* wavestate;
    hip::check(hipMalloc(&wavestate, sizeof(hip::WaveState)));

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
}
