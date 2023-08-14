/** \file wave_id.cpp
 * \brief Test wave id computation
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/gpu_queue.hpp"

#include "../src/pass/gpu_pass_instr.hip"

#include <chrono>
#include <iostream>
#include <thread>

constexpr uint32_t wavefronts = 64;

__global__ void store_waves(uint32_t* wave_ids) {
    if (threadIdx.x % warpSize == 0) {
        auto wave = _hip_wave_id_1d();
        printf("Wave %d\n", wave);
        wave_ids[wave] = wave;
    }
}

int main() {
    uint32_t* wave_ids;
    constexpr size_t alloc_size = wavefronts * sizeof(uint32_t);

    hip::check(hipMalloc(&wave_ids, wavefronts * sizeof(uint32_t)));
    hip::check(hipMemset(wave_ids, 0, alloc_size));

    store_waves<<<1, 64 * wavefronts>>>(wave_ids);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    hip::check(hipDeviceSynchronize());

    uint32_t buffer[wavefronts];
    hip::check(hipMemcpy(buffer, wave_ids, alloc_size, hipMemcpyDeviceToHost));

    for (auto i = 0u; i < wavefronts; ++i) {
        std::cout << i << ' ' << buffer[i] << '\n';
    }
}
