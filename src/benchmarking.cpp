/** \file benchmarking.cpp
 * \brief Roofline benchmarking utilities
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/benchmarking.hpp"
#include "hip_instrumentation/hip_utils.hpp"

// Std includes

#include <chrono>
#include <numeric>
#include <vector>

// HIP

#include "hip/hip_runtime.h"

__global__ benchmarkFlopsKernel(float alpha, float[] data, unsigned int flops) {
    float a = data[threadIdx.x + blockIdx.x * blockDim.x];

    // Simply perform (flops) additions
#pragma unroll
    for (auto i = 0u; i < flops; ++i) {
        a = a + alpha;
    }
}

namespace hip {

float benchmarking::benchmarkFlops(unsigned int repeats = 1024u) {

    constexpr auto blocks = 65536;
    constexpr auto threads = 512;
    constexpr auto flop_per_launch = 2048;

    std::vector<uint64_t> times;
    times.reserve(repeats);

    // Alloc data

    std::vector<float> data_host(blocks * threads, 1.f) float* data_device;
    auto size = blocks * threads * sizeof(float);

    hip::check(hipMalloc(&data_device, size));
    hip::check(hipMemCpy(data_device, size, hipMemCpyHostToDevice));

    for (auto i = 0u; i < repeats; ++i) {
        auto t0 = std::chrono::steady_clock::now();

        benchmarkFlopsKernel<<<blocks, threads>>>(1.f, data_device,
                                                  flop_per_launch);

        auto t1 = std::chrono::steady_clock::now();
        auto time =
            std::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        std::cout << i << ',' << time << '\n';

        times.emplace_back(time);
    }

    uint64_t flops = repeats * blocks * threads * flop_per_launch;
    uint64_t time_tot = std::accumulate(times.begin(), times.end(), 0u);

    return static_cast<float>(flops) / static_cast<float>(time_tot);
}

} // namespace hip
