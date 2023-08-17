/** \file get_wave_trace_offset.cpp
 * \brief Test offset getter
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/gpu_queue.hpp"

#include "../src/pass/gpu_pass_instr.hip"

#include <iostream>
#include <vector>

__global__ void create_one_event(void* buffer, size_t* offsets) {
    uint32_t idx = 0u;
    void* sub_buffer = _hip_get_wave_trace_offset(buffer, offsets, 24ul);

    // We'll store the wave id in the 'bb' field

    auto wave_id = _hip_wave_id_1d();

    _hip_create_wave_event(sub_buffer, &idx, sizeof(hip::WaveState),
                           _hip_wavestate_ctor, wave_id);
}

bool test_n_waves(dim3 blocks, dim3 threads, size_t stride = 1ul) {
    auto waves_per_block = threads.x / 64u;
    if (threads.x % 64u != 0) {
        ++waves_per_block;
    }

    auto total_waves = waves_per_block * blocks.x;

    std::cout << "Testing  : " << blocks.x << ", " << threads.x << " ("
              << total_waves << " waves)\n";

    // Allocate buffer
    size_t buf_size = stride * total_waves * sizeof(hip::WaveState);
    hip::WaveState* wavestate;
    hip::check(hipMalloc(&wavestate, buf_size));

    // Allocate offsets & compute
    size_t* offsets_gpu;
    hip::check(hipMalloc(&offsets_gpu, total_waves * sizeof(size_t)));

    std::vector<size_t> offsets;
    for (auto i = 0ul; i < total_waves; ++i) {
        offsets.push_back(i * stride);
    }

    hip::check(hipMemcpy(offsets_gpu, offsets.data(),
                         total_waves * sizeof(size_t), hipMemcpyHostToDevice));

    // Launch Kernel

    create_one_event<<<blocks, threads>>>(wavestate, offsets_gpu);
    hip::check(hipDeviceSynchronize());

    // Fetch back data & check

    std::vector<uint8_t> buffer(buf_size, '\0');
    hip::check(
        hipMemcpy(buffer.data(), wavestate, buf_size, hipMemcpyDeviceToHost));

    auto* ptr = reinterpret_cast<hip::WaveState*>(buffer.data());

    hip::check(hipFree(wavestate));

    for (auto i = 0u; i < total_waves; ++i) {
        auto& event = ptr[i * stride];
        std::cout << "Wavestate :\n\tbb : " << event.bb
                  << "\n\tStamp : " << event.stamp
                  << "\n\tExec : " << event.exec
                  << "\n\thw_id : " << event.hw_id << "\n\n";

        if (event.bb != i) {
            std::cerr << "Event " << i * stride << ", expected " << i
                      << " but got " << event.bb << '\n';
            return false;
        }
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
    // Test hip::WaveState size
    hip::assertC(sizeof(hip::WaveState) == 24, "hip::WaveState size");

    // Test fetching offsets & creating event
    std::cout << "Stride 1\n";
    hip::assertC(test_n_waves(1, 64), "1 wave");
    hip::assertC(test_n_waves(1, 128), "2 waves");
    hip::assertC(test_n_waves(2, 64), "2 waves");
    hip::assertC(test_n_waves(2, 128), "4 waves");

    std::cout << "Stride 2\n";
    hip::assertC(test_n_waves(1, 64, 2), "1 wave");
    hip::assertC(test_n_waves(1, 128, 2), "2 waves");
    hip::assertC(test_n_waves(2, 64, 2), "2 waves");
    hip::assertC(test_n_waves(2, 128, 2), "4 waves");

    return 0;
}
