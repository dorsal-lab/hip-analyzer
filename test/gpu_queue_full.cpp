/** \file gpu_queue_full.cpp
 * \brief Full test of the gpu queue
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip/hip_runtime.h"

#include "hip_instrumentation/gpu_queue.hpp"

struct TestEvent {
    char c;

    static std::string description;
    static std::string name;
};

std::string TestEvent::description =
    hip::HipEventFields<decltype(TestEvent::c)>();

std::string TestEvent::name = "TestEvent";

constexpr auto NB_ELEMENTS = 8u;

__global__ void fill_counters(uint8_t* counters) {
    counters[threadIdx.x] = NB_ELEMENTS;
}

__global__ void enqueue(TestEvent* storage, size_t* offsets) {
    hip::ThreadQueue<TestEvent> queue{storage, offsets};

    for (auto i = 0u; i < NB_ELEMENTS; ++i) {
        queue.push_back({static_cast<char>('a' + i)});
    }
}

int main() {
    hip::init();

    constexpr auto blocks = 1u;
    constexpr auto threads = 64u;

    hip::KernelInfo ki{"", 1, blocks, threads};
    hip::Instrumenter instr{ki};

    auto gpu_counters = instr.toDevice();

    fill_counters<<<blocks, threads>>>(gpu_counters);

    hip::check(hipDeviceSynchronize());

    instr.fromDevice(gpu_counters);
    hip::check(hipFree(gpu_counters));

    auto queue_cpu = hip::QueueInfo::thread<TestEvent>(instr);
    auto storage = queue_cpu.allocBuffer<TestEvent>();
    auto offsets = queue_cpu.allocOffsets();

    instr.record();

    enqueue<<<blocks, threads>>>(storage, offsets);
    hip::check(hipDeviceSynchronize());

    queue_cpu.fromDevice(storage);
    queue_cpu.record();

    hip::check(hipFree(storage));
    hip::check(hipFree(offsets));

    return 0;
}
