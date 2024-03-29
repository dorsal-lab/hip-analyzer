/** \file gpu_queue_wave.cpp
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
        queue.emplace_back(TestEvent{static_cast<char>('a' + i)});
    }
}

int main() {
    hip::init();

    constexpr auto blocks = 1u;
    constexpr auto threads = 64u;

    hip::KernelInfo ki{"", 1, blocks, threads};
    hip::ThreadCounterInstrumenter instr{ki};

    auto gpu_counters = instr.toDevice();

    fill_counters<<<blocks, threads>>>(
        reinterpret_cast<uint8_t*>(gpu_counters));

    hip::check(hipDeviceSynchronize());

    instr.fromDevice(gpu_counters);
    hip::check(hipFree(gpu_counters));

    auto queue_cpu = hip::QueueInfo::thread<TestEvent>(instr);
    auto storage = queue_cpu.allocBuffer<TestEvent>();
    auto offsets = queue_cpu.allocOffsets();

    instr.record();

    enqueue<<<blocks, threads>>>(storage, offsets);
    hip::check(hipDeviceSynchronize());

    queue_cpu.record(storage);

    return 0;
}
