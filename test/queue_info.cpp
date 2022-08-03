/** \file queue_info.cpp
 * \brief Test queue info class
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/gpu_queue.hpp"

#include "hip/hip_runtime.h"

using counter_t = hip::Instrumenter::counter_t;

__global__ void fill_values(counter_t* counters, size_t size, counter_t value) {
    for (auto i = 0u; i < size; ++i) {
        counters[i] = value;
    }
}

struct Event {
    unsigned int value;
};

int main() {
    constexpr auto blocks = 64u;
    constexpr auto threads = 64u;

    hip::KernelInfo ki("", 1, blocks, threads);
    hip::Instrumenter instr(ki);

    // Generate fake counters with a single kernel running on the gpu

    auto gpu_counters = instr.toDevice();

    fill_values<<<1, 1>>>(gpu_counters, ki.instr_size, 8u);
    hip::check(hipDeviceSynchronize());

    instr.fromDevice(gpu_counters);

    // Now we can generate the queue info and verify that it's correct

    auto thread_queue_info = hip::QueueInfo::thread<Event>(instr);

    auto offsets = thread_queue_info.offsets();
    for (auto& offset : offsets) {
        std::cout << offset << '\n';
    }

    std::cout << "Total queue size : " << thread_queue_info.queueSize() << '\n';

    // Free memory

    hip::check(hipFree(gpu_counters));

    return 0;
}
