/** \file global_atomics_tracing.cpp
 * \brief Test single buffer tracing
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip/hip_runtime.h"
#include "hip_instrumentation/hip_utils.hpp"

#include "hip_instrumentation/managed_queue_info.hpp"

#include <iostream>

struct Event {
    size_t id;
    size_t producer;
};

__global__ void
create_n_events(hip::GlobalMemoryQueueInfo::GlobalMemoryTrace* buffer,
                size_t n) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (auto i = 0ul; i < n; ++n) {
        auto* ptr = reinterpret_cast<void*>(atomicAdd(
            reinterpret_cast<size_t*>(buffer->current), sizeof(Event)));

        Event* new_event = reinterpret_cast<Event*>(ptr);
        new_event->id = i;
        new_event->producer = tid;
    }
}

bool test_n_threads(size_t n) {
    std::cout << "Testing n_threads : " << n << '\n';

    hip::GlobalMemoryQueueInfo queue(sizeof(Event));
    auto* device_ptr = queue.toDevice();

    constexpr auto n_iter = 8u;
    create_n_events<<<1, n>>>(device_ptr, n_iter);

    hip::check(hipDeviceSynchronize());

    queue.fromDevice(device_ptr);

    const Event* cpu_queue =
        reinterpret_cast<const Event*>(queue.buffer().data());

    for (auto i = 0; i < queue.queueLength(); ++i) {
        const Event& e = cpu_queue[i];
        std::cout << e.producer << " : " << e.id << '\n';
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
