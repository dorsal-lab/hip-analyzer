/** \file gpu_queue.cpp
 * \brief Device-side fixed-size queue test
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/gpu_queue.hpp"
#include "hip_instrumentation/hip_utils.hpp"

#include "hip/hip_runtime.h"

struct Event {
    unsigned int value;
};

constexpr auto NB_ELEMENTS = 8u;

__global__ void enqueue_kernel(Event* e, size_t* offsets) {
    hip::ThreadQueue<Event> queue{e, offsets};

    for (auto i = 0u; i < NB_ELEMENTS; ++i) {
        queue.push_back({i});
    }
}

int main() {
    hip::init();

    Event* storage;
    hip::check(hipMalloc(&storage, sizeof(Event) * NB_ELEMENTS));

    size_t* offsets;
    hip::check(hipMalloc(&offsets, sizeof(size_t) * NB_ELEMENTS));
    hip::check(hipMemset(offsets, 0u, sizeof(size_t) * NB_ELEMENTS));

    enqueue_kernel<<<1, 1>>>(storage, offsets);
    hip::check(hipDeviceSynchronize());

    std::vector<Event> events_cpu(NB_ELEMENTS, {0});

    hip::check(hipMemcpy(events_cpu.data(), storage,
                         sizeof(Event) * NB_ELEMENTS, hipMemcpyDeviceToHost));

    for (auto& e : events_cpu) {
        std::cout << e.value << '\n';
    }

    return 0;
}
