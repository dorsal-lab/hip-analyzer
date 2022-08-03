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

template <class Queue>
__global__ void enqueue_kernel(Event* e, size_t* offsets) {
    Queue queue{e, offsets};

    for (auto i = 0u; i < NB_ELEMENTS; ++i) {
        Event e{i};
        queue.push_back(e);
        // printf("%d\n", e.value);
    }
}

constexpr auto NB_THREADS = 64u;

int main() {
    hip::init();

    std::vector<size_t> offsets(NB_THREADS, 0u);
    for (auto i = 0u; i < NB_THREADS; ++i) {
        offsets[i] = NB_ELEMENTS * i;
    }

    // ThreadQueue test

    Event* storage;
    hip::check(hipMalloc(&storage, sizeof(Event) * NB_ELEMENTS * NB_THREADS));

    size_t* offsets_gpu;
    hip::check(hipMalloc(&offsets_gpu, sizeof(size_t) * NB_THREADS));

    hip::check(hipMemcpy(offsets_gpu, offsets.data(),
                         sizeof(size_t) * NB_THREADS, hipMemcpyHostToDevice));

    enqueue_kernel<hip::ThreadQueue<Event>>
        <<<1, NB_THREADS>>>(storage, offsets_gpu);
    hip::check(hipDeviceSynchronize());

    std::vector<Event> events_cpu(NB_ELEMENTS * NB_THREADS, {0});

    hip::check(hipMemcpy(events_cpu.data(), storage,
                         sizeof(Event) * NB_ELEMENTS * NB_THREADS,
                         hipMemcpyDeviceToHost));

    for (auto& e : events_cpu) {
        std::cout << e.value << '\n';
    }

    // WaveQueue test

    hip::check(
        hipMemset(storage, 0u, sizeof(Event) * NB_ELEMENTS * NB_THREADS));

    // Basically run the same test, but with more blocks as to create more
    // wavefronts
    enqueue_kernel<hip::WaveQueue<Event>>
        <<<NB_THREADS, NB_THREADS>>>(storage, offsets_gpu);

    hip::check(hipMemcpy(events_cpu.data(), storage,
                         sizeof(Event) * NB_ELEMENTS * NB_THREADS,
                         hipMemcpyDeviceToHost));

    for (auto& e : events_cpu) {
        std::cout << e.value << '\n';
    }

    return 0;
}
