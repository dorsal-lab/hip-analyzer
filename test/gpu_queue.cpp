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

__global__ void enqueue_kernel(Event* e) {
    hip::WaveQueue<Event> queue{e};

    for (auto i = 0u; i < NB_ELEMENTS; ++i) {
        queue.push_back({i});
    }
}

int main() {
    hip::init();
    return 0;
}
