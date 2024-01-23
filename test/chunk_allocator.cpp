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

__global__ void create_n_events(hip::ChunkAllocator::Registry* buffer,
                                size_t n) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    hip::ChunkAllocator::SubBuffer* sb = buffer->alloc(tid);
    size_t offset = 0ull;

    for (auto i = 0ul; i < n; ++i) {
        Event e{i, tid};
        buffer->tracepoint(&sb, &offset, reinterpret_cast<std::byte*>(&e),
                           sizeof(Event), tid);
    }
}

bool test_n_threads(size_t n) {
    std::cout << "Testing n_threads : " << n << '\n';

    hip::ChunkAllocator allocator{16ull, 8 + 16 * 4};

    auto* device_ptr = allocator.toDevice();

    constexpr auto n_iter = 8u;

    create_n_events<<<1, n>>>(device_ptr, n_iter);

    hip::check(hipDeviceSynchronize());

    allocator.record(n);
    allocator.printBuffer(std::cout);

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
