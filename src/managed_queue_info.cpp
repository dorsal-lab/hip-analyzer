/** \file managed_queue_info.cpp
 * \brief GPU Queue host-side handler for dynamic memory tracing
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/managed_queue_info.hpp"
#include "hip_instrumentation/hip_trace_manager.hpp"
#include <cstddef>
#include <memory>
#include <ostream>

namespace hip {

GlobalMemoryQueueInfo::GlobalMemoryQueueInfo(size_t elem_size,
                                             size_t buffer_size)
    : elem_size(elem_size) {
    // Get the timestamp for unique identification
    auto now = std::chrono::steady_clock::now();
    stamp = std::chrono::duration_cast<std::chrono::microseconds>(
                now.time_since_epoch())
                .count();

    cpu_queue.resize(buffer_size);
}

GlobalMemoryQueueInfo::GlobalMemoryTrace* GlobalMemoryQueueInfo::toDevice() {
    GlobalMemoryTrace* device_ptr;
    hip::check(hipMalloc(&device_ptr, sizeof(GlobalMemoryTrace)));

    hip::check(hipMalloc(&cpu_trace.current, cpu_queue.size()));
    cpu_trace.end =
        reinterpret_cast<std::byte*>(cpu_trace.current) + cpu_queue.size();

    hip::check(hipMemcpy(device_ptr, &cpu_trace, sizeof(GlobalMemoryTrace),
                         hipMemcpyHostToDevice));

    // std::cerr << "GlobalMemoryQueueInfo : " << device_ptr << ' '
    //           << cpu_trace.current << '\n';

    return device_ptr;
}

void GlobalMemoryQueueInfo::fromDevice(
    GlobalMemoryQueueInfo::GlobalMemoryTrace* device_ptr) {
    GlobalMemoryTrace gpu_trace;

    hip::check(hipMemcpy(&gpu_trace, device_ptr, sizeof(GlobalMemoryTrace),
                         hipMemcpyDeviceToHost));

    auto size = reinterpret_cast<std::byte*>(gpu_trace.current) -
                reinterpret_cast<std::byte*>(cpu_trace.current);

    // std::cerr << "Size : " << size << '\n';

    cpu_queue.resize(size);
    hip::check(hipMemcpy(cpu_queue.data(), cpu_trace.current, size,
                         hipMemcpyDeviceToHost));

    cpu_trace.end = gpu_trace.current;

    hip::check(hipFree(device_ptr));
    hip::check(hipFree(cpu_trace.current));
}

void GlobalMemoryQueueInfo::record(
    GlobalMemoryQueueInfo::GlobalMemoryTrace* device_ptr) {
    hip::HipTraceManager::getInstance().registerGlobalMemoryQueue(*this,
                                                                  device_ptr);
}

// ----- ChunkAllocator ----- //

ChunkAllocator::ChunkAllocator(size_t buffer_count, size_t buffer_size)
    : last_registry{buffer_count, buffer_size, nullptr, 0ull} {
    size_t alloc_size = buffer_size * buffer_count;

    hip::check(hipMalloc(&buffer_ptr, alloc_size));
    last_registry.begin = buffer_ptr;

    hip::check(hipMalloc(&device_ptr, sizeof(Registry)));

    hip::check(hipMemcpy(device_ptr, &last_registry, sizeof(Registry),
                         hipMemcpyHostToDevice));
}

ChunkAllocator::~ChunkAllocator() {
    hip::check(hipFree(device_ptr));
    hip::check(hipFree(buffer_ptr));
}

void ChunkAllocator::update() {
    hip::check(hipMemcpy(&last_registry, device_ptr, sizeof(Registry),
                         hipMemcpyDeviceToHost));
}

void ChunkAllocator::record() {
    hip::check(hipDeviceSynchronize());

    size_t begin_id = last_registry.current_id;

    update();

    std::cerr << "Queue: " << begin_id << ", " << last_registry.current_id
              << '\n';
}

std::unique_ptr<std::byte[]> ChunkAllocator::copyBuffer() {
    update();
    size_t alloc_size = last_registry.buffer_size * last_registry.buffer_count;
    auto buf = std::make_unique<std::byte[]>(alloc_size);

    hip::check(
        hipMemcpy(buf.get(), buffer_ptr, alloc_size, hipMemcpyDeviceToHost));

    return buf;
}

std::ostream& ChunkAllocator::printBuffer(std::ostream& out) {
    update();
    auto contents = copyBuffer();

    return last_registry.printBuffer(
        out, reinterpret_cast<SubBuffer*>(contents.get()));
}

std::ostream& ChunkAllocator::Registry::printBuffer(std::ostream& out,
                                                    SubBuffer* sbs) {
    std::cout << "ChunkAllocator " << buffer_count << " SubBuffers of "
              << buffer_size << " bytes\n";

    for (auto i = 0ul; i < buffer_count; ++i) {
        if (i == (current_id % buffer_count)) {
            out << "=>";
        } else {
            out << "  ";
        }

        std::byte* ptr = reinterpret_cast<std::byte*>(sbs) + buffer_size * i;
        SubBuffer* sb = reinterpret_cast<SubBuffer*>(ptr);

        out << "SubBuffer " << i << ", owner : " << sb->owner << "\n    "
            << std::hex;
        for (auto j = 0; j < buffer_size - offsetof(SubBuffer, data); ++j) {
            out << "0x" << static_cast<unsigned int>(sb->data[j]) << ' ';
        }
        out << std::dec << '\n';
    }

    return out;
}

} // namespace hip
