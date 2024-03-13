/** \file managed_queue_info.cpp
 * \brief GPU Queue host-side handler for dynamic memory tracing
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include <bit>
#include <chrono>
#include <cstddef>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <thread>

#include "hip_instrumentation/hip_trace_manager.hpp"
#include "hip_instrumentation/managed_queue_info.hpp"
#include "hip_instrumentation/state_recoverer.hpp"

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

ChunkAllocator::ChunkAllocator(size_t buffer_count, size_t buffer_size,
                               bool alloc_gpu)
    : last_registry{buffer_count, buffer_size, nullptr, 0ull} {

    if (std::popcount(buffer_count) != 1) {
        throw std::runtime_error("ChunkAllocator::ChunckAllocator() : "
                                 "buffer_count must be a power of two");
    }

    size_t alloc_size = buffer_size * buffer_count;

    if (alloc_gpu) {
        hip::check(hipMalloc(&buffer_ptr, alloc_size));
        last_registry.begin = buffer_ptr;

        hip::check(hipMalloc(&device_ptr, sizeof(Registry)));

        hip::check(hipMemcpy(device_ptr, &last_registry, sizeof(Registry),
                             hipMemcpyHostToDevice));
    }
}

ChunkAllocator::~ChunkAllocator() {
    while (process_count > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
    hip::check(hipFree(device_ptr));
    hip::check(hipFree(buffer_ptr));
}

void ChunkAllocator::update() {
    Registry old{last_registry};

    hip::check(hipMemcpy(&last_registry, device_ptr, sizeof(Registry),
                         hipMemcpyDeviceToHost));

    auto diff = last_registry.current_id - old.current_id;
    if (diff > old.buffer_count) {
        std::cout << "ChunkAllocator::update() : Overflow detected\n";
    }
}

void ChunkAllocator::record(uint64_t stamp) {
    hip::check(hipDeviceSynchronize());

    size_t begin_id = last_registry.current_id;

    update();

    ++process_count;
    hip::HipTraceManager::getInstance().registerChunkAllocatorEvents(
        this, stamp, last_registry, begin_id);
}

std::unique_ptr<std::byte[]> ChunkAllocator::copyBuffer() {
    update();
    size_t alloc_size = last_registry.buffer_size * last_registry.buffer_count;
    auto buf = std::make_unique<std::byte[]>(alloc_size);

    hip::check(
        hipMemcpy(buf.get(), buffer_ptr, alloc_size, hipMemcpyDeviceToHost));

    return buf;
}

std::unique_ptr<std::byte[]> ChunkAllocator::slice(size_t begin, size_t end) {
    // No update needed
    auto& reg = last_registry;

    size_t size = end - begin;

    if (size > reg.buffer_count) {
        throw std::runtime_error("ChunkAllocator::slice() : Requesting a slice "
                                 "larger than buffer_count");
    }

    auto buf = std::make_unique<std::byte[]>(size * reg.buffer_size);

    size_t begin_wrap = begin % reg.buffer_count;
    size_t end_wrap = end % reg.buffer_count;

    std::byte* begin_ptr =
        reinterpret_cast<std::byte*>(reg.begin) + begin_wrap * reg.buffer_size;

    if (begin_wrap <= end_wrap) {
        // Easiest : contiguous copy

        hip::check(hipMemcpy(buf.get(), begin_ptr, size * reg.buffer_size,
                             hipMemcpyDeviceToHost));
    } else {
        // Have to do two copies
        size_t size_slice = (reg.buffer_count - begin_wrap) * reg.buffer_size;

        hip::check(
            hipMemcpy(buf.get(), begin_ptr, size_slice, hipMemcpyDeviceToHost));

        hip::check(hipMemcpy(buf.get() + size_slice, reg.begin,
                             end_wrap * reg.buffer_size,
                             hipMemcpyDeviceToHost));
    }

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
        for (auto j = 0u; j < buffer_size - offsetof(SubBuffer, data); ++j) {
            out << "0x" << static_cast<unsigned int>(sb->data[j]) << ' ';
        }
        out << std::dec << '\n';
    }

    return out;
}

const std::string& hip::ChunkAllocator::event_desc =
    hip::WaveState::description;
const std::string& hip::ChunkAllocator::event_name = hip::WaveState::name;

ChunkAllocator* ChunkAllocator::getStreamAllocator(hipStream_t stream,
                                                   size_t buffer_count,
                                                   size_t buffer_size) {
    auto& allocators = HipMemoryManager::getInstance().allocators();
    allocators.try_emplace(stream, buffer_count, buffer_size);

    return &allocators.at(stream);
}

// ----- CUChunkAllocator ----- //

CUChunkAllocator::CUChunkAllocator(size_t buffer_count, size_t buffer_size,
                                   bool alloc_gpu)
    : last_registry{{{buffer_count, buffer_size, nullptr, 0ull}}},
      buffer_count(buffer_count), buffer_size(buffer_size) {

    if (std::popcount(buffer_count) != 1) {
        throw std::runtime_error("ChunkAllocator::ChunckAllocator() : "
                                 "buffer_count must be a power of two");
    }

    size_t alloc_size = buffer_size * buffer_count * TOTAL_CU_COUNT;

    if (alloc_gpu) {
        hip::check(hipMalloc(&buffer_ptr, alloc_size));

        for (auto i = 0u; i < TOTAL_CU_COUNT; ++i) {
            last_registry[i].reg.begin =
                buffer_ptr + i * (buffer_size * buffer_count);
        }

        hip::check(hipMalloc(&device_ptr,
                             sizeof(CacheAlignedRegistry) * TOTAL_CU_COUNT));

        hip::check(hipMemcpy(device_ptr, last_registry.data(),
                             sizeof(CacheAlignedRegistry) * TOTAL_CU_COUNT,
                             hipMemcpyHostToDevice));
    }
}

} // namespace hip
