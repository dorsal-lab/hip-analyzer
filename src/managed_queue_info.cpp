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
    : ChunkAllocatorBase(buffer_size, 1, true), elem_size(elem_size) {}

// void GlobalMemoryQueueInfo::record(
//     GlobalMemoryQueueInfo::GlobalMemoryTrace* device_ptr) {
//     hip::HipTraceManager::getInstance().registerGlobalMemoryQueue(*this,
//                                                                   device_ptr);
// }

// ----- ChunkAllocator ----- //

template <typename T>
ChunkAllocatorBase<T>::ChunkAllocatorBase(size_t buffer_count,
                                          size_t buffer_size, bool alloc_gpu)
    : last_registry{buffer_count, buffer_size, nullptr, 0ull} {

    if (std::popcount(buffer_count) != 1) {
        throw std::runtime_error("ChunkAllocator::ChunkAllocator() : "
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

template <typename T> ChunkAllocatorBase<T>::~ChunkAllocatorBase() {
    while (process_count > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
    hip::check(hipFree(device_ptr));
    hip::check(hipFree(buffer_ptr));
}

template <typename T> void ChunkAllocatorBase<T>::update() {
    Registry old{last_registry};

    hip::check(hipMemcpy(&last_registry, device_ptr, sizeof(Registry),
                         hipMemcpyDeviceToHost));

    auto diff = last_registry.current_id - old.current_id;
    if (diff > old.buffer_count) {
        std::cout << "ChunkAllocator::update() : Overflow detected\n";
    }
}

template <> void ChunkAllocatorBase<SubBuffer>::record(uint64_t stamp) {
    hip::check(hipDeviceSynchronize());

    size_t begin_id = last_registry.current_id;

    update();

    ++process_count;
    hip::HipTraceManager::getInstance().registerChunkAllocatorEvents(
        reinterpret_cast<ChunkAllocator*>(this), stamp, last_registry,
        begin_id);
}

template <> void ChunkAllocatorBase<std::byte>::record(uint64_t stamp) {
    hip::check(hipDeviceSynchronize());

    size_t begin_id = last_registry.current_id;

    update();

    ++process_count;
    hip::HipTraceManager::getInstance().registerGlobalMemoryQueue(
        reinterpret_cast<hip::GlobalMemoryQueueInfo*>(this), stamp,
        last_registry, begin_id);
}

template <typename T>
std::unique_ptr<std::byte[]> ChunkAllocatorBase<T>::copyBuffer() {
    update();
    size_t alloc_size = last_registry.buffer_size * last_registry.buffer_count;
    auto buf = std::make_unique<std::byte[]>(alloc_size);

    hip::check(
        hipMemcpy(buf.get(), buffer_ptr, alloc_size, hipMemcpyDeviceToHost));

    return buf;
}

template <typename T>
size_t ChunkAllocatorBase<T>::Registry::wrappedSize(size_t slice_begin,
                                                    size_t slice_end) {
    slice_begin = slice_begin % buffer_count;
    slice_end = slice_end % buffer_count;

    if (slice_begin <= slice_end) {
        return slice_end - slice_begin;
    } else {
        return slice_end + (buffer_count - slice_begin);
    }
}

template <typename T>
std::unique_ptr<std::byte[]>
ChunkAllocatorBase<T>::Registry::slice(size_t slice_begin, size_t slice_end) {
    auto data = sliceAsync(slice_begin, slice_end, 0);
    hip::check(hipStreamSynchronize(0));
    return data;
}

template <typename T>
std::unique_ptr<std::byte[]> ChunkAllocatorBase<T>::Registry::sliceAsync(
    size_t slice_begin, size_t slice_end, hipStream_t stream) {
    size_t size = wrappedSize(slice_begin, slice_end);

    if (size > buffer_count) {
        throw std::runtime_error(
            "ChunkAllocator::Registry::slice() : Requesting a slice "
            "larger than buffer_count");
    }

    auto buf = std::make_unique<std::byte[]>(size * buffer_size);

    size_t begin_wrap = slice_begin % buffer_count;
    size_t end_wrap = slice_end % buffer_count;

    std::byte* begin_ptr =
        reinterpret_cast<std::byte*>(begin) + begin_wrap * buffer_size;

    if (begin_wrap <= end_wrap) {
        // Easiest : contiguous copy

        hip::check(hipMemcpyDtoHAsync(buf.get(), begin_ptr, size * buffer_size,
                                      stream));
    } else {
        // Have to do two copies
        size_t size_slice = (buffer_count - begin_wrap) * buffer_size;

        hip::check(
            hipMemcpyDtoHAsync(buf.get(), begin_ptr, size_slice, stream));

        hip::check(hipMemcpyDtoHAsync(buf.get() + size_slice, begin,
                                      end_wrap * buffer_size, stream));
    }

    return buf;
}

template <>
std::ostream&
ChunkAllocatorBase<SubBuffer>::Registry::printBuffer(std::ostream& out,
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

std::ostream& ChunkAllocator::printBuffer(std::ostream& out) {
    update();
    auto contents = copyBuffer();

    return last_registry.printBuffer(
        out, reinterpret_cast<SubBuffer*>(contents.get()));
}

const std::string& hip::ChunkAllocator::event_desc =
    hip::WaveState::description;
const std::string& hip::ChunkAllocator::event_name = hip::WaveState::name;

const std::string& GlobalMemoryQueueInfo::event_name =
    hip::GlobalWaveState::name;
const std::string& GlobalMemoryQueueInfo::event_desc =
    hip::GlobalWaveState::description;

ChunkAllocator* ChunkAllocator::getStreamAllocator(hipStream_t stream,
                                                   size_t buffer_count,
                                                   size_t buffer_size) {
    auto& allocators = HipMemoryManager::getInstance().allocators();
    allocators.try_emplace(stream, buffer_count, buffer_size);

    return &allocators.at(stream);
}

template class ChunkAllocatorBase<SubBuffer>;
template class ChunkAllocatorBase<std::byte>;

// ----- CUChunkAllocator ----- //

extern void dryRunRegistries(CUChunkAllocator&);

CUChunkAllocator::CUChunkAllocator(size_t buffer_count, size_t buffer_size,
                                   bool alloc_gpu)
    : buffer_count(buffer_count), buffer_size(buffer_size) {

    // Initialize all registries
    last_registry.fill({{buffer_count, buffer_size, nullptr, 0ull}});

    size_t alloc_size = buffer_size * buffer_count * TOTAL_CU_COUNT;

    if (alloc_gpu) {
        hip::check(hipMalloc(&buffer_ptr, alloc_size));

        for (auto i = 0u; i < TOTAL_CU_COUNT; ++i) {
            last_registry[i].reg.begin = reinterpret_cast<SubBuffer*>(
                reinterpret_cast<std::byte*>(buffer_ptr) +
                i * (buffer_size * buffer_count));
        }

        hip::check(hipMalloc(&device_ptr,
                             sizeof(CacheAlignedRegistry) * TOTAL_CU_COUNT));

        hip::check(hipMemcpy(device_ptr, last_registry.data(),
                             sizeof(CacheAlignedRegistry) * TOTAL_CU_COUNT,
                             hipMemcpyHostToDevice));
    }

    dryRunRegistries(*this);
}

CUChunkAllocator::CUChunkAllocator(const std::vector<size_t>& buffer_count,
                                   size_t buffer_size)
    : buffer_count(-1), buffer_size(buffer_size), loaded(true) {

    // Initialize all registries

    if (buffer_count.size() != TOTAL_CU_COUNT) {
        throw std::runtime_error("CUChunkAllocator::CUChunkAllocator() : "
                                 "Unexpected number of buffer counts");
    }

    for (auto i = 0u; i < TOTAL_CU_COUNT; ++i) {
        auto* buf = reinterpret_cast<SubBuffer*>(
            new std::byte[buffer_count[i] * buffer_size]);

        last_registry[i].reg = {buffer_count[i], buffer_size, buf, 0ull};
    }
}

CUChunkAllocator::~CUChunkAllocator() {
    while (process_count > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }

    if (!loaded) {
        hip::check(hipFree(device_ptr));
        hip::check(hipFree(buffer_ptr));
    } else {
        // Must free allocated memory from registries
        for (auto& reg : last_registry) {
            delete[] reg.reg.begin;
        }
    }
}

void CUChunkAllocator::record(uint64_t stamp) {
    auto begin_registries = std::make_unique<Registries>(last_registry);

    hip::check(hipDeviceSynchronize());

    hip::check(hipMemcpy(last_registry.data(), device_ptr,
                         sizeof(CacheAlignedRegistry) * TOTAL_CU_COUNT,
                         hipMemcpyDeviceToHost));

    auto end_registries = std::make_unique<Registries>(last_registry);

    ++process_count;

    hip::HipTraceManager::getInstance().registerCUChunkAllocatorEvents(
        this, stamp, std::move(begin_registries), std::move(end_registries));
}

void CUChunkAllocator::sync() {
    while (process_count > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
}

void CUChunkAllocator::fetchBuffers(
    const Registries& begin_registries, const Registries& end_registries,
    std::array<std::unique_ptr<std::byte[]>, TOTAL_CU_COUNT>& sub_buffers_out,
    std::array<size_t, TOTAL_CU_COUNT>& sizes_out) {

    for (auto i = 0u; i < TOTAL_CU_COUNT; ++i) {
        auto old = begin_registries[i].reg.current_id;
        auto current = end_registries[i].reg.current_id;

        sizes_out[i] = last_registry[i].reg.wrappedSize(old, current);
        sub_buffers_out[i] = last_registry[i].reg.sliceAsync(old, current, 0);
    }

    hip::check(hipStreamSynchronize(0));

    return;
}

CUChunkAllocator* CUChunkAllocator::getStreamAllocator(hipStream_t stream,
                                                       size_t buffer_count,
                                                       size_t buffer_size) {
    auto& allocators = HipMemoryManager::getInstance().CUAllocators();
    allocators.try_emplace(stream, buffer_count, buffer_size);

    return &allocators.at(stream);
}

const std::string& hip::CUChunkAllocator::event_desc =
    hip::WaveState::description;
const std::string& hip::CUChunkAllocator::event_name = hip::WaveState::name;

} // namespace hip
