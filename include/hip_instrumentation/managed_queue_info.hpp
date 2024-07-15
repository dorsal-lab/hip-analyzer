/** \file managed_queue_info.hpp
 * \brief GPU Queue host-side handler for dynamic memory tracing
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include "gpu_queue.hpp"
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace hip {

template <typename T> class ChunkAllocatorBase {
  public:
    struct Registry {
        /** \brief Number of SubBuffers in the global ring buffer. Must be a
         * power of two
         */
        size_t buffer_count;

        /** \brief Size of the SubBuffers
         */
        size_t buffer_size;

        T* begin;
        size_t current_id; // To be atomically incremented

        __device__ __host__ T* bufferEnd(T* sb) {
            return reinterpret_cast<T*>(reinterpret_cast<std::byte*>(sb) +
                                        buffer_size);
        }

        __device__ T* alloc(size_t id);

        __device__ void tracepoint(T** sb, size_t* offset,
                                   const std::byte buffer[], size_t size,
                                   size_t id);

        std::ostream& printBuffer(std::ostream& out, T* sb);

        size_t wrappedSize(size_t begin, size_t end);

        std::unique_ptr<std::byte[]> slice(size_t begin, size_t end);

        std::unique_ptr<std::byte[]> sliceAsync(size_t begin, size_t end,
                                                hipStream_t stream);
    };

    ChunkAllocatorBase(size_t buffer_count, size_t buffer_size,
                       bool alloc_gpu = true);
    ~ChunkAllocatorBase();

    Registry* toDevice() { return device_ptr; }

    void record(uint64_t stamp);

    T* cpuBuffer() { return buffer_ptr; }
    Registry getRegistry() const { return last_registry; }

    std::unique_ptr<std::byte[]> copyBuffer();

    /** \fn doneProcessing
     * \brief To be called by the trace manager thread to signal the
     * ChunkAllocator that it's done processing a record() request.
     *
     * \details This mechanism is needed to ensure the buffer isn't freed by the
     * time the trace processor reaches the last record() payload
     */
    void notifyDoneProcessing() { --process_count; }

    uint64_t getStamp() const { return stamp; }

  protected:
    void update();

    Registry* device_ptr;
    T* buffer_ptr;
    Registry last_registry;
    uint64_t stamp;

    std::atomic<unsigned int> process_count{0u};
};

struct SubBuffer {
    size_t owner;
    std::byte data[];
};

class ChunkAllocator : public ChunkAllocatorBase<SubBuffer> {
  public:
    ChunkAllocator(size_t buffer_count, size_t buffer_size,
                   bool alloc_gpu = true)
        : ChunkAllocatorBase(buffer_count, buffer_size, alloc_gpu) {}

    std::ostream& printBuffer(std::ostream& out);

    static const std::string& event_desc;
    static const std::string& event_name;

    static ChunkAllocator* getStreamAllocator(hipStream_t stream,
                                              size_t buffer_count,
                                              size_t buffer_size);
};

template <>
inline __device__ SubBuffer*
ChunkAllocatorBase<SubBuffer>::Registry::alloc(size_t id) {
#ifdef __HIPCC__
    auto next = atomicAdd(&current_id, 1);

    // Computing an address
    next = (next % buffer_count) * buffer_size;
    std::byte* ptr = reinterpret_cast<std::byte*>(begin) + next;

    auto sb = reinterpret_cast<SubBuffer*>(ptr);
    sb->owner = id;

    return sb;
#else
    return nullptr;
#endif
}

template <>
inline __device__ void ChunkAllocatorBase<SubBuffer>::Registry::tracepoint(
    SubBuffer** sb, size_t* offset, const std::byte buffer[], size_t size,
    size_t id) {
    if (*offset + size > buffer_size - offsetof(SubBuffer, data)) {
        *sb = alloc(id);
        *offset = 0;
    }

    memcpy((*sb)->data + *offset, buffer, size);
    *offset += size;
}

/** \class GlobalMemoryQueueInfo
 * \brief Queue Info for "naive", atomic-intensive global memory buffers for
 * tracing. New implem is basically a ChunkAllocator with buffers of a single
 * byte.
 */
class GlobalMemoryQueueInfo : public ChunkAllocatorBase<std::byte> {
  public:
    constexpr static size_t DEFAULT_SIZE = 1048576LLU; // 2 Mb

    GlobalMemoryQueueInfo(size_t elem_size, size_t buffer_size = DEFAULT_SIZE);

    static const std::string& event_name;
    static const std::string& event_desc;

  private:
    size_t elem_size;
    uint64_t stamp;
};

template <typename T> class CUChunkAllocatorBase {
  public:
    constexpr static size_t TOTAL_CU_COUNT = 112ull;
    constexpr static size_t CACHE_LINE_SIZE = 64ull;

    struct CacheAlignedRegistry {
        ChunkAllocatorBase<T>::Registry reg;
        std::byte padding[CACHE_LINE_SIZE - sizeof(reg)];
    };

    using Registries = std::array<CacheAlignedRegistry, TOTAL_CU_COUNT>;

    CUChunkAllocatorBase(size_t buffer_count, size_t buffer_size,
                         bool alloc_gpu = true);

    CUChunkAllocatorBase(const std::vector<size_t>& buffer_count,
                         size_t buffer_size);

    ~CUChunkAllocatorBase();

    CacheAlignedRegistry* toDevice() { return device_ptr; }

    void record(uint64_t stamp);

    /** \fn sync
     * \brief Wait for completion of all previous record() calls
     */
    void sync();

    Registries& getRegistries() { return last_registry; }

    std::ostream& printBuffer(std::ostream& out);

    void fetchBuffers(const Registries& begin_registries,
                      const Registries& end_registries,
                      std::array<std::unique_ptr<std::byte[]>, TOTAL_CU_COUNT>&
                          sub_buffers_out,
                      std::array<size_t, TOTAL_CU_COUNT>& sizes_out);

    /** \fn doneProcessing
     * \brief To be called by the trace manager thread to signal the
     * ChunkAllocator that it's done processing a record() request.
     *
     * \details This mechanism is needed to ensure the buffer isn't freed by
     * the time the trace processor reaches the last record() payload
     */
    void notifyDoneProcessing() { --process_count; }

    uint64_t getStamp() const { return stamp; }

  protected:
    void update();

    CacheAlignedRegistry* device_ptr;
    SubBuffer* buffer_ptr;
    Registries last_registry;

    size_t buffer_count, buffer_size;

    uint64_t stamp;

    std::atomic<unsigned int> process_count{0u};
    bool loaded = false;
};

class CUChunkAllocator : public CUChunkAllocatorBase<SubBuffer> {
  public:
    CUChunkAllocator(size_t buffer_count, size_t buffer_size,
                     bool alloc_gpu = true);

    CUChunkAllocator(const std::vector<size_t>& buffer_count,
                     size_t buffer_size);

    static CUChunkAllocator* getStreamAllocator(hipStream_t stream,
                                                size_t buffer_count,
                                                size_t buffer_size);

    static const std::string& event_desc;
    static const std::string& event_name;
};

class CUMemoryTrace : public CUChunkAllocatorBase<std::byte> {
  public:
    constexpr static size_t DEFAULT_SIZE = 1048576LLU; // 2 Mb

    CUMemoryTrace(size_t elem_size, size_t buffer_size = DEFAULT_SIZE);

    static const std::string& event_name;
    static const std::string& event_desc;

  private:
    size_t elem_size;
    uint64_t stamp;
};

} // namespace hip
