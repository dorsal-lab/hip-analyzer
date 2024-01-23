/** \file managed_queue_info.hpp
 * \brief GPU Queue host-side handler for dynamic memory tracing
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include "gpu_queue.hpp"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace hip {

/** \class GlobalMemoryQueueInfo
 * \brief Queue Info for "naive", atomic-intensive global memory buffers for
 * tracing.
 */
class GlobalMemoryQueueInfo {
  public:
    constexpr static size_t DEFAULT_SIZE = 1048576LLU; // 2 Mb
    struct GlobalMemoryTrace {
        void* current; // At init : contains
        void* end;
    };

    GlobalMemoryQueueInfo(size_t elem_size, size_t buffer_size = DEFAULT_SIZE);

    size_t bufferSize() const { return cpu_queue.size(); }
    size_t queueLength() const { return cpu_queue.size() / elem_size; }
    size_t elemSize() const { return elem_size; }
    const GlobalMemoryTrace& cpuTrace() const { return cpu_trace; }
    const std::vector<std::byte>& buffer() const { return cpu_queue; }

    /** \fn toDevice
     * \brief Create a global memory trace on the device
     */
    GlobalMemoryTrace* toDevice();

    /** \fn fromDevice
     * \brief Copies the queue data from the device
     */
    void fromDevice(GlobalMemoryTrace* device_ptr);

    /** \fn record
     * \brief Transfers the queue to the trace manager
     */
    void record(GlobalMemoryTrace* device_ptr);

    /** \fn getStamp
     * \brief Returns the Instrumenter timestamp (construction)
     */
    uint64_t getStamp() const { return stamp; }

    const std::string& event_name = hip::GlobalWaveState::name;
    const std::string& event_desc = hip::GlobalWaveState::description;

  private:
    std::vector<std::byte> cpu_queue;
    GlobalMemoryTrace cpu_trace;
    size_t elem_size;
    uint64_t stamp;
};

class ChunkAllocator {
  public:
    struct SubBuffer {
        size_t owner;
        std::byte data[];
    };

    struct Registry {
        /** \brief Number of SubBuffers in the global ring buffer. Must be a
         * power of two
         */
        size_t buffer_count;

        /** \brief Size of the SubBuffers
         */
        size_t buffer_size;

        SubBuffer* begin;
        size_t current_id; // To be atomically incremented

        __device__ __host__ SubBuffer* bufferEnd(SubBuffer* sb) {
            return reinterpret_cast<SubBuffer*>(
                reinterpret_cast<std::byte*>(sb) + buffer_size);
        }

        __device__ SubBuffer* alloc(size_t id) {
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

        __device__ void tracepoint(SubBuffer** sb, size_t* offset,
                                   const std::byte buffer[], size_t size,
                                   size_t id) {
            if (*offset + size > buffer_size - offsetof(SubBuffer, data)) {
                *sb = alloc(id);
                *offset = 0;
            }

            memcpy((*sb)->data + *offset, buffer, size);
            *offset += size;
        }

        std::ostream& printBuffer(std::ostream& out, SubBuffer* sb);
    };

    ChunkAllocator(size_t buffer_count, size_t buffer_size);
    ~ChunkAllocator();

    Registry* toDevice() { return device_ptr; }

    void record(uint64_t stamp);

    std::unique_ptr<std::byte[]> copyBuffer();
    std::unique_ptr<std::byte[]> slice(size_t begin, size_t end);
    std::ostream& printBuffer(std::ostream& out);

    /**
     *
     */
    static ChunkAllocator* getStreamAllocator(hipStream_t stream,
                                              size_t buffer_count,
                                              size_t buffer_size);

    static const std::string& event_desc;
    static const std::string& event_name;

  private:
    void update();

    Registry* device_ptr;
    SubBuffer* buffer_ptr;
    Registry last_registry;

    // Singleton map of allocators
    static std::map<hipStream_t, ChunkAllocator> allocators;
};

} // namespace hip
