/** \file managed_queue_info.hpp
 * \brief GPU Queue host-side handler for dynamic memory tracing
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include <vector>

namespace hip {

/** \class GlobalMemoryQueueInfo
 * \brief Queue Info for "naive", atomic-intensive global memory buffers for
 * tracing.
 */
class GlobalMemoryQueueInfo {
    constexpr static size_t DEFAULT_SIZE = 1048576LLU; // 2 Mb
  public:
    struct GlobalMemoryTrace {
        void* current; // At init : contains
        void* end;
    };

    GlobalMemoryQueueInfo(size_t elem_size, size_t buffer_size = DEFAULT_SIZE);

    size_t bufferSize() const { return cpu_queue.size(); }
    size_t queueLength() const { return cpu_queue.size() / elem_size; }
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

  private:
    std::vector<std::byte> cpu_queue;
    GlobalMemoryTrace cpu_trace;
    size_t elem_size;
};

} // namespace hip
