/** \file queue_info.hpp
 * \brief GPU Queue host-side handler
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include "hip_instrumentation.hpp"

namespace hip {

/** \struct QueueInfo
 * \brief Computes and holds information about a queue to be submitted to the
 * GPU : buffer offset, total size, etc
 */
struct QueueInfo {
  public:
    /** \fn thread
     * \brief Creates a ThreadQueue buffer
     */
    template <class EventType> static QueueInfo thread(Instrumenter& instr) {
        static_assert(std::is_trivially_copyable_v<EventType>);
        return QueueInfo{instr, sizeof(EventType), true};
    }

    /** \fn wave
     * \brief creates a WaveQueue buffer
     */
    template <class EventType> static QueueInfo wave(Instrumenter& instr) {
        static_assert(std::is_trivially_copyable_v<EventType>);
        return QueueInfo{instr, sizeof(EventType), false};
    }

    /** \fn offsets
     * \brief Returns a vector of offsets in the
     */
    const std::vector<size_t>& offsets() const { return offsets_vec; }

    /** \fn parallelism
     * \brief Returns the number of parallel queues effectively allocated
     */
    size_t parallelism() const { return offsets_vec.size() - 1; }

    /** \fn queueSize
     * \brief Returns the total number of elements to be allocated in the queue
     */
    size_t queueLength() const;

    /** \fn totalSize
     * \brief Returns the total size (in bytes) of the queue buffer
     */
    size_t totalSize() const { return queueLength() * elem_size; }

    /** \fn elemSize
     * \brief Returns the size of one element
     */
    size_t elemSize() const { return elem_size; }

    /** \fn allocBuffer
     * \brief Allocates the queue buffer on the device
     */
    template <class EventType> EventType* allocBuffer() const {
        EventType* ptr;
        hip::check(hipMalloc(&ptr, totalSize()));
        hip::check(hipMemset(ptr, 0u, totalSize()));
        return ptr;
    }

    /** \fn allocOffsets
     * \brief Allocates the offsets list on the device
     */
    size_t* allocOffsets() const {
        size_t* ptr;
        hip::check(hipMalloc(&ptr, parallelism() * sizeof(size_t)));
        hip::check(hipMemcpy(ptr, offsets_vec.data(),
                             parallelism() * sizeof(size_t),
                             hipMemcpyHostToDevice));
        return ptr;
    }

    /** \fn fromDevice
     * \brief Copies the queue data from the device
     */
    void fromDevice(void* ptr);

    /** \fn record
     * \brief Transfers the queue to the trace manager
     */
    void record();

  private:
    QueueInfo(Instrumenter& instr, size_t elem_size, bool is_thread);

    void computeSize();

    Instrumenter& instr;
    bool is_thread;
    size_t elem_size;
    std::vector<size_t> offsets_vec;
    std::vector<std::byte> cpu_queue;
};

} // namespace hip