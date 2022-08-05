/** \file gpu_queue.hpp
 * \brief Data-parallel execution queue
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

  private:
    QueueInfo(Instrumenter& instr, size_t elem_size, bool is_thread);

    void computeSize();

    Instrumenter& instr;
    bool is_thread;
    size_t elem_size;
    std::vector<size_t> offsets_vec;
};

/** \class ThreadQueue
 * \brief A queue to be filled with events from every thread, of every block.
 */
template <class EventType> class ThreadQueue {
  public:
    /** ctor
     * TODO : pass counter (size) requirements
     */
    __device__ ThreadQueue(EventType* storage, size_t* offsets);

    /** \fn push_back
     * \brief Appends an event to the queue
     */
    __device__ EventType& push_back(const EventType& event);

    __device__ size_t index() const { return curr_id; }

  private:
    size_t thread_id;
    size_t offset;
    EventType* storage;
    size_t curr_id = 0u;
};

/** \class WaveQueue
 * \brief A queue with events from every wavefront of the kernel
 */
template <class EventType> class WaveQueue {
  public:
    __device__ WaveQueue(EventType* storage, size_t* offsets);

    /** \fn push_back
     * \brief Appends an event to the queue
     */
    __device__ EventType& push_back(const EventType& event);

  private:
    size_t wavefront_id;
    size_t offset;
    EventType* storage;
    size_t curr_id = 0u;
};

// ----- Template definition ----- //

template <class EventType>
__device__ ThreadQueue<EventType>::ThreadQueue(EventType* storage,
                                               size_t* offsets)
    : storage(storage) {
    static_assert(std::is_trivially_copyable_v<EventType>);
    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    offset = offsets[thread_id];
}

template <class EventType>
__device__ EventType&
ThreadQueue<EventType>::push_back(const EventType& event) {
    auto* ptr = &storage[offset + curr_id];
    *ptr = event;

    ++curr_id;
    return *ptr;
}

template <class EventType>
__device__ WaveQueue<EventType>::WaveQueue(EventType* storage, size_t* offsets)
    : storage(storage) {
    static_assert(std::is_trivially_copyable_v<EventType>);
    // Compute how many waves per blocks there is
    unsigned int waves_per_block;
    if (blockDim.x % warpSize == 0) {
        waves_per_block = blockDim.x / warpSize;
    } else {
        waves_per_block = blockDim.x / warpSize + 1;
    }

    auto wave_in_block = threadIdx.x % warpSize;

    wavefront_id = blockIdx.x * waves_per_block + wave_in_block;
    offset = offsets[wavefront_id];
}

template <class EventType>
__device__ EventType& WaveQueue<EventType>::push_back(const EventType& event) {
    auto* ptr = &storage[offset + curr_id];
    if (threadIdx.x % warpSize == 0) {
        *ptr = event;
        ++curr_id;
    }

    return *ptr;
}

// Sample event classes

/** \struct Event
 * \brief Contains just the basic block id
 */
struct Event {
    size_t bb;
};

/** \struct TaggedEvent
 * \brief Event, with an associated timestamp
 */
struct TaggedEvent {
    TaggedEvent(size_t bb) : bb(bb) {
        // TODO : fetch GPU timestamp, asm?
    }
    size_t bb;
};

} // namespace hip
