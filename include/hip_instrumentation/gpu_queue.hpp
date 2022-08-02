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
        return QueueInfo{instr, sizeof(EventType), true};
    }

    /** \fn wave
     * \brief creates a WaveQueue buffer
     */
    template <class EventType> static QueueInfo wave(Instrumenter& instr) {
        return QueueInfo{instr, sizeof(EventType), false};
    }

    /** \fn offsets
     * \brief Returns a vector of offsets in the
     */
    const std::vector<size_t>& offsets() const { return offsets_vec; }

    /** \fn queueSize
     * \brief Returns the total number of elements to be allocated in the queue
     */
    size_t queueSize() const;

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
    __device__ EventType& push_back(EventType&& event);

  private:
    size_t wavefront_id;
    EventType* storage_array;
    size_t curr_id = 0u;
};

// ----- Template definition ----- //

template <class EventType>
__device__ ThreadQueue<EventType>::ThreadQueue(EventType* storage,
                                               size_t* offsets) {
    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    offset = offsets[thread_id];
}

template <class EventType>
__device__ EventType&
ThreadQueue<EventType>::push_back(const EventType& event) {
    storage[offset + curr_id] = event;
    auto& ref = storage[offset + curr_id];
    ++curr_id;

    return ref;
}

template <class EventType>
__device__ WaveQueue<EventType>::WaveQueue(EventType* storage,
                                           size_t* offsets) {
    // Compute how many waves per blocks there is
    unsigned int waves_per_block;
    if (blockDim.x % warpSize == 0) {
        waves_per_block = blockDim.x / warpSize;
    } else {
        waves_per_block = blockDim.x / warpSize + 1;
    }

    auto wave_in_block = threadIdx.x % warpSize;

    wavefront_id = blockIdx.x * waves_per_block + wave_in_block;
}

template <class EventType>
__device__ EventType& WaveQueue<EventType>::push_back(EventType&& event) {
    if (threadIdx.x % warpSize == 0) {
    }
    ++curr_id;
}

} // namespace hip
