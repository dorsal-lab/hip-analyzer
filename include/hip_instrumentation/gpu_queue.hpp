/** \file GpuQueue.hpp
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
    template <class EventType> static QueueInfo thread(Instrumenter& instr);

    /** \fn wave
     * \brief creates a WaveQueue buffer
     */
    template <class EventType> static QueueInfo wave(Instrumenter& instr);
};

/** \class ThreadQueue
 * \brief A queue to be filled with events from every thread, of every block.
 */
template <class EventType> class ThreadQueue {
  public:
    /** ctor
     * TODO : pass counter (size) requirements
     */
    ThreadQueue(EventType* storage);

    /** \fn push_back
     * \brief Appends an event to the queue
     */
    EventType& push_back(EventType&& event);

  private:
    unsigned int thread_id;
    unsigned int curr_id;
};

/** \class WaveQueue
 * \brief A queue with events from every wavefront of the kernel
 */
template <class EventType> class WaveQueue {
  public:
    __device__ WaveQueue(EventType* storage);

    /** \fn push_back
     * \brief Appends an event to the queue
     */
    __device__ EventType& push_back(EventType&& event);

  private:
    unsigned int wavefront_id;
    unsigned int curr_id = 0u;
};

// ----- Template definition ----- //

template <class EventType>
__device__ WaveQueue<EventType>::WaveQueue(EventType* storage) {
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

    ++curr_id;
}

} // namespace hip
