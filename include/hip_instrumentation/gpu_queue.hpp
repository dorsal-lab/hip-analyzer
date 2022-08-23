/** \file gpu_queue.hpp
 * \brief Data-parallel execution queue
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include "hip_instrumentation.hpp"

#include "queue_info.hpp"

#include "cuda_wrappers/new"

namespace hip {

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
    __device__ EventType& push_back(const EventType& event) {
        return emplace_back(event);
    }

    template <typename... Args>
    __device__ EventType& emplace_back(Args&&... args);

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
    __device__ EventType& push_back(const EventType& event) {
        return emplace_back(event);
    }

    template <typename... Args>
    __device__ EventType& emplace_back(Args&&... args);

    __device__ size_t index() const { return curr_id; }

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
template <typename... Args>
__device__ EventType& ThreadQueue<EventType>::emplace_back(Args&&... args) {
    auto* ptr = &storage[offset + curr_id];

    // Placement new
    new (ptr) EventType(std::forward<Args>(args)...);

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

    auto wave_in_block = threadIdx.x / warpSize;

    wavefront_id = blockIdx.x * waves_per_block + wave_in_block;
    offset = offsets[wavefront_id];
}

template <class EventType>
template <typename... Args>
__device__ EventType& WaveQueue<EventType>::emplace_back(Args&&... args) {
    auto* ptr = &storage[offset + curr_id];
    if (threadIdx.x % warpSize == 0) {
        // Placement new
        new (ptr) EventType(std::forward<Args>(args)...);
        ++curr_id;
    }

    return *ptr;
}

template <typename T, typename... Args> std::string HipEventFields() {
    if constexpr (sizeof...(Args) > 0) {
        return HipEventFields<T>() + ',' + HipEventFields<Args...>();
    } else {
        std::stringstream ss;
        ss << typeid(T).name() << ',' << sizeof(T);
        return ss.str();
    }
}

namespace gcnasm {

/** \fn get_hw_id
 * \brief Returns the value of the harware id register
 */
inline __device__ uint32_t get_hw_id() {
    uint32_t hw_id;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(hw_id) :);

    return hw_id;
}

/** \fn get_exec
 * \brief Returns the value of the EXEC register (execution mask)
 */
inline __device__ uint64_t get_exec() {
    uint64_t exec;
    asm volatile("s_mov_b64 %0, exec" : "=s"(exec) :);

    return exec;
}

inline __device__ uint64_t get_stamp() {
    uint64_t stamp;
    asm volatile("s_memrealtime %0" : "=s"(stamp) :);

    return stamp;
}

} // namespace gcnasm

// Sample event classes

/** \struct Event
 * \brief Contains just the basic block id
 */
struct Event {
    size_t bb;

    __device__ Event(size_t bb) : bb(bb) {}

    static std::string description;
    static std::string name;
};

/** \struct TaggedEvent
 * \brief Event, with an associated timestamp
 */
struct TaggedEvent {
    __device__ TaggedEvent(size_t bb) : bb(bb) { stamp = gcnasm::get_stamp(); }

    size_t bb;
    uint64_t stamp;

    static std::string description;
    static std::string name;
};

struct WaveState {
    __device__ WaveState(size_t bb) : bb(bb) {
        stamp = gcnasm::get_stamp();
        exec = gcnasm::get_exec();
        hw_id = gcnasm::get_hw_id();
    }

    size_t bb;
    uint64_t stamp;
    uint64_t exec;
    uint32_t hw_id;

    static std::string description;
    static std::string name;
};

} // namespace hip
