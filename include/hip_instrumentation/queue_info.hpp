/** \file queue_info.hpp
 * \brief GPU Queue host-side handler for precomputed buffers & offsets
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include "hip_instrumentation.hpp"

#include <type_traits>
#include <variant>

namespace hip {

class HipTraceFile;

/** \struct QueueInfo
 * \brief Computes and holds information about a queue to be submitted to the
 * GPU : buffer offset, total size, etc
 */
struct QueueInfo {
    friend class HipTraceFile;

  public:
    /** \fn thread
     * \brief Creates a ThreadQueue buffer
     */
    template <class EventType>
    static QueueInfo thread(ThreadCounterInstrumenter& instr,
                            size_t extra_size = 0u, float size_factor = 1.f) {
        static_assert(std::is_trivially_copyable_v<EventType>);
        static_assert(
            std::is_same_v<decltype(EventType::description), std::string>);
        static_assert(std::is_same_v<decltype(EventType::name), std::string>);

        return QueueInfo{instr,
                         sizeof(EventType),
                         true,
                         EventType::description,
                         EventType::name,
                         extra_size,
                         size_factor};
    }

    /** \fn wave
     * \brief creates a WaveQueue buffer
     */
    template <class EventType>
    static QueueInfo wave(ThreadCounterInstrumenter& instr,
                          size_t extra_size = 0u, float size_factor = 1.f) {
        static_assert(std::is_trivially_copyable_v<EventType>);
        static_assert(
            std::is_same_v<decltype(EventType::description), std::string>);
        static_assert(std::is_same_v<decltype(EventType::name), std::string>);

        return QueueInfo{instr,           sizeof(EventType),
                         false,           EventType::description,
                         EventType::name, extra_size,
                         size_factor};
    }

    /** \fn wave
     * \brief Create a WaveQueue buffer, requiring a WaveCounterInstrumenter
     */
    template <class EventType>
    static QueueInfo wave(WaveCounterInstrumenter& instr,
                          size_t extra_size = 0u, float size_factor = 1.f) {
        static_assert(std::is_trivially_copyable_v<EventType>);
        static_assert(
            std::is_same_v<decltype(EventType::description), std::string>);
        static_assert(std::is_same_v<decltype(EventType::name), std::string>);

        return QueueInfo{
            instr,           sizeof(EventType), EventType::description,
            EventType::name, extra_size,        size_factor};
    }

    QueueInfo(QueueInfo&&) = default;

    /** \fn offsets
     * \brief Returns a vector of offsets in the
     */
    const std::vector<size_t>& offsets() const { return offsets_vec; }

    /** \fn events
     * \brief Returns events
     */
    const std::vector<std::byte>& events() const { return cpu_queue; }

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

    /** \fn getDesc
     * \brief Returns the queue type description, from EventType::description
     */
    const std::string& getDesc() const { return type_desc; }

    /** \fn getName
     * \brief Returns the queue type name (ref to static member)
     */
    const std::string& getName() const { return type_name; }

    /** \fn allocBuffer
     * \brief Allocates the queue buffer on the device
     */
    template <class EventType> EventType* allocBuffer() const {
        EventType* ptr;
        hip::check(hipMalloc(&ptr, totalSize()));
        hip::check(hipMemset(ptr, 0u, totalSize()));
        return ptr;
    }

    void* allocBuffer() const {
        void* ptr;
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
    void record(void* ptr);

    const CounterInstrumenter* getInstrumenter() const {
        return std::visit(
            [](const auto& i) { return static_cast<CounterInstrumenter*>(i); },
            instr);
    }

  private:
    QueueInfo(ThreadCounterInstrumenter& instr, size_t elem_size,
              bool is_thread, const std::string& type_desc,
              const std::string& type_name, size_t extra_size,
              float size_factor);

    QueueInfo(WaveCounterInstrumenter& instr, size_t elem_size,
              const std::string& type_desc, const std::string& type_name,
              size_t extra_size, float size_factor);

    QueueInfo(size_t elem_size, const std::string& type_desc,
              const std::string& type_name, std::vector<size_t>&& offsets_vec,
              std::vector<std::byte>&& events)
        : instr(static_cast<WaveCounterInstrumenter*>(nullptr)),
          is_thread(false), elem_size(elem_size), extra_size(0u),
          size_factor(1.f), offsets_vec(offsets_vec), cpu_queue(events),
          type_desc(type_desc), type_name(type_name) {}

    void computeSizeThreadFromThread();

    void computeSizeWaveFromThread();

    void computeSizeWaveFromWave();

    std::variant<ThreadCounterInstrumenter*, WaveCounterInstrumenter*> instr;
    bool is_thread;
    size_t elem_size;
    size_t extra_size;
    float size_factor;

    std::vector<size_t> offsets_vec;
    std::vector<std::byte> cpu_queue;

    const std::string& type_desc;
    const std::string& type_name;
};

} // namespace hip
