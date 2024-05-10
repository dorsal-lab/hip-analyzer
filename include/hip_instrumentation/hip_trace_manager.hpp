/** \file hip_trace_manager.cpp
 * \brief Singleton trace manager interface. Handles asynchronously traces
 * coming from instrumenters
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation.hpp"
#include "managed_queue_info.hpp"
#include <hip_instrumentation/hip_instrumentation.hpp>
#include <hip_instrumentation/queue_info.hpp>

#include <condition_variable>
#include <fstream>
#include <mutex>
#include <queue>
#include <variant>
#include <vector>

namespace hip {

/** \class TraceManager
 * \brief Singleton manager to record traces and save them to the filesystem
 */
class HipTraceManager {
  public:
    using ThreadCounters = std::vector<ThreadCounterInstrumenter::counter_t>;
    using WaveCounters = std::vector<WaveCounterInstrumenter::counter_t>;

    // <Counters data> - <kernel launch info> - <stamp> - <pair of roctracer
    // stamp>
    template <typename T>
    using CountersQueuePayload =
        std::tuple<T, KernelInfo, uint64_t, std::pair<uint64_t, uint64_t>>;

    // <queue data> - <offsets> - <event size> - <event description (types,
    // sizes)> - <event type name>
    /*using EventsQueuePayload = std::tuple<void*, std::vector<size_t>, size_t,
                                          std::string, std::string>;*/
    using EventsQueuePayload = std::tuple<void*, QueueInfo>;

    using GlobalMemoryEventsQueuePayload =
        std::tuple<GlobalMemoryQueueInfo*, uint64_t,
                   GlobalMemoryQueueInfo::Registry, size_t>;

    // <allocator> - <stamp> - <end registry> - <'currend_id' at launch>
    using ChunkAllocatorEventsQueuePayload =
        std::tuple<ChunkAllocator*, uint64_t, ChunkAllocator::Registry, size_t>;

    // <allocator> - <stamp> - <begin registries> - <end registries>
    using CUChunkAllocatorEventsQueuePayload =
        std::tuple<CUChunkAllocator*, uint64_t,
                   std::unique_ptr<CUChunkAllocator::Registries>,
                   std::unique_ptr<CUChunkAllocator::Registries>>;

    // Either a counters payload or an events payload
    using Payload =
        std::variant<CountersQueuePayload<ThreadCounters>,
                     CountersQueuePayload<WaveCounters>, EventsQueuePayload,
                     GlobalMemoryEventsQueuePayload,
                     ChunkAllocatorEventsQueuePayload,
                     CUChunkAllocatorEventsQueuePayload>;

    HipTraceManager(const HipTraceManager&) = delete;
    HipTraceManager operator=(const HipTraceManager&) = delete;
    ~HipTraceManager();

    static HipTraceManager& getInstance() {
        if (instance.get() == nullptr) {
            instance = std::unique_ptr<HipTraceManager>(new HipTraceManager());
        }

        return *instance;
    }

    void registerThreadCounters(ThreadCounterInstrumenter& instr,
                                ThreadCounters&& counters);

    void registerWaveCounters(WaveCounterInstrumenter& instr,
                              WaveCounters&& counters);

    void registerQueue(QueueInfo& queue, void* queue_data);

    void registerGlobalMemoryQueue(GlobalMemoryQueueInfo* queue, uint64_t stamp,
                                   GlobalMemoryQueueInfo::Registry queue_data,
                                   size_t begin_offset);

    void registerChunkAllocatorEvents(ChunkAllocator* alloc, uint64_t stamp,
                                      ChunkAllocator::Registry end_registry,
                                      size_t begin_offset);

    void registerCUChunkAllocatorEvents(
        CUChunkAllocator* alloc, uint64_t stamp,
        std::unique_ptr<CUChunkAllocator::Registries> begin_registries,
        std::unique_ptr<CUChunkAllocator::Registries> end_registries);

    size_t queuedPayloads();

    bool isEmpty() { return queuedPayloads() == 0; }

    /** \fn flush
     * \brief Returns once the queue has been emptied to disk
     */
    void flush();

  private:
    HipTraceManager();

    void runThread();
    template <typename T> void handlePayload(T&& payload, std::ofstream& out);

    static std::unique_ptr<HipTraceManager> instance;

    /** \brief Thread writing to the file system
     */
    std::unique_ptr<std::thread> fs_thread;
    bool cont = true;
    std::mutex mutex;
    std::condition_variable cond;
    std::queue<Payload> queue;

    constexpr static std::string_view HIPTRACE_ENV = "HIPTRACE_OUTPUT";
};

/** \class HipTraceFile
 * \brief Stateful representation of a HipTrace input file. Generates
 * CounterInstrumenters and QueueInfos
 */
class HipTraceFile {
  public:
    enum class Kind {
        Managed,
        Counters,
        WaveCounters,
        Events,
        ChunkAlloc,
        CUChunkAlloc,
        ErrorKind
    };

    /** ctor
     * \brief Constructor. Load a trace file and initialize
     */
    HipTraceFile(std::string_view filename);

    /** \fn getNext
     * \brief Return the next payload
     */
    HipTraceManager::Payload getNext();

    /** \fn done
     * \brief Returns whether the current file has been fully consumed
     */
    bool done();

  private:
    Kind parseHeader(std::string_view header);

    /** loadEvents
     * \brief Needed to be a member function to access a private ctor for
     * QueueInfo
     */
    HipTraceManager::EventsQueuePayload loadEvents(const std::string& header,
                                                   std::ifstream& f) const;

    size_t offset = 0u;
    std::ifstream input;
    Kind trace_kind;
};

/** \brief Small header to validate the trace type - composite tracefile
 */
constexpr auto hiptrace_managed_name = "hiptrace_managed";

/** \brief Small header to validate the trace type - thread counters
 */
constexpr std::string_view hiptrace_counters_name = "hiptrace_counters";

/** \brief Small header to validate the trace type - wave counters
 */
constexpr std::string_view hiptrace_wave_counters_name =
    "hiptrace_wave_counters";

/** \brief Hiptrace event trace type
 */
constexpr std::string_view hiptrace_events_name = "hiptrace_events";

/** \brief Hiptrace raw event trace type
 */
constexpr std::string_view hiptrace_raw_events_name = "hiptrace_raw_events";

/** \brief Hiptrace chunk-allocated events
 */
constexpr std::string_view hiptrace_chunk_events_name = "hiptrace_chunks";

/** \brief Hiptrace multiple chunk-allocated events
 */
constexpr std::string_view hiptrace_cu_chunk_events_name = "hiptrace_cu_chunks";

/** \brief Hiptrace begin kernel info
 */
constexpr std::string_view hiptrace_geometry = "kernel_info";

/** \brief Hiptrace event fields
 */
constexpr std::string_view hiptrace_event_fields = "begin_fields";

template <typename Counters>
std::ostream& dumpTraceBin(std::ostream& out, Counters& counters,
                           KernelInfo& kernel_info, uint64_t stamp,
                           std::pair<uint64_t, uint64_t> interval);

template <>
std::ostream& dumpTraceBin(std::ostream& out,
                           HipTraceManager::ThreadCounters& counters,
                           KernelInfo& kernel_info, uint64_t stamp,
                           std::pair<uint64_t, uint64_t> interval);

template <>
std::ostream& dumpTraceBin(std::ostream& out,
                           HipTraceManager::WaveCounters& counters,
                           KernelInfo& kernel_info, uint64_t stamp,
                           std::pair<uint64_t, uint64_t> interval);

} // namespace hip
