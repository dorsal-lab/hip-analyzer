/** \file hip_trace_manager.cpp
 * \brief Singleton trace manager interface. Handles asynchronously traces
 * coming from instrumenters
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation.hpp"
#include <hip_instrumentation/hip_instrumentation.hpp>
#include <hip_instrumentation/queue_info.hpp>

#include <condition_variable>
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

    // Either a counters payload or an events payload
    using Payload =
        std::variant<CountersQueuePayload<ThreadCounters>,
                     CountersQueuePayload<WaveCounters>, EventsQueuePayload>;

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
                              WaveCounterInstrumenter&& counters);

    void registerQueue(QueueInfo& queue, void* queue_data);

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
};

/** \brief Small header to validate the trace type
 */
constexpr std::string_view hiptrace_counters_name = "hiptrace_counters";

/** \brief Hiptrace event trace type
 */
constexpr std::string_view hiptrace_events_name = "hiptrace_events";

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
