/** \file hip_trace_manager.cpp
 * \brief Singleton trace manager interface. Handles asynchronously traces
 * coming from instrumenters
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

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
    using Counters = std::vector<Instrumenter::counter_t>;

    // <Counters data> - <kernel launch info> - <stamp> - <pair of roctracer
    // stamp>
    using CountersQueuePayload = std::tuple<Counters, KernelInfo, uint64_t,
                                            std::pair<uint64_t, uint64_t>>;

    // <queue data> - <offsets> - <event size> - <event description (types,
    // sizes)> - <event type name>
    /*using EventsQueuePayload = std::tuple<void*, std::vector<size_t>, size_t,
                                          std::string, std::string>;*/
    using EventsQueuePayload = std::tuple<void*, QueueInfo>;

    // Either a counters payload or an events payload
    using Payload = std::variant<CountersQueuePayload, EventsQueuePayload>;

    HipTraceManager(const HipTraceManager&) = delete;
    HipTraceManager operator=(const HipTraceManager&) = delete;
    ~HipTraceManager();

    static HipTraceManager& getInstance() {
        if (instance.get() == nullptr) {
            instance = std::unique_ptr<HipTraceManager>(new HipTraceManager());
        }

        return *instance;
    }

    void registerCounters(Instrumenter& instr, Counters&& counters);

    void registerQueue(QueueInfo& queue, void* queue_data);

  private:
    HipTraceManager();

    void runThread();

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

std::ostream& dumpTraceBin(std::ostream& out,
                           HipTraceManager::Counters& counters,
                           KernelInfo& kernel_info, uint64_t stamp,
                           std::pair<uint64_t, uint64_t> interval);

} // namespace hip
