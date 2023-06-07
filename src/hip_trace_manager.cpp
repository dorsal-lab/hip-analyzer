/** \file hip_trace_manager.cpp
 * \brief Singleton trace manager interface. Handles asynchronously traces
 * coming from instrumenters
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/hip_trace_manager.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "hip_analyzer_tracepoints.h"

namespace hip {

std::ostream& dumpBinCounters(std::ostream& out, const void* counters,
                              size_t num_counters, size_t counter_size,
                              KernelInfo& kernel_info, uint64_t stamp,
                              std::pair<uint64_t, uint64_t> interval,
                              std::string_view header) {
    // Write header

    // Like "hiptrace_counters,<kernel name>,<num
    // counters>,<stamp>,<stamp_begin>,<stamp_end>,<counter size>\n"

#ifdef HIPANALYZER_BIN_OFFSETS
    std::cout << "Counters @ " << out.tellp() << '\n';
#endif

    out << header << ',' << kernel_info.name << ',' << kernel_info.instr_size
        << ',' << stamp << ',' << interval.first << ',' << interval.second
        << ',' << static_cast<unsigned int>(counter_size) << ',';

    // Kernel call configuration

    // "kernel_info,<bblocks>,<blockDim.x>,<blockDim.y>,<blockDim.z>,
    // <threadDim.x>,<threadDim.y>,<threadDim.z>\n"

    out << hiptrace_geometry << ',' << kernel_info.basic_blocks << ','
        << kernel_info.blocks.x << ',' << kernel_info.blocks.y << ','
        << kernel_info.blocks.z << ',' << kernel_info.threads_per_blocks.x
        << ',' << kernel_info.threads_per_blocks.y << ','
        << kernel_info.threads_per_blocks.z << '\n';

    // Write binary dump of counters

#ifdef HIPANALYZER_BIN_OFFSETS
    std::cout << "\tData @ " << out.tellp() << '\n';
#endif

    out.write(reinterpret_cast<const char*>(counters),
              num_counters * counter_size);

    return out;
}

template <>
std::ostream& dumpTraceBin(std::ostream& out,
                           HipTraceManager::ThreadCounters& counters,
                           KernelInfo& kernel_info, uint64_t stamp,
                           std::pair<uint64_t, uint64_t> interval) {
    using counter_t = HipTraceManager::ThreadCounters::value_type;
    lttng_ust_tracepoint(hip_instrumentation, collector_dump_thread, &out,
                         &counters, stamp);
    auto& ret = dumpBinCounters(out, counters.data(), counters.size(),
                                sizeof(counter_t), kernel_info, stamp, interval,
                                hiptrace_counters_name);
    lttng_ust_tracepoint(hip_instrumentation, collector_dump_end, &counters,
                         stamp);
    return ret;
}

template <>
std::ostream& dumpTraceBin(std::ostream& out,
                           HipTraceManager::WaveCounters& counters,
                           KernelInfo& kernel_info, uint64_t stamp,
                           std::pair<uint64_t, uint64_t> interval) {
    using counter_t = HipTraceManager::WaveCounters::value_type;
    lttng_ust_tracepoint(hip_instrumentation, collector_dump_wave, &out,
                         &counters, stamp);
    auto& ret = dumpBinCounters(out, counters.data(), counters.size(),
                                sizeof(counter_t), kernel_info, stamp, interval,
                                hiptrace_wave_counters_name);
    lttng_ust_tracepoint(hip_instrumentation, collector_dump_end, &counters,
                         stamp);
    return ret;
}

std::ostream& dumpEventsBin(std::ostream& out,
                            const std::vector<std::byte>& queue_data,
                            const std::vector<size_t>& offsets,
                            size_t event_size, std::string_view event_desc,
                            std::string_view event_name) {

    // Write header, but we don't need to include as much info as before since
    // this is the logical next step after the counter dump

    auto num_offsets = offsets.size() - 1;

#ifdef HIPANALYZER_BIN_OFFSETS
    std::cout << "Events @ " << out.tellp() << '\n';
#endif

    // Like "hiptrace_events,<event_size>,<queue_parallelism>,<event
    // name>,[<mangled_type>,<type_size>,...]\n
    out << hiptrace_events_name << ',' << static_cast<unsigned int>(event_size)
        << ',' << num_offsets << ',' << event_name << ','
        << hiptrace_event_fields << ',' << event_desc << '\n';

    // Binary dump of offsets
#ifdef HIPANALYZER_BIN_OFFSETS
    std::cout << "\tOffsets @ " << out.tellp() << '\n';
#endif

    out.write(reinterpret_cast<const char*>(offsets.data()),
              offsets.size() *
                  sizeof(size_t)); // need to keep the last offset (the end) to
                                   // know where the end is

    // Binary dump

#ifdef HIPANALYZER_BIN_OFFSETS
    std::cout << "\tQueue @ " << out.tellp() << '\n';
#endif

    out.write(reinterpret_cast<const char*>(queue_data.data()),
              queue_data.size());
    return out;
}

std::unique_ptr<HipTraceManager> HipTraceManager::instance;

HipTraceManager::HipTraceManager() {
    // Startup thread
    fs_thread = std::make_unique<std::thread>([this]() { runThread(); });
}

HipTraceManager::~HipTraceManager() {
    // Wait for completion

    auto not_empty = [this]() {
        std::lock_guard lock{mutex};
        return queue.size() != 0;
    };

    while (not_empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    cont = false;
    cond.notify_one();
    fs_thread->join();
}

void HipTraceManager::registerThreadCounters(ThreadCounterInstrumenter& instr,
                                             ThreadCounters&& counters) {
    std::lock_guard lock{mutex};

    lttng_ust_tracepoint(hip_instrumentation, register_wave_counters, &instr,
                         counters.data(), instr.getStamp());

    queue.push({CountersQueuePayload<ThreadCounters>{
        std::forward<ThreadCounters>(counters), instr.kernelInfo(),
        instr.getStamp(), instr.getInterval()}});

    cond.notify_one();
}

void HipTraceManager::registerWaveCounters(WaveCounterInstrumenter& instr,
                                           WaveCounters&& counters) {
    std::lock_guard lock{mutex};

    lttng_ust_tracepoint(hip_instrumentation, register_wave_counters, &instr,
                         counters.data(), instr.getStamp());

    queue.push({CountersQueuePayload<WaveCounters>{
        std::forward<WaveCounters>(counters), instr.kernelInfo(),
        instr.getStamp(), instr.getInterval()}});

    cond.notify_one();
}

void HipTraceManager::registerQueue(QueueInfo& queue_info, void* queue_data) {
    std::lock_guard lock{mutex};

    lttng_ust_tracepoint(hip_instrumentation, register_queue,
                         queue_info.getInstrumenter(), queue_data,
                         queue_info.getInstrumenter()->getStamp());

    queue.push({EventsQueuePayload{queue_data, std::move(queue_info)}});

    cond.notify_one();
}

/** \brief Small header to validate the trace type
 */
constexpr auto hiptrace_managed_name = "hiptrace_managed";

template <>
void HipTraceManager::handlePayload(
    CountersQueuePayload<ThreadCounters>&& payload, std::ofstream& out) {
    auto& [counters, kernel_info, stamp, rocm_stamp] = payload;

    dumpTraceBin(out, counters, kernel_info, stamp, rocm_stamp);
}

template <>
void HipTraceManager::handlePayload(
    CountersQueuePayload<WaveCounters>&& payload, std::ofstream& out) {
    auto& [counters, kernel_info, stamp, rocm_stamp] = payload;

    dumpTraceBin(out, counters, kernel_info, stamp, rocm_stamp);
}

template <>
void HipTraceManager::handlePayload(EventsQueuePayload&& payload,
                                    std::ofstream& out) {
    auto& [events, queue_info] = payload;

    const auto* data = queue_info.events().data();
    auto stamp = queue_info.getInstrumenter()->getStamp();

    lttng_ust_tracepoint(hip_instrumentation, collector_dump_wave, &out, data,
                         stamp);
    queue_info.fromDevice(events);

    hip::check(hipFree(events));
    /*
                        std::move(offsets),
                                       queue_info.elemSize(),
       queue_info.getDesc(), queue_info.getName()}}
    */
    dumpEventsBin(out, queue_info.events(), queue_info.offsets(),
                  queue_info.elemSize(), queue_info.getDesc(),
                  queue_info.getName());
    lttng_ust_tracepoint(hip_instrumentation, collector_dump_end, data, stamp);
}

template <class> inline constexpr bool always_false_v = false;

void HipTraceManager::runThread() {
    // Init output file
    auto now = std::chrono::steady_clock::now();
    uint64_t stamp = std::chrono::duration_cast<std::chrono::microseconds>(
                         now.time_since_epoch())
                         .count();

    std::stringstream filename;
    filename << "hiptrace_" << stamp << ".hiptrace";

    std::ofstream out{filename.str(), std::ostream::binary};

    if (!out.is_open()) {
        throw std::runtime_error(
            "HipTraceManager::runThread() : Could not open output file " +
            filename.str());
    }

    // Managed header
    out << hiptrace_managed_name << '\n';

    while (true) {
        std::unique_ptr<Payload> payload;

        // Thread-sensitive part, perform all moves / copies
        {
            // Acquire mutex
            std::unique_lock<std::mutex> lock(mutex);
            cond.wait(lock, [&]() { return queue.size() != 0 || !cont; });

            if (!cont) {
                return;
            }

            auto& front = queue.front();

            payload = std::make_unique<Payload>(std::move(front));

            queue.pop();
        }

        // Some template magic for you
        std::visit(
            [this, &out](auto& arg) { handlePayload(std::move(arg), out); },
            *payload);
    }
}

} // namespace hip
