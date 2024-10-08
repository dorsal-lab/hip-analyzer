/** \file hip_trace_manager.cpp
 * \brief Singleton trace manager interface. Handles asynchronously traces
 * coming from instrumenters
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/hip_trace_manager.hpp"

#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

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
        << ',' << num_counters << ',' << static_cast<unsigned int>(counter_size)
        << ',';

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

std::ofstream& dumpEventsBin(std::ofstream& out, uint64_t stamp,
                             const std::vector<std::byte>& queue_data,
                             size_t event_size, std::string_view event_desc,
                             std::string_view event_name) {
    out << hiptrace_raw_events_name << ',' << stamp << ',' << event_size << ','
        << queue_data.size() << ',' << event_name << ','
        << hiptrace_event_fields << ',' << event_desc << '\n';

    out.write(reinterpret_cast<const char*>(queue_data.data()),
              queue_data.size());

    return out;
}

std::ofstream& dumpEventsBin(std::ofstream& out, uint64_t stamp,
                             const std::unique_ptr<std::byte[]>& queue_data,
                             std::string_view event_desc,
                             std::string_view event_name, size_t buffer_count,
                             size_t buffer_size) {
    out << hiptrace_chunk_events_name << ',' << stamp << ',' << buffer_count
        << ',' << buffer_size << ',' << event_name << ','
        << hiptrace_event_fields << event_desc << '\n';

    out.write(reinterpret_cast<const char*>(queue_data.get()),
              buffer_count * buffer_size);

    return out;
}

std::ofstream&
dumpEventsBin(std::ofstream& out, uint64_t stamp,
              const std::array<std::unique_ptr<std::byte[]>,
                               CUChunkAllocator::TOTAL_CU_COUNT>& queue_data,
              const std::array<size_t, CUChunkAllocator::TOTAL_CU_COUNT>& sizes,
              std::string_view event_desc, std::string_view event_name,
              size_t buffer_count, size_t buffer_size) {

    out << hiptrace_cu_chunk_events_name << ',' << stamp << ',' << buffer_count
        << ',' << buffer_size << ',' << CUChunkAllocator::TOTAL_CU_COUNT
        << event_name << ',' << hiptrace_event_fields << ',' << event_desc
        << ',' << '\n';

    for (auto i = 0u; i < CUChunkAllocator::TOTAL_CU_COUNT; ++i) {
        auto size = sizes[i];
        out << size << '\n';

        out.write(reinterpret_cast<const char*>(queue_data[i].get()),
                  buffer_size * size);
    }

    return out;
}

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

void HipTraceManager::registerGlobalMemoryQueue(
    GlobalMemoryQueueInfo* queue_info, uint64_t stamp,
    GlobalMemoryQueueInfo::Registry end_registry, size_t begin_offset) {
    std::lock_guard lock{mutex};

    lttng_ust_tracepoint(hip_instrumentation, register_global_memory_queue,
                         queue_info, stamp, begin_offset,
                         end_registry.current_id);

    queue.push(GlobalMemoryEventsQueuePayload{queue_info, stamp, end_registry,
                                              begin_offset});

    cond.notify_one();
}

void HipTraceManager::registerCUMemoryQueue(
    CUMemoryTrace* queue_info, uint64_t stamp,
    std::unique_ptr<CUMemoryTrace::Registries> begin_registries,
    std::unique_ptr<CUMemoryTrace::Registries> end_registries) {
    std::lock_guard lock{mutex};

    lttng_ust_tracepoint(hip_instrumentation, register_cu_memory_queue,
                         queue_info, stamp);

    queue.push(CUMemoryQueuePayload{queue_info, stamp,
                                    std::move(begin_registries),
                                    std::move(end_registries)});

    cond.notify_one();
}

void HipTraceManager::registerChunkAllocatorEvents(
    ChunkAllocator* alloc, uint64_t stamp,
    ChunkAllocator::Registry end_registry, size_t begin_offset) {
    std::lock_guard lock{mutex};

    lttng_ust_tracepoint(hip_instrumentation, register_chunk_allocator_events,
                         alloc, stamp, begin_offset, end_registry.current_id);

    queue.push(ChunkAllocatorEventsQueuePayload{alloc, stamp, end_registry,
                                                begin_offset});

    cond.notify_one();
}

void HipTraceManager::registerCUChunkAllocatorEvents(
    CUChunkAllocator* alloc, uint64_t stamp,
    std::unique_ptr<CUChunkAllocator::Registries> begin_registries,
    std::unique_ptr<CUChunkAllocator::Registries> end_registries) {
    std::lock_guard lock{mutex};

    lttng_ust_tracepoint(hip_instrumentation,
                         register_cu_chunk_allocator_events, alloc, stamp);

    queue.push(CUChunkAllocatorEventsQueuePayload{
        alloc, stamp, std::move(begin_registries), std::move(end_registries)});

    cond.notify_one();
}

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

    [[maybe_unused]] const auto* data = queue_info.events().data();
    [[maybe_unused]] auto stamp = queue_info.getInstrumenter()->getStamp();

    lttng_ust_tracepoint(hip_instrumentation, collector_dump_queue, &out, data,
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

template <>
void HipTraceManager::handlePayload(GlobalMemoryEventsQueuePayload&& payload,
                                    std::ofstream& out) {
    auto& [queue_info, stamp, registry, begin] = payload;

    lttng_ust_tracepoint(hip_instrumentation, collector_dump_queue, &out,
                         registry.begin, stamp);

    auto end = registry.current_id;

    auto buf = registry.slice(begin, end);

    /*
    std::cerr << "hip::GlobalMemoryQueueInfo : used "
              << (reinterpret_cast<std::byte*>(cpu_trace.end) -
                  reinterpret_cast<std::byte*>(cpu_trace.current))
              << '\n';
    */

    dumpEventsBin(out, stamp, buf, GlobalMemoryQueueInfo::event_desc,
                  GlobalMemoryQueueInfo::event_name, end - begin,
                  registry.buffer_size);

    queue_info->notifyDoneProcessing();

    lttng_ust_tracepoint(hip_instrumentation, collector_dump_end,
                         registry.begin, stamp);
}

template <>
void HipTraceManager::handlePayload(CUMemoryQueuePayload&& payload,
                                    std::ofstream& out) {
    auto& [queue_info, stamp, begin_registries, end_registries] = payload;
    lttng_ust_tracepoint(hip_instrumentation, collector_dump_queue, &out,
                         queue_info->toDevice(), stamp);
    // TODO

    queue_info->notifyDoneProcessing();
    lttng_ust_tracepoint(hip_instrumentation, collector_dump_end,
                         queue_info->toDevice(), stamp);
}

template <>
void HipTraceManager::handlePayload(ChunkAllocatorEventsQueuePayload&& payload,
                                    std::ofstream& out) {
    auto& [allocator, stamp, registry, begin] = payload;

    lttng_ust_tracepoint(hip_instrumentation, collector_dump_queue, &out,
                         registry.begin, stamp);

    auto end = registry.current_id;

    auto buf = registry.slice(begin, end);

    dumpEventsBin(out, stamp, buf, ChunkAllocator::event_desc,
                  ChunkAllocator::event_name, end - begin,
                  registry.buffer_size);
    allocator->notifyDoneProcessing();
    lttng_ust_tracepoint(hip_instrumentation, collector_dump_end,
                         registry.begin, stamp);
}

template <>
void HipTraceManager::handlePayload(
    CUChunkAllocatorEventsQueuePayload&& payload, std::ofstream& out) {
    auto& [cu_allocator, stamp, begin_registries, end_registries] = payload;

    lttng_ust_tracepoint(hip_instrumentation, collector_dump_queue, &out,
                         cu_allocator->toDevice(), stamp);

    std::array<std::unique_ptr<std::byte[]>, CUChunkAllocator::TOTAL_CU_COUNT>
        buffers;

    std::array<size_t, CUChunkAllocator::TOTAL_CU_COUNT> sizes;

    cu_allocator->fetchBuffers(*begin_registries, *end_registries, buffers,
                               sizes);

    dumpEventsBin(out, stamp, buffers, sizes, CUChunkAllocator::event_desc,
                  CUChunkAllocator::event_name,
                  CUChunkAllocator::TOTAL_CU_COUNT,
                  cu_allocator->getRegistries()[0].reg.buffer_size);

    cu_allocator->notifyDoneProcessing();
    lttng_ust_tracepoint(hip_instrumentation, collector_dump_end,
                         cu_allocator->toDevice(), stamp);
}

template <class> inline constexpr bool always_false_v = false;

void HipTraceManager::runThread() {
    // Init output file
    auto now = std::chrono::steady_clock::now();
    uint64_t stamp = std::chrono::duration_cast<std::chrono::microseconds>(
                         now.time_since_epoch())
                         .count();

    std::stringstream filename;
    if (auto* env = std::getenv(HIPTRACE_ENV.data())) {
        filename << env << '_' << stamp << ".hiptrace";
    } else {
        filename << "hiptrace_" << stamp << ".hiptrace";
    }

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

size_t HipTraceManager::queuedPayloads() {
    std::lock_guard lock{mutex};

    return queue.size();
}

void HipTraceManager::flush() {
    while (!isEmpty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

HipTraceFile::HipTraceFile(std::string_view filename) : input(filename.data()) {
    // Check that file is open
    if (!input.is_open()) {
        throw std::runtime_error(
            "HipTraceFile::HipTraceFile() : Could not open file " +
            std::string(filename));
    }

    // Read header for trace type
    std::string buffer;
    if (!std::getline(input, buffer)) {
        throw std::runtime_error(
            "HipTraceFile::HipTraceFile() : Could not get header");
    }

    trace_kind = parseHeader(buffer);
    if (trace_kind == Kind::ErrorKind) {
        throw std::runtime_error(
            "HipTraceFile::HipTraceFile() : Non-valid token (\"" + buffer +
            "\")");
    }

    // update offset
    if (trace_kind == Kind::Managed) {
        offset += buffer.size();
    } else {
        // Have to rewind
        input.seekg(0);
    }
}

HipTraceFile::Kind HipTraceFile::parseHeader(std::string_view header) {
    auto end = header.find(',');

    auto header_trimmed = header.substr(0, end);

    if (header_trimmed == hiptrace_managed_name) {
        return Kind::Managed;
    } else if (header_trimmed == hiptrace_counters_name) {
        return Kind::Counters;
    } else if (header_trimmed == hiptrace_wave_counters_name) {
        return Kind::WaveCounters;
    } else if (header_trimmed == hiptrace_events_name) {
        return Kind::Events;
    } else if (header_trimmed == hiptrace_chunk_events_name) {
        return Kind::ChunkAlloc;
    } else if (header_trimmed == hiptrace_cu_chunk_events_name) {
        return Kind::CUChunkAlloc;
    } else {
        return Kind::ErrorKind;
    }
}

class Parser {
  public:
    Parser(const std::string& header, char _sep = ',') : sep(_sep) {
        ss << header;
    }

    std::string token() {
        std::string token;
        std::getline(ss, token, sep);
        return token;
    }

    template <typename T> T parse() {
        T val;
        auto tok = token();
        if (std::from_chars(tok.data(), tok.data() + tok.length(), val).ec !=
            std::errc()) {
            throw std::runtime_error("Could not parse token : " + tok);
        }
        return val;
    }

  private:
    std::stringstream ss;
    char sep;
};

template <typename Instr>
HipTraceManager::CountersQueuePayload<Instr>
loadInstr(const std::string& header, std::ifstream& f) {
    // Get header, construct kernel info
    Parser p{header};

    auto type_header = p.token(); // Ignore for now
    auto kernel_name = p.token();
    auto instr_size = p.parse<uint64_t>();
    auto stamp = p.parse<uint64_t>();

    std::pair<uint64_t, uint64_t> interval;
    interval.first = p.parse<uint64_t>();
    interval.second = p.parse<uint64_t>();

    auto num_counters = p.parse<uint64_t>();
    auto counter_size = p.parse<uint64_t>();

    // Kernel info
    dim3 blocks, threads;
    uint64_t bb;

    auto geometry_header = p.token();
    bb = p.parse<uint64_t>();
    blocks.x = p.parse<uint64_t>();
    blocks.y = p.parse<uint64_t>();
    blocks.z = p.parse<uint64_t>();

    threads.x = p.parse<uint64_t>();
    threads.y = p.parse<uint64_t>();
    threads.z = p.parse<uint64_t>();

    KernelInfo ki(kernel_name, bb, blocks, threads);
    // ki.dump();

    // Read counters
    Instr vec;
    vec.resize(num_counters);
    f.read(reinterpret_cast<char*>(vec.data()), num_counters * counter_size);

    return {vec, ki, stamp, interval};
}

HipTraceManager::EventsQueuePayload
HipTraceFile::loadEvents(const std::string& header, std::ifstream& f) const {
    // Get header, construct queue info
    Parser p{header};

    auto assert_header = [](const std::string& token,
                            std::string_view expected) {
        if (token != expected) {
            throw std::runtime_error("Unexpected header \"" + token +
                                     "\", expected " + std::string(expected));
        }
    };

    auto type_header = p.token();
    auto event_size = p.parse<uint64_t>();
    auto num_offsets = p.parse<uint64_t>() + 1;
    auto event_name = p.token();
    auto fields_header = p.token();

    assert_header(type_header, hiptrace_events_name);
    assert_header(fields_header, hiptrace_event_fields);

    // TODO : Check event members & sizes through header

    std::vector<size_t> offsets;
    offsets.resize(num_offsets);
    f.read(reinterpret_cast<char*>(offsets.data()),
           num_offsets * sizeof(size_t));

    auto num_events =
        offsets.back(); // Last offset is the total number of events

    std::vector<std::byte> events;
    events.resize(num_events * event_size);
    f.read(reinterpret_cast<char*>(events.data()), events.size());

    return {nullptr, QueueInfo{event_size, type_header, type_header,
                               std::move(offsets), std::move(events)}};
}

HipTraceManager::ChunkAllocatorEventsQueuePayload
loadChunkAlloc(const std::string& header, std::ifstream& f) {
    // Get header, construct kernel info
    Parser p{header};

    auto type_header = p.token(); // Ignore for now
    auto stamp = p.parse<uint64_t>();
    auto buffer_count = p.parse<uint64_t>();
    auto buffer_size = p.parse<uint64_t>();
    auto event_name = p.token();
    auto fields_header = p.token();

    // TODO do something with the fields

    // Instantiate a new chunk alloc & copy to buffer
    auto* alloc = new ChunkAllocator(buffer_count, buffer_size, false);
    f.read(reinterpret_cast<char*>(alloc->cpuBuffer()),
           buffer_count * buffer_size);

    return {alloc, stamp, alloc->getRegistry(), 0ull};
}

HipTraceManager::CUChunkAllocatorEventsQueuePayload
loadCUChunkAlloc(const std::string& header, std::ifstream& f) {
    // Get header, construct kernel info
    Parser p{header};

    auto type_header = p.token(); // Ignore for now
    auto stamp = p.parse<uint64_t>();
    auto buffer_count = p.parse<uint64_t>();
    auto buffer_size = p.parse<uint64_t>();
    auto count = p.parse<uint64_t>();
    auto event_name = p.token();
    auto fields_header = p.token();

    // TODO do something with the fields

    // First step : get sizes

    std::vector<size_t> counts, offsets;
    counts.resize(count);
    offsets.resize(count);

    for (auto i = 0u; i < count; ++i) {
        std::string buf;
        if (!std::getline(f, buf)) {
            throw std::runtime_error("loadCUChunkAlloc()");
        }
        Parser subparser{buf};

        counts[i] = subparser.parse<uint64_t>();
        offsets[i] = f.tellg();

        f.seekg(offsets[i] + counts[i] * buffer_size);
    }

    // Allocate

    auto* cu = new CUChunkAllocator{counts, buffer_size};

    // Fill buffers

    for (auto i = 0u; i < count; ++i) {
        f.seekg(offsets[i]);
        f.read(reinterpret_cast<char*>(cu->getRegistries()[i].reg.begin),
               counts[i] * buffer_size);
    }

    return {cu, stamp, nullptr, nullptr};
}

HipTraceManager::Payload HipTraceFile::getNext() {
    // Get header
    std::string buffer;

    if (!std::getline(input, buffer)) {
        throw std::runtime_error(
            "HipTraceFile::getNext() : Could not get header");
    }

    auto event_kind = parseHeader(buffer);

    // Rewind, and re-run parser

    switch (event_kind) {
    case Kind::Counters:
        return loadInstr<HipTraceManager::ThreadCounters>(buffer, input);
    case Kind::WaveCounters:
        return loadInstr<HipTraceManager::WaveCounters>(buffer, input);
    case Kind::Events:
        return loadEvents(buffer, input);
    case Kind::ChunkAlloc:
        return loadChunkAlloc(buffer, input);
    case Kind::CUChunkAlloc:
        return loadCUChunkAlloc(buffer, input);
    case Kind::ErrorKind:
        throw std::runtime_error(
            "HipTraceFile::getNext() : Could not parse header " + buffer);
    default:
        throw std::runtime_error(
            "HipTraceManager::getNext() : Unexpected header");
    }
}

bool HipTraceFile::done() {
    return input.peek() == std::ifstream::traits_type::eof();
}

} // namespace hip
