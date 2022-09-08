/** \file hip_instrumentation.cpp
 * \brief Kernel instrumentation embedded code
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/hip_instrumentation.hpp"
#include "hip_instrumentation/hip_utils.hpp"
#include "hip_instrumentation/queue_info.hpp"

#include <chrono>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <variant>

// Jsoncpp (shipped with ubuntu & debian)

#include <json/json.h>

#include "rocprofiler/rocprofiler.h"
#include "roctracer.h"

namespace hip {

namespace {

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
    using EventsQueuePayload =
        std::tuple<std::vector<std::byte>, std::vector<size_t>, size_t,
                   std::string, std::string>;

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

    void registerQueue(QueueInfo& queue, std::vector<std::byte>&& queue_data);

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
                           std::pair<uint64_t, uint64_t> interval) {
    using counter_t = HipTraceManager::Counters::value_type;

    // Write header

    // Like "hiptrace_counters,<kernel name>,<num
    // counters>,<stamp>,<stamp_begin>,<stamp_end>,<counter size>\n"

#ifdef HIPANALYZER_BIN_OFFSETS
    std::cout << "Counters @ " << out.tellp() << '\n';
#endif

    out << hiptrace_counters_name << ',' << kernel_info.name << ','
        << kernel_info.instr_size << ',' << stamp << ',' << interval.first
        << ',' << interval.second << ','
        << static_cast<unsigned int>(sizeof(counter_t)) << ',';

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

    out.write(reinterpret_cast<const char*>(counters.data()),
              counters.size() * sizeof(counter_t));

    return out;
}

std::ostream& dumpEventsBin(std::ostream& out,
                            std::vector<std::byte>& queue_data,
                            std::vector<size_t>& offsets, size_t event_size,
                            std::string_view event_desc,
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

} // namespace

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

void HipTraceManager::registerCounters(Instrumenter& instr,
                                       Counters&& counters) {
    std::lock_guard lock{mutex};
    queue.push({CountersQueuePayload{std::move(counters), instr.kernelInfo(),
                                     instr.getStamp(), instr.getInterval()}});

    cond.notify_one();
}

void HipTraceManager::registerQueue(QueueInfo& queue_info,
                                    std::vector<std::byte>&& queue_data) {
    std::lock_guard lock{mutex};

    std::vector offsets(queue_info.offsets().begin(),
                        queue_info.offsets().end());

    queue.push({EventsQueuePayload{std::move(queue_data), std::move(offsets),
                                   queue_info.elemSize(), queue_info.getDesc(),
                                   queue_info.getName()}});

    cond.notify_one();
}

/** \brief Small header to validate the trace type
 */
constexpr auto hiptrace_managed_name = "hiptrace_managed";

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
            [&](auto&& val) {
                using T = std::decay_t<decltype(val)>;
                if constexpr (std::is_same_v<T, CountersQueuePayload>) {
                    auto& [counters, kernel_info, stamp, rocm_stamp] = val;

                    dumpTraceBin(out, counters, kernel_info, stamp, rocm_stamp);
                } else if constexpr (std::is_same_v<T, EventsQueuePayload>) {
                    auto& [events, offsets, size, desc, name] = val;

                    dumpEventsBin(out, events, offsets, size, desc, name);
                } else {
                    static_assert(always_false_v<T>, "Non-exhaustive visitor");
                }
            },
            *payload);
    }
}

// GCN Assembly

/** \brief Saves the EXEC registers in two VGPRs (variables h & l)
 */
constexpr auto save_register =
    "asm volatile (\"s_mov_b32 s6, exec_lo\\n s_mov_b32 s7, exec_hi\\n "
    "v_mov_b32 %0, s6\\n v_mov_b32 %1, s7\":\"=v\" (l), \"=v\" (h):);";

// ---- Instrumentation ----- //

void KernelInfo::dump() const {
    std::cout << "Kernel info (" << name << ") :\n"
              << "\tTotal blocks : " << total_blocks << '\n'
              << "\tTotal threads : " << total_threads_per_blocks << '\n'
              << "\tBasic blocks : " << basic_blocks << '\n'
              << "\tInstr size : " << instr_size << '\n';
}

std::string KernelInfo::json() {
    std::stringstream ss;

    auto t_x = threads_per_blocks.x, t_y = threads_per_blocks.y,
         t_z = threads_per_blocks.z;

    auto b_x = blocks.x, b_y = blocks.y, b_z = blocks.z;

    // Sorry about this monstruosity, but trust me it works (I think?)
    ss << "{\"name\":\"" << name << "\",\"bblocks\":" << basic_blocks
       << ",\"geometry\":{\"threads\":{\"x\":" << t_x << ",\"y\":" << t_y
       << ",\"z\":" << t_z << "},\"blocks\":{\"x\":" << b_x << ",\"y\":" << b_y
       << ",\"z\":" << b_z << "}}}";

    return ss.str();
}

dim3 dim3FromJson(const Json::Value& root) {
    auto x = root.get("x", 1u).asUInt();
    auto y = root.get("y", 1u).asUInt();
    auto z = root.get("z", 1u).asUInt();

    return {x, y, z};
}

KernelInfo KernelInfo::fromJson(const std::string& filename) {
    Json::Value root;

    std::ifstream file_in(filename);

    file_in >> root;

    auto geometry = root.get("geometry", Json::Value());

    dim3 blocks = dim3FromJson(geometry.get("blocks", Json::Value()));
    dim3 threads = dim3FromJson(geometry.get("threads", Json::Value()));
    unsigned int bblocks = root.get("bblocks", 0u).asUInt();
    std::string kernel_name = root.get("name", "").asString();

    return {kernel_name, bblocks, blocks, threads};
}

Instrumenter::Instrumenter(KernelInfo& ki)
    : kernel_info(ki), host_counters(ki.instr_size, 0u) {

    // Get the timestamp for unique identification
    auto now = std::chrono::steady_clock::now();
    stamp = std::chrono::duration_cast<std::chrono::microseconds>(
                now.time_since_epoch())
                .count();
}

uint64_t getRoctracerStamp() {
    uint64_t rt_timestamp, ret;

    roctracer_status_t err = roctracer_get_timestamp(&rt_timestamp);
    if (err != ROCTRACER_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(
                "hip::Instrumenter::toDevice() : Could not get timestamp") +
            roctracer_error_string());
    }

    rocprofiler_get_time(ROCPROFILER_TIME_ID_CLOCK_REALTIME, rt_timestamp, &ret,
                         NULL);

    return ret;
}

Instrumenter::counter_t* Instrumenter::toDevice() {
    counter_t* data_device;
    auto size = kernel_info.instr_size * sizeof(counter_t);

    hip::check(hipMalloc(&data_device, size));

    hip::check(hipMemcpy(data_device, host_counters.data(),
                         kernel_info.instr_size * sizeof(counter_t),
                         hipMemcpyHostToDevice));

    hip::check(hipMemset(data_device, 0u, size));

    // We get the timestamp at this point because the toDevice method is
    // executed right before the kernel launch

    stamp_begin = getRoctracerStamp();

    return data_device;
}

void Instrumenter::fromDevice(void* device_ptr) {
    // Likewise, the fromDevice method is executed right after the end of
    // the kernel launch

    stamp_end = getRoctracerStamp();

    hip::check(hipMemcpy(host_counters.data(), device_ptr,
                         kernel_info.instr_size * sizeof(counter_t),
                         hipMemcpyDeviceToHost));
}

std::string Instrumenter::autoFilenamePrefix() const {
    std::stringstream ss;
    ss << kernel_info.name << '_' << stamp;

    return ss.str();
}

constexpr auto csv_header = "block,thread,bblock,count";

void Instrumenter::dumpCsv(const std::string& filename_in) {
    std::string filename;

    if (filename_in.empty()) {
        filename = autoFilenamePrefix() + ".csv";
    } else {
        filename = filename_in;
    }

    std::ofstream out(filename);
    out << csv_header << '\n';

    for (auto block = 0; block < kernel_info.total_blocks; ++block) {
        for (auto thread = 0; thread < kernel_info.total_threads_per_blocks;
             ++thread) {
            for (auto bblock = 0; bblock < kernel_info.basic_blocks; ++bblock) {
                auto index = block * kernel_info.total_threads_per_blocks *
                                 kernel_info.basic_blocks +
                             thread * kernel_info.basic_blocks + bblock;

                out << block << ',' << thread << ',' << bblock << ','
                    << static_cast<unsigned int>(host_counters[index]) << '\n';
            }
        }
    }

    out.close();
}

void Instrumenter::dumpBin(const std::string& filename_in) {
    std::string filename;

    if (filename_in.empty()) {
        filename = autoFilenamePrefix() + ".hiptrace";
    } else {
        filename = filename_in;
    }

    std::ofstream out(filename, std::ios::binary);

    if (!out.is_open()) {
        throw std::runtime_error(
            "Instrumenter::dumpBin() : Could not open output file " + filename);
    }

    dumpTraceBin(out, host_counters, kernel_info, stamp, getInterval());

    out.close();

    std::ofstream db(filename + ".json");
    db << kernel_info.json();
    db.close();
}

size_t Instrumenter::loadCsv(const std::string& filename) {
    // Load from file
    std::ifstream in(filename);
    if (!in.is_open()) {
        throw std::runtime_error(
            "hip::Instrumenter::loadCsv() : Could not open file " + filename);
    }

    std::string buf;

    // Try to read the first line

    if (!std::getline(in, buf)) {
        throw std::runtime_error(
            "hip::Instrumenter::loadCsv() : Could not parse header");
    }

    // Compare csv header with expected header

    if (buf != csv_header) {
        throw std::runtime_error(
            "hip::Instrumenter::loadCsv() : Wrong header, expected " +
            std::string(csv_header) + ", got : " + buf);
    }

    unsigned int line_no = 0u;

    while (std::getline(in, buf)) {
        // Parse line
        std::stringstream ss;
        ss << buf;

        std::vector<std::string> tokens(4, "");

        for (auto i = 0u; i < 4; ++i) {
            if (!std::getline(ss, tokens[i], ',')) {
                throw std::runtime_error("Could not parse token from line " +
                                         ss.str());
            }
        }

        host_counters[line_no] =
            static_cast<counter_t>(std::atol(tokens[3].c_str()));

        std::cout << static_cast<unsigned int>(host_counters[line_no]) << '\n';
        ++line_no;

        // We can discard the rest of the information, since it is implied
        // by the kernel call geometry
    }

    return line_no;
}

bool Instrumenter::parseHeader(const std::string& header) {
    std::stringstream ss;
    ss << header;

    auto get_token = [&]() -> std::string {
        std::string buf;
        if (!std::getline(ss, buf, ',')) {
            throw std::runtime_error("hip::Instrumenter::parseHeader() : Could "
                                     "not read token from header : " +
                                     header);
        }

        return std::move(buf);
    };

    auto trace_type = get_token();
    if (trace_type != hiptrace_counters_name) {
        return false;
    }

    auto kernel_name =
        get_token(); // Is a different kernel name a reason to fail?

    auto instr_size = std::stoul(get_token());
    if (instr_size != kernel_info.instr_size) {
        return false;
        //"hip::Instrumenter::parseHeader() : Incompatible counter number,
        // faulty database?"
    }

    auto stamp_in = std::stoull(get_token());
    stamp = stamp_in;

    auto counter_size = std::stoul(get_token());
    if (counter_size != kernel_info.instr_size) {
        return false;
    }

    return true;
}

size_t Instrumenter::loadBin(const std::string& filename) {
    // Load from file

    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error(
            "hip::Instrumenter::loadBin() : Could not open file " + filename);
    }

    std::string buffer;
    if (!std::getline(in, buffer)) {
        throw std::runtime_error(
            "hip::Instrumenter::loadBin() : Could not read header " + filename);
    }

    if (!parseHeader(buffer)) {
        throw std::runtime_error(
            "hip::Instrumenter::loadBin() : Incompatible header : " + buffer);
    }

    // Ugly cast, but works
    in.read(reinterpret_cast<char*>(host_counters.data()),
            host_counters.size() * sizeof(counter_t));

    return in.gcount();
}

const std::vector<hip::BasicBlock>&
Instrumenter::loadDatabase(const std::string& filename_in) {
    std::string filename;

    if (filename_in.empty()) {
        filename = BasicBlock::getEnvDatabaseFile(kernel_info.name);
    } else {
        filename = filename_in;
    }

    blocks = BasicBlock::fromJsonArray(filename);

    return blocks;
}

void Instrumenter::record() {
    auto& trace_manager = HipTraceManager::getInstance();

    trace_manager.registerCounters(*this, std::move(host_counters));
}

void QueueInfo::record() {
    auto& trace_manager = HipTraceManager::getInstance();

    trace_manager.registerQueue(*this, std::move(cpu_queue));
}

} // namespace hip
