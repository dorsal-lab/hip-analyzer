/** \file hip_instrumentation.cpp
 * \brief Kernel instrumentation embedded code
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/hip_instrumentation.hpp"
#include "hip_instrumentation/hip_trace_manager.hpp"
#include "hip_instrumentation/hip_utils.hpp"
#include "hip_instrumentation/state_recoverer.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <variant>

// Jsoncpp (shipped with ubuntu & debian)

#include <json/json.h>

#include "rocprofiler/v2/rocprofiler.h"

namespace hip {

// The order in which these are destroyed may cause a deadlock, due to the
// HipMemoryManager hanging on hipFree if it is called after destruction from
// the fs_thread
std::unique_ptr<HipMemoryManager> HipMemoryManager::instance;
std::unique_ptr<HipTraceManager> HipTraceManager::instance;

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

uint32_t KernelInfo::wavefrontCount() const {
    uint32_t wavefronts_per_blocks = total_threads_per_blocks / wavefrontSize;
    if (total_threads_per_blocks % wavefrontSize > 0) {
        ++wavefronts_per_blocks;
    }

    return wavefronts_per_blocks * total_blocks;
}

std::unordered_map<std::string, std::vector<hip::BasicBlock>>
    CounterInstrumenter::known_blocks;

bool CounterInstrumenter::rocprofiler_initializer = false;

CounterInstrumenter::CounterInstrumenter(KernelInfo& ki)
    : CounterInstrumenter() {
    kernel_info.emplace(ki);
}

CounterInstrumenter::CounterInstrumenter() {
    if (!rocprofiler_initializer) {
        rocprofiler_initialize();
        rocprofiler_initializer = true;
    }
    // Get the timestamp for unique identification
    auto now = std::chrono::steady_clock::now();
    stamp = std::chrono::duration_cast<std::chrono::microseconds>(
                now.time_since_epoch())
                .count();
}

uint64_t getRoctracerStamp() {
    rocprofiler_timestamp_t rt_timestamp;

    rocprofiler_status_t err = rocprofiler_get_timestamp(&rt_timestamp);
    if (err != ROCPROFILER_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(
                "hip::Instrumenter::toDevice() : Could not get timestamp") +
            rocprofiler_error_str(err));
    }

    return rt_timestamp.value;
}

void* CounterInstrumenter::toDevice(size_t size) {
    void* data_device;

    hip::check(hipMalloc(&data_device, size));
    hip::check(hipMemset(data_device, 0u, size));

    // We get the timestamp at this point because the toDevice method is
    // executed right before the kernel launch

    stamp_begin = getRoctracerStamp();

    return data_device;
}

void* ThreadCounterInstrumenter::toDevice() {
    auto size = kernel_info->instr_size * sizeof(counter_t);
    return CounterInstrumenter::toDevice(size);
}

void ThreadCounterInstrumenter::fromDevice(void* device_ptr) {
    // Likewise, the fromDevice method is executed right after the end of
    // the kernel launch

    stamp_end = getRoctracerStamp();

    hip::check(hipMemcpy(host_counters.data(), device_ptr,
                         kernel_info->instr_size * sizeof(counter_t),
                         hipMemcpyDeviceToHost));
}

std::string CounterInstrumenter::autoFilenamePrefix() const {
    std::stringstream ss;
    ss << kernel_info->name << '_' << stamp;

    return ss.str();
}

constexpr auto csv_header = "block,thread,bblock,count";

void ThreadCounterInstrumenter::dumpCsv(const std::string& filename_in) {
    std::string filename;

    if (filename_in.empty()) {
        filename = autoFilenamePrefix() + ".csv";
    } else {
        filename = filename_in;
    }

    std::ofstream out(filename);
    out << csv_header << '\n';

    for (auto block = 0u; block < kernel_info->total_blocks; ++block) {
        for (auto thread = 0u; thread < kernel_info->total_threads_per_blocks;
             ++thread) {
            for (auto bblock = 0u; bblock < kernel_info->basic_blocks;
                 ++bblock) {
                auto index = block * kernel_info->total_threads_per_blocks *
                                 kernel_info->basic_blocks +
                             thread * kernel_info->basic_blocks + bblock;

                out << block << ',' << thread << ',' << bblock << ','
                    << static_cast<unsigned int>(host_counters[index]) << '\n';
            }
        }
    }

    out.close();
}

void ThreadCounterInstrumenter::dumpBin(const std::string& filename_in) {
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

    dumpTraceBin(out, host_counters, *kernel_info, stamp, getInterval());

    out.close();

    std::ofstream db(filename + ".json");
    db << kernel_info->json();
    db.close();
}

size_t ThreadCounterInstrumenter::loadCsv(const std::string& filename) {
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

bool ThreadCounterInstrumenter::parseHeader(const std::string& header) {
    std::stringstream ss;
    ss << header;

    auto get_token = [&]() -> std::string {
        std::string buf;
        if (!std::getline(ss, buf, ',')) {
            throw std::runtime_error("hip::Instrumenter::parseHeader() : Could "
                                     "not read token from header : " +
                                     header);
        }

        return buf;
    };

    auto trace_type = get_token();
    if (trace_type != hiptrace_counters_name) {
        return false;
    }

    auto kernel_name =
        get_token(); // Is a different kernel name a reason to fail?

    auto instr_size = std::stoul(get_token());
    if (instr_size != kernel_info->instr_size) {
        return false;
        //"hip::Instrumenter::parseHeader() : Incompatible counter number,
        // faulty database?"
    }

    auto stamp_in = std::stoull(get_token());
    stamp = stamp_in;

    auto counter_size = std::stoul(get_token());
    if (counter_size != kernel_info->instr_size) {
        return false;
    }

    return true;
}

size_t ThreadCounterInstrumenter::loadBin(const std::string& filename) {
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

std::string CounterInstrumenter::getDatabaseName() const {
    namespace fs = std::filesystem;

    if (auto* env = std::getenv(HIP_ANALYZER_ENV)) {
        return env;
    } else if (kernel_info && fs::exists(kernel_info->name + ".json")) {
        return kernel_info->name + ".json";
    } else if (fs::exists(HIP_ANALYZER_DEFAULT_FILE)) {
        return HIP_ANALYZER_DEFAULT_FILE;
    } else {
        throw std::runtime_error(
            "hip::Instrumenter::loadDatabase() : Could not file database file");
    }
}

const std::vector<hip::BasicBlock>& CounterInstrumenter::loadDatabase() {
    if (!kernel_info.has_value()) {
        throw std::runtime_error("hip::Instrumenter::loadDatabase() : Unknown "
                                 "kernel name, cannot load database");
    }

    return loadDatabase(kernel_info->name);
}

const std::vector<hip::BasicBlock>&
CounterInstrumenter::loadDatabase(const std::string& kernel_name) {
    // First, attempt to get the data from the stored database to avoid parsing
    // json (slow)
    auto it = known_blocks.find(kernel_name);
    if (it != std::end(known_blocks)) {
        blocks = &it->second;
        return it->second;
    } else {
        return loadDatabase(getDatabaseName(), kernel_name);
    }
}

const std::vector<hip::BasicBlock>&
CounterInstrumenter::loadDatabase(const std::string& filename_in,
                                  const std::string& kernel_name) {
    auto loaded_blocks = BasicBlock::fromJsonArray(filename_in, kernel_name);
    auto entry = known_blocks.emplace(kernel_name, std::move(loaded_blocks));

    blocks = &entry.first->second;

    return *blocks;
}

void ThreadCounterInstrumenter::record() {
    auto& trace_manager = HipTraceManager::getInstance();

    trace_manager.registerThreadCounters(*this, std::move(host_counters));
}

// ----- hip::WaveCounterInstrumenter ----- //
void* WaveCounterInstrumenter::toDevice() {
    auto size = host_counters.size() * sizeof(counter_t);
    return CounterInstrumenter::toDevice(size);
}

void WaveCounterInstrumenter::fromDevice(void* device_ptr) {
    // Likewise, the fromDevice method is executed right after the end of
    // the kernel launch

    stamp_end = getRoctracerStamp();

    hip::check(hipMemcpy(host_counters.data(), device_ptr,
                         instr_size * sizeof(counter_t),
                         hipMemcpyDeviceToHost));
}

void WaveCounterInstrumenter::dumpCsv(const std::string& filename) {}

void WaveCounterInstrumenter::dumpBin(const std::string& filename) {}

void WaveCounterInstrumenter::record() {
    auto& trace_manager = HipTraceManager::getInstance();

    trace_manager.registerWaveCounters(*this, std::move(host_counters));
}

size_t WaveCounterInstrumenter::loadCsv(const std::string& filename) {}
size_t WaveCounterInstrumenter::loadBin(const std::string& filename) {}

} // namespace hip
