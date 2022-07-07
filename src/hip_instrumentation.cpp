/** \file hip_instrumentation.cpp
 * \brief Kernel instrumentation embedded code
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/hip_instrumentation.hpp"
#include "hip_instrumentation/hip_utils.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

// Jsoncpp (shipped with ubuntu & debian)

#include <json/json.h>

#include "rocprofiler/rocprofiler.h"
#include "roctracer.h"

namespace hip {

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
    ss << "{ \"name\": \"" << name << "\", \"bblocks\": " << basic_blocks
       << ",\"geometry\": {\"threads\": {\"x\": " << t_x << ", \"y\": " << t_y
       << ", \"z\": " << t_z << "}, \"blocks\": {\"x\": " << b_x
       << ", \"y\": " << b_y << ", \"z\": " << b_z << "}}}";

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
    hip::check(
        hipMalloc(&data_device, kernel_info.instr_size * sizeof(counter_t)));

    hip::check(hipMemcpy(data_device, host_counters.data(),
                         kernel_info.instr_size * sizeof(counter_t),
                         hipMemcpyHostToDevice));

    // We get the timestamp at this point because the toDevice method is
    // executed right before the kernel launch

    stamp_begin = getRoctracerStamp();

    return data_device;
}

void Instrumenter::fromDevice(void* device_ptr) {
    // Likewise, the fromDevice method is executed right after the end of the
    // kernel launch

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

/** \brief Small header to validate the trace type
 */
constexpr auto hiptrace_name = "hiptrace";

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

    // Write header

    // Like "hiptrace,<kernel name>,<num
    // counters>,<stamp>,<stamp_begin>,<stamp_end>,<counter size>\n"

    out << hiptrace_name << ',' << kernel_info.name << ','
        << kernel_info.instr_size << ',' << stamp << ',' << stamp_begin << ','
        << stamp_end << ',' << static_cast<unsigned int>(sizeof(counter_t))
        << '\n';

    // Write binary dump of counters

    out.write(reinterpret_cast<const char*>(host_counters.data()),
              host_counters.size() * sizeof(counter_t));

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

        // We can discard the rest of the information, since it is implied by
        // the kernel call geometry
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
    if (trace_type != hiptrace_name) {
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

} // namespace hip
