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
    ss << "{ \"geometry\": {\"threads\": {\"x\": " << t_x << ", \"y\": " << t_y
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

Instrumenter::counter_t* Instrumenter::toDevice() {
    counter_t* data_device;
    hip::check(
        hipMalloc(&data_device, kernel_info.instr_size * sizeof(counter_t)));

    hip::check(hipMemcpy(data_device, host_counters.data(),
                         kernel_info.instr_size * sizeof(counter_t),
                         hipMemcpyHostToDevice));

    return data_device;
}

void Instrumenter::fromDevice(void* device_ptr) {
    hip::check(hipMemcpy(host_counters.data(), device_ptr,
                         kernel_info.instr_size * sizeof(counter_t),
                         hipMemcpyDeviceToHost));
}

std::string Instrumenter::autoFilenamePrefix() const {
    std::stringstream ss;
    ss << kernel_info.name << '_' << stamp;

    return ss.str();
}

constexpr auto csv_header = "block,thread,bblock,count\n";

void Instrumenter::dumpCsv(const std::string& filename_in) {
    std::string filename;

    if (filename_in.empty()) {
        filename = autoFilenamePrefix() + ".csv";
    } else {
        filename = filename_in;
    }

    std::ofstream out(filename);
    out << csv_header;

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
    out.write(reinterpret_cast<const char*>(host_counters.data()),
              host_counters.size() * sizeof(counter_t));

    out.close();

    std::ofstream db(filename + ".json");
    db << kernel_info.json();
    db.close();
}

void Instrumenter::loadCsv(const std::string& filename) {
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

    while (std::getline(in, buf)) {
        // Parse line

        // TODO

        // We can discard the rest of the information, since it is implied by
        // the kernel call geometry
    }
}

void Instrumenter::loadBin(const std::string& filename) {
    // Load from file

    using buffer_iterator = std::istreambuf_iterator<counter_t>;

    std::basic_ifstream<counter_t> in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error(
            "hip::Instrumenter::loadBin() : Could not open file " + filename);
    }

    std::copy(buffer_iterator(in), buffer_iterator(), host_counters.begin());
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
