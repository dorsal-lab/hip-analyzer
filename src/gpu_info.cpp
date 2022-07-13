/** \file gpu_info.hpp
 * \brief GPU performance informations (bandwidth, FLOP/s)
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/gpu_info.hpp"

#include <sstream>

std::string hip::GpuInfo::json() const {
    std::stringstream ss;

    ss << "{\"name\":\"" << id << "\",\"memory_roofs\":[";

    for (const auto& roof : memory_roofs) {
        ss << "{\"name\":\"" << roof.name
           << "\",\"peak_bandwidth\":" << roof.peak_bandwidth << "},";
    }

    if (memory_roofs.size() != 0) {
        ss.seekp(-1, ss.cur);
    }

    ss << "],\"compute_roofs\":[";

    for (const auto& roof : compute_roofs) {
        ss << "{\"name\":\"" << roof.name
           << "\",\"peak_flops_s\":" << roof.peak_flops_s << "},";
    }

    if (compute_roofs.size() != 0) {
        ss.seekp(-1, ss.cur);
    }

    ss << "]}";

    return ss.str();
}

void hip::GpuInfo::benchmark() {
    // Memory benchmarks

    memory_roofs.emplace_back(benchmark::benchmarkGlobalMemoryBandwidth(0));
    memory_roofs.emplace_back(benchmark::benchmarkGlobalMemoryBandwidth(1));
    memory_roofs.emplace_back(benchmark::benchmarkGlobalMemoryBandwidth(2));
    memory_roofs.emplace_back(benchmark::benchmarkGlobalMemoryBandwidth(32));

    // Compute benchmarks

    compute_roofs.emplace_back(benchmark::benchmarkMultiplyFlops());
    compute_roofs.emplace_back(benchmark::benchmarkAddFlops());
    compute_roofs.emplace_back(benchmark::benchmarkFmaFlops());
}
