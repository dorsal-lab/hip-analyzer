/** \file gpu_benchmark.cpp
 * \brief Full GPU benchmarking for roofline modeling
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

// Std includes

#include <fstream>

// LLVM includes

#include "llvm/Support/CommandLine.h"

// Local includes

#include "hip_instrumentation/gpu_info.hpp"

static llvm::cl::opt<std::string> output("o", llvm::cl::desc("Output file"),
                                         llvm::cl::value_desc("output"),
                                         llvm::cl::init("gpu_info.json"));

int main(int argc, char** argv) {
    llvm::cl::ParseCommandLineOptions(argc, argv);

    hip::GpuInfo gpu_info;

    // Perform benchmark

    // TODO

    // Save to file

    std::ofstream out(output.getValue(), std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Could not open output file " +
                                 output.getValue());
    }

    out << gpu_info.json() << '\n';
    out.close();

    return 0;
}
