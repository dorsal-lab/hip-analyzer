/** \file load_csv_dump
 * \brief Load a csv trace from a file
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/hip_instrumentation.hpp"

#include <iostream>

#include "llvm/Support/CommandLine.h"

static llvm::cl::opt<std::string>
    kernel_geometry("k", llvm::cl::desc("Kernel launch geometry"),
                    llvm::cl::value_desc("kernel_info"), llvm::cl::Required);

static llvm::cl::opt<std::string> hiptrace("t",
                                           llvm::cl::desc("csv trace file"),
                                           llvm::cl::value_desc("trace"),
                                           llvm::cl::Required);

int main(int argc, char** argv) {
    llvm::cl::ParseCommandLineOptions(argc, argv);

    auto kernel_info = hip::KernelInfo::fromJson(kernel_geometry.getValue());

    kernel_info.dump();

    hip::ThreadCounterInstrumenter instrumenter(kernel_info);

    auto elements = instrumenter.loadCsv(hiptrace.getValue());

    std::cout << "Read " << elements << '\n';
}
