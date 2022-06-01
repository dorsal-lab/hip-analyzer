/** \file load_bin_dump
 * \brief Load a binary trace from a file
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/hip_instrumentation.hpp"

#include "llvm/Support/CommandLine.h"

static llvm::cl::opt<std::string>
    kernel_geometry("k", llvm::cl::desc("Kernel launch geometry"),
                    llvm::cl::value_desc("kernel_info"), llvm::cl::Required);

static llvm::cl::opt<std::string> hiptrace("t", llvm::cl::desc("Hiptrace file"),
                                           llvm::cl::value_desc("hiptrace"),
                                           llvm::cl::Required);

int main(int argc, char** argv) {
    llvm::cl::ParseCommandLineOptions(argc, argv);

    auto kernel_info = hip::KernelInfo::fromJson(kernel_geometry.getValue());

    kernel_info.dump();

    hip::Instrumenter instrumenter(kernel_info);

    instrumenter.loadBin(hiptrace.getValue());
}
