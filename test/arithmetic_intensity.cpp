/** \file arithmetic_intensity.cpp
 * \brief Compute a compute kernel's arithmetic intensity
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

static llvm::cl::opt<std::string>
    database("d", llvm::cl::desc("Hip-analyzer database"),
             llvm::cl::value_desc("database"),
             llvm::cl::init(hip::default_database));

int main(int argc, char** argv) {
    llvm::cl::ParseCommandLineOptions(argc, argv);

    // Load kernel info
    auto kernel_info = hip::KernelInfo::fromJson(kernel_geometry.getValue());
    kernel_info.dump();

    // Load binary dump of counters
    hip::ThreadCounterInstrumenter instrumenter(kernel_info);
    auto elements = instrumenter.loadBin(hiptrace.getValue());

    std::cout << "Read " << elements << '\n';

    // Load database
    const auto& blocks = instrumenter.loadDatabase(database.getValue());

    // Compute AI
    uint64_t total_flops = 0u;
    uint64_t total_memory = 0u;

    for (auto i = 0u; i < kernel_info.instr_size; ++i) {
        auto bb = i % kernel_info.basic_blocks;
        auto& block = blocks[bb];
        total_flops += static_cast<uint64_t>(block.flops);
        total_memory +=
            static_cast<uint64_t>(block.floating_ld + block.floating_st);
    }

    auto arithmetic_intensity =
        static_cast<float>(total_flops) / static_cast<float>(total_memory);

    std::cout << "Kernel arithmetic intensity : " << arithmetic_intensity
              << " FLOPs/byte\n";
}
