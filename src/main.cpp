/** \file main.cpp
 * \brief Main entry point
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

#include "llvm/Support/CommandLine.h"

#include "hip_instrumentation/basic_block.hpp"

#include "actions_processor.h"

// ----- Statics ----- //

#ifdef ROCM_PATH
#define ROCM_PATH_STR #ROCM_PATH
static std::string rocm_path = ROCM_PATH_STR;
#else
static std::string rocm_path = "/opt/rocm";
#endif

static llvm::cl::OptionCategory llvmClCategory("HipAnalyzer options");

static llvm::cl::extrahelp
    CommonHelp(clang::tooling::CommonOptionsParser::HelpMessage);

static llvm::cl::extrahelp MoreHelp("\nTODO: Extra help\n");

static llvm::cl::opt<std::string>
    kernel_name("k", llvm::cl::desc("Specify kernel name"),
                llvm::cl::value_desc("kernel"), llvm::cl::Required);

static llvm::cl::opt<bool> trace("trace",
                                 llvm::cl::desc("Enable finer tracing"),
                                 llvm::cl::value_desc("Trace kernel"));

static llvm::cl::opt<std::string>
    output_file("o", llvm::cl::desc("Output file path"),
                llvm::cl::value_desc("output"), llvm::cl::Required);

static llvm::cl::opt<std::string>
    database_file("db", llvm::cl::desc("Output database path"),
                  llvm::cl::value_desc("database"),
                  llvm::cl::init(hip::default_database));

static llvm::cl::opt<bool> include_original_call(
    "d",
    llvm::cl::desc("Duplicate : Keep a call to the original, non-instrumented "
                   "kernel with memory rollback"));

// ----- Utils ----- //

void appendFlag(clang::tooling::CompilationDatabase& db_in,
                const std::string& flag) {
    auto adjuster = clang::tooling::getInsertArgumentAdjuster(flag.c_str());

    // The reinterpret cast is ugly, but I was not able to compile LLVM with
    // RTTI which prohibits me from using a (much cleaner) dynamic cast
    auto& db =
        reinterpret_cast<clang::tooling::ArgumentsAdjustingCompilations&>(
            db_in);

    db.appendArgumentsAdjuster(adjuster);
}

void saveDatabase(const std::vector<hip::BasicBlock>& blocks,
                  const std::string& database_filename) {
    std::error_code err;
    llvm::raw_fd_ostream database_file(database_filename, err);

    database_file << hip::BasicBlock::jsonArray(blocks) << '\n';
}

// ----- Main entry point ----- //

int main(int argc, const char** argv) {
    auto parser =
        clang::tooling::CommonOptionsParser::create(argc, argv, llvmClCategory);

    if (!parser) {
        llvm::errs() << parser.takeError();
        return -1;
    }

    auto& options_parser = parser.get();
    auto& db = options_parser.getCompilations();
    appendFlag(db, "--cuda-device-only");
    appendFlag(db, "-I" + rocm_path + "/hip/bin/include");
    appendFlag(db, "-gline-directives-only");

    // Instrumentation info

    std::vector<hip::BasicBlock> blocks;
    auto err = 0;
    clang::tooling::ClangTool tool(db, options_parser.getSourcePathList());

    std::string main_path;

    for (auto f : options_parser.getSourcePathList()) {
        main_path = f;
    }

    // Kernel name
    auto kernel = kernel_name.getValue();
    auto instrumented_bb_name =
        hip::actions::InstrumentBasicBlocks::getInstrumentedKernelName(kernel);

    hip::ActionsProcessor actions(main_path, db, output_file.getValue());

    actions
        // Create a new kernel, with a new name
        .process(hip::actions::DuplicateKernel(kernel, instrumented_bb_name,
                                               err, include_original_call))
        // Instrument the newly created kernel
        .process(hip::actions::InstrumentBasicBlocks(instrumented_bb_name,
                                                     blocks, err));

    if (include_original_call) {
        // Replace calls to the original kernel to the new instrumented one
        actions.process(hip::actions::DuplicateKernelCall(
            kernel, instrumented_bb_name, err));
    } else {
        actions.process(
            hip::actions::ReplaceKernelCall(kernel, instrumented_bb_name, err));
    }

    if (trace) {
        // Duplicate again the kernel, this time with the trace-d version
        auto traced_kernel_name =
            hip::actions::TraceBasicBlocks::getInstrumentedKernelName(kernel);

        actions
            .process(hip::actions::DuplicateKernel(kernel, traced_kernel_name,
                                                   err, include_original_call))
            .process(
                hip::actions::TraceBasicBlocks(traced_kernel_name, blocks, err))
            .process(hip::actions::DuplicateKernelCall(kernel,
                                                       traced_kernel_name, err))
            .process(
                hip::actions::TraceKernelCall(traced_kernel_name, blocks, err));
    }

    actions
        // Add the necessary tools to instrument the kernel call
        .process(hip::actions::InstrumentKernelCall(
            instrumented_bb_name, blocks, err, include_original_call))
        // Analyze the original IR to get better info on the kernel execution
        .observeOriginal(hip::actions::AnalyzeIR(kernel, blocks, err));

    saveDatabase(blocks, database_file.getValue());

    return err;
}
