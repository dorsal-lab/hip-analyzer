/** \file main.cpp
 * \brief Main entry point
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

#include "llvm/Support/CommandLine.h"

#include "hip_instrumentation/basic_block.hpp"

#include "callbacks.h"
#include "llvm_ir_consumer.h"
#include "matchers.h"

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
static llvm::cl::opt<std::string>
    output_file("o", llvm::cl::desc("Output file path"),
                llvm::cl::value_desc("output"), llvm::cl::Required);
static llvm::cl::opt<std::string>
    database_file("db", llvm::cl::desc("Output database path"),
                  llvm::cl::value_desc("database"),
                  llvm::cl::init(hip::default_database));

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

    // Instrumentation info

    std::vector<hip::BasicBlock> blocks;

    // Kernel matcher
    clang::ast_matchers::MatchFinder finder;
    auto kernel_matcher = hip::kernelMatcher(kernel_name.getValue());
    auto kernel_call_matcher = hip::kernelCallMatcher(kernel_name.getValue());

    // Instrument basic blocks
    auto kernel_instrumenter = hip::makeCfgInstrumenter(
        kernel_name.getValue(), output_file.getValue(), blocks);

    /* auto kernel_call_instrumenter = hip::makeCudaCallInstrumenter(
        kernel_name.getValue(), output_file.getValue()); */

    finder.addMatcher(kernel_matcher, kernel_instrumenter.get());
    finder.addMatcher(kernel_call_matcher, kernel_instrumenter.get());

    auto err = 0;

    clang::tooling::ClangTool tool(db, options_parser.getSourcePathList());
    err |= tool.run(clang::tooling::newFrontendActionFactory(&finder).get());

    auto codegen = makeLLVMAction(kernel_name.getValue(), blocks);
    err |= tool.run(codegen.get());

    saveDatabase(blocks, database_file.getValue());

    return err;
}
