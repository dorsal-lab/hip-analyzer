/** \file main.cpp
 * \brief Main entry point
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

#include "llvm/Support/CommandLine.h"

#include "callbacks.h"
#include "matchers.h"

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

int main(int argc, const char** argv) {
    auto parser =
        clang::tooling::CommonOptionsParser::create(argc, argv, llvmClCategory);

    if (!parser) {
        llvm::errs() << parser.takeError();
        return -1;
    }

    auto& options_parser = parser.get();
    clang::tooling::ClangTool tool(options_parser.getCompilations(),
                                   options_parser.getSourcePathList());

    clang::ast_matchers::MatchFinder finder;

    /*
    auto printer = hip::makeFunPrinter();

    finder.addMatcher(hip::function_call_matcher, printer.get());
    finder.addMatcher(hip::geometry_matcher, printer.get());
    */

    // Kernel matcher
    auto matcher = hip::kernelMatcher(kernel_name.getValue());

    // Instrument basic blocks
    auto instrumenter =
        hip::makeCfgInstrumenter(kernel_name.getValue(),
                                 output_file.getValue()); // redundant ?

    finder.addMatcher(matcher, instrumenter.get());

    auto err = 0;
    err |= tool.run(clang::tooling::newFrontendActionFactory(&finder).get());

    // Add instrumentation arguments and local variables

    /*clang::tooling::ClangTool param_tool(options_parser.getCompilations(),
                                         options_parser.getSourcePathList());*/
    clang::ast_matchers::MatchFinder param_finder;

    auto param_instrumenter = hip::makeBaseInstrumenter(kernel_name.getValue(),
                                                        output_file.getValue());
    param_finder.addMatcher(matcher, param_instrumenter.get());

    err |=
        tool.run(clang::tooling::newFrontendActionFactory(&param_finder).get());

    // Commit results

    return err;
}
