/** \file main.cpp
 * \brief Main entry point
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/FrontendActions.h"
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
    auto printer = hip::makeCfgPrinter(kernel_name.getValue(),
                                       output_file.getValue()); // redundant ?
    auto matcher = hip::cfgMatcher(kernel_name.getValue());

    finder.addMatcher(matcher, printer.get());

    return tool.run(clang::tooling::newFrontendActionFactory(&finder).get());
}
