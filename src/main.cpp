/** \file main.cpp
 * \brief Main entry point
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

#include "llvm/Support/CommandLine.h"

static llvm::cl::OptionCategory llvmClCategory("HipAnalyzer options");

static llvm::cl::extrahelp
    CommonHelp(clang::tooling::CommonOptionsParser::HelpMessage);

static llvm::cl::extrahelp MoreHelp("\nTODO: Extra help\n");

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

    return tool.run(
        clang::tooling::newFrontendActionFactory<clang::SyntaxOnlyAction>()
            .get());
}
