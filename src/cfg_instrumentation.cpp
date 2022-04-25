/** \file cfg_instrumentation.cpp
 * \brief
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/CFG.h"
#include "clang/Tooling/Core/Replacement.h"

#include "clang/Lex/Lexer.h"

#include <memory>
#include <sstream>
#include <string>

// Temporary
#include <iostream>

using namespace clang;
using namespace clang::ast_matchers;

namespace hip {

std::string generateBlockCode(unsigned int id) {
    std::stringstream ss;
    ss << "// BB" << id << '\n';
    return ss.str();
}

/** \class KernelFinder
 * \brief AST Matcher callback to fetch the function text
 */
class KernelFinder : public MatchFinder::MatchCallback {
  public:
    KernelFinder(const std::string& kernel_name) : name(kernel_name) {}

    virtual void run(const MatchFinder::MatchResult& Result) {
        if (const auto* match =
                Result.Nodes.getNodeAs<clang::FunctionDecl>(name)) {
            match->dump();
            auto body = match->getBody();
            auto cfg = CFG::buildCFG(match, body, Result.Context,
                                     clang::CFG::BuildOptions());
            auto lang_opt = Result.Context->getLangOpts();
            cfg->dump(lang_opt, true);

            // Add replacement for counter declaration

            // Add replacement for counter initialization

            // Print First elements
            for (auto block : *cfg.get()) {
                auto id = block->getBlockID();

                std::cout << "\nBlock " << id << '\n';

                auto first = block->front();
                auto first_statement = first.getAs<clang::CFGStmt>();

                if (first_statement.hasValue()) {
                    auto stmt = first_statement->getStmt();
                    auto begin_loc = stmt->getBeginLoc();
                    begin_loc.dump(*Result.SourceManager);

                    auto rep_loc = clang::Lexer::getLocForEndOfToken(
                        begin_loc, 0, *Result.SourceManager, lang_opt);
                    rep_loc.dump(*Result.SourceManager);

                    stmt->dumpColor();

                    // Create replacement
                    clang::tooling::Replacement rep(*Result.SourceManager,
                                                    stmt->getBeginLoc(), 0,
                                                    generateBlockCode(id));

                    std::cout << rep.toString();
                }
            }
        }
        // TODO : store function decl
    }

    clang::FunctionDecl* getKernel() { return kernel; }

  private:
    const std::string name;
    clang::FunctionDecl* kernel = nullptr;
};

clang::ast_matchers::DeclarationMatcher
cfgMatcher(const std::string& kernel_name) {
    // Lifetime problem, todoooo
    return functionDecl(hasName(kernel_name)).bind(kernel_name);
}

std::unique_ptr<MatchFinder::MatchCallback>
makeCfgPrinter(const std::string& name) {
    return std::make_unique<KernelFinder>(name);
}

} // namespace hip
