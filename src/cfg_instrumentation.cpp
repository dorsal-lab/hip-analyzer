/** \file cfg_instrumentation.cpp
 * \brief
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/CFG.h"

#include <memory>
#include <string>

using namespace clang;
using namespace clang::ast_matchers;

namespace hip {

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
