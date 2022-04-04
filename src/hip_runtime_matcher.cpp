/** \file hip_runtime_matcher.cpp
 * \brief AST Matcher for HIP intrinsics (threadIdx, etc.)
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang;
using namespace clang::ast_matchers;

namespace hip {

StatementMatcher function_call_matcher = callExpr().bind("funcall");

class FunctionPrinter : public MatchFinder::MatchCallback {
  public:
    virtual void run(const MatchFinder::MatchResult& Result) {
        if (const auto* FS = Result.Nodes.getNodeAs<clang::CallExpr>("funcall"))
            FS->dump();
    }
};

std::unique_ptr<MatchFinder::MatchCallback> makeFunPrinter() {
    return std::make_unique<FunctionPrinter>();
}

} // namespace hip
