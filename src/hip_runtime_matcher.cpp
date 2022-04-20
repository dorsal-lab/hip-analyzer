/** \file hip_runtime_matcher.cpp
 * \brief AST Matcher for HIP intrinsics (threadIdx, etc.)
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang;
using namespace clang::ast_matchers;

namespace {

/** \brief HIP intrinsics matchers
 * \details HIP seems to have two distinct methods to access the kernel
 * geometry. The legacy one (prefixed with hip*) is a macro to the old OpenCL
 * runtime, while the second one uses more "modern" C++ structs (which no doubt
 * call the same functions under the hood). We however have to match againts
 * both of those to be generic enough!
 */

// OpenCL type functions (referenced using hipThreadIdx, ...) -> macros to ockl
// intrinsics
const auto hipThreadIdx = "__ockl_get_local_id";
const auto hipBlockIdx = "__ockl_get_group_id";
const auto hipBlockdim = "__ockl_get_local_size";
const auto hipGridDim = "__ockl_get_num_groups";

// Modern struct access : threadIdx.x, ...
const auto geometry = "__HIP_Coordinates";
// __HIP_Coordinates is a template type : we need to find the bound type to
// identify which coord is accessed
const auto threadIdx_t = "__HIP_ThreadIdx";
const auto blockIdx_t = "__HIP_BlockIdx";
const auto blockdim_t = "__HIP_BlockDim";
const auto gridDim_t = "__HIP_GridDim";

} // namespace

namespace hip {

StatementMatcher function_call_matcher =
    callExpr(callee(functionDecl(hasName(hipThreadIdx)))).bind(hipThreadIdx);

StatementMatcher geometry_matcher =
    memberExpr(
        hasDeclaration(decl(hasAncestor(cxxRecordDecl(hasName(threadIdx_t))))))
        .bind(threadIdx_t);

class FunctionPrinter : public MatchFinder::MatchCallback {
  public:
    virtual void run(const MatchFinder::MatchResult& Result) {

        // ThreadIdx
        if (const auto* FS =
                Result.Nodes.getNodeAs<clang::CallExpr>(hipThreadIdx)) {
            FS->dump();
        }

        if (const auto* FS =
                Result.Nodes.getNodeAs<clang::MemberExpr>(geometry)) {
            FS->dump();
        }
    }
};

std::unique_ptr<MatchFinder::MatchCallback> makeFunPrinter() {
    return std::make_unique<FunctionPrinter>();
}

} // namespace hip
