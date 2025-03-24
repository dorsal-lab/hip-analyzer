/** \file Optimal_tracing_pass.cpp
 * \brief Optimal tracing analysis pass implementation. Identify optimal
 * placement of instrumentation nodes
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "optimal_tracing_pass.h"

#include "llvm/IR/CFG.h"
#include "llvm/Transforms/Utils/UnifyFunctionExitNodes.h"

#include <algorithm>
#include <unordered_set>

using namespace llvm;

namespace {

template <typename T>
std::set<T> s_union(const std::set<T>& a, const std::set<T>& b) {
    std::set<T> c;
    std::set_union(a.begin(), a.end(), b.begin(), b.end(), c.back_inserter());
    return c;
}
} // namespace

namespace hip {

OptimalTracingPass::TracingSet
OptimalTracingPass::dfs(const BasicBlock* bb,
                        std::set<const BasicBlock*> explored_vertices) {

    TracingSet set;

    for (const auto& succ : successors(bb)) {
        if (explored_vertices.contains(bb)) {
            // If it is a back edge, add it to the tracing set
            set.insert({bb, succ});
        }
    }

    // Recursive call

    return set;
}

bool OptimalTracingPass::runOnFunction(Function& F) {
    analysis_result.clear();

    std::unordered_set<const BasicBlock*> work_set;

    auto* entry = &F.getEntryBlock();

    auto edges = dfs(entry);

    std::unordered_set<const BasicBlock*> end_cond = {exit_block};
    while (work_set != end_cond) {
        // Select a bb that isn't exit
        const llvm::BasicBlock* bb = nullptr;
        for (auto it = work_set.begin(); it != work_set.end(); ++it) {
            if (*it != exit_block) {
                bb = work_set.extract(*it).value();
                break;
            }
        }

        // Select edges
    }

    return false;
}

void OptimalTracingPass::getAnalysisUsage(AnalysisUsage& Info) const {
    // Info.addRequired<UnifyFunctionExitNodesPass>();
    Info.setPreservesAll();
}

void OptimalTracingPass::print(raw_ostream& O, Module const*) const {
    O << "OptimalTracingPass\n";
    for (auto& [in, out] : analysis_result) {
        O << in->getName() << " -> " << out->getName() << '\n';
    }
}

} // namespace hip
