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

template <typename T> T s_union(const T& a, const T& b) {
    T c = a;
    c.insert(b.begin(), b.end());
    return c;
}
} // namespace

namespace hip {

OptimalTracingPass::TracingSet OptimalTracingPass::dfs(
    const BasicBlock* bb,
    std::unordered_set<const BasicBlock*> explored_vertices) {

    TracingSet set;

    for (const auto& succ : successors(bb)) {
        if (explored_vertices.contains(bb)) {
            // If it is a back edge, add it to the tracing set
            set.insert({bb, succ});
        }
    }

    // Recursive call

    for (const auto& succ : successors(bb)) {
        // Maybe merge to avoid unnecessary copies & assignments?
        set = s_union(dfs(succ, s_union(explored_vertices, {bb})), set);
    }

    return set;
}

bool OptimalTracingPass::runOnFunction(Function& F) {
    analysis_result.clear();

    std::unordered_set<const BasicBlock*> work_set;
    std::unordered_set<const BasicBlock*> processed;

    auto* entry = &F.getEntryBlock();

    analysis_result = dfs(entry);

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

        llvm::errs() << "Working on BB " << bb->getName() << '\n';

        // Select edges. We need to extract an odd one out. First go through
        // successors, and identify if they may already be instrumented.
        const BasicBlock* odd_one_out = nullptr;
        for (const auto& succ : successors(bb)) {
            bool may_be_candidate = true;
            for (auto& [in, out] : analysis_result) {
                if (out == succ) {
                    may_be_candidate = false;
                    break;
                }
            }

            if (may_be_candidate) {
                odd_one_out = succ;
                break;
            }
        }

        if (odd_one_out == nullptr) {
            // Successors are not targets of back edges, so they have the same
            // cost (in theory). Select the first one
            odd_one_out = *successors(bb).begin();
        }

        // Add successors to working set
        for (const auto& succ : successors(bb)) {
            if (succ != odd_one_out) {
                // Mark edges as instrumented
                analysis_result.insert({bb, succ});
            }

            bool may_be_processed = true;
            for (const auto& pred : predecessors(succ)) {
                if (!processed.contains(pred)) {
                    may_be_processed = false;
                    break;
                }
            }

            // All predecessors have been processed. So the successor can be
            // analyzed.
            if (may_be_processed) {
                work_set.insert(succ);
            }
        }
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
