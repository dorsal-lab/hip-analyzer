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

void print_bbs(const std::unordered_set<const BasicBlock*>& set) {
    llvm::dbgs() << '{';
    for (const auto& bb : set) {
        llvm::dbgs() << bb->getName() << ',';
    }
    llvm::dbgs() << "}\n";
}

void print_edges(const hip::OptimalTracingPassBase::TracingSet& set) {
    llvm::dbgs() << '{';
    for (const auto& [in, out] : set) {
        llvm::dbgs() << '(' << in->getName() << " -> " << out->getName()
                     << "),";
    }
    llvm::dbgs() << "}\n";
}

} // namespace

namespace hip {

OptimalTracingPassBase::TracingSet OptimalTracingPassBase::dfs(
    const BasicBlock* bb,
    std::unordered_set<const BasicBlock*> explored_vertices) {

    llvm::dbgs() << "DFS : " << bb->getName() << '\n';

    if (isa<ReturnInst>(bb->getTerminator())) {
        exit_block = bb;
        llvm::dbgs() << "\tTerminator\n";
        return {};
    }

    explored_vertices.insert(bb);

    TracingSet set;

    for (const auto& succ : successors(bb)) {
        if (explored_vertices.contains(succ)) {
            // If it is a back edge, add it to the tracing set
            set.insert({bb, succ});
            llvm::dbgs() << "\tSucc: " << succ->getName() << '\n';
        } // Do not explore a back edge
        else {
            // Maybe merge to avoid unnecessary copies & assignments?

            // Recursive call
            set = s_union(dfs(succ, explored_vertices), set);
        }
    }

    llvm::dbgs() << "\tSuccs ";
    print_edges(set);

    return set;
}

template <typename F>
std::vector<const BasicBlock*>
exclude_edges(F f, const BasicBlock* bb,
              const OptimalTracingPassBase::TracingSet& exclude_edges) {
    std::vector<const BasicBlock*> succs;
    for (const auto& succ : f(bb)) {
        bool may_be_added = true;
        for (const auto& [in, out] : exclude_edges) {
            if (in == bb && out == succ) {
                may_be_added = false;
                break;
            }
        }

        if (may_be_added) {
            succs.push_back(succ);
        }
    }

    return succs;
}

bool OptimalTracingPassBase::run(Function& F) {
    analysis_result.clear();

    F.dump();

    auto* entry = &F.getEntryBlock();
    std::unordered_set<const BasicBlock*> work_set({entry});
    std::unordered_set<const BasicBlock*> processed;

    auto back_edges = dfs(entry);
    print_edges(back_edges);

    // Init : Instrument all back edges
    analysis_result = back_edges;

    std::unordered_set<const BasicBlock*> end_cond = {exit_block};
    while (work_set != end_cond) {
        print_bbs(work_set);
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
        for (const auto& succ : exclude_edges(
                 [](auto* bb) { return successors(bb); }, bb, back_edges)) {
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

        processed.insert(bb);

        // Add successors to working set
        for (const auto& succ : exclude_edges(
                 [](auto* bb) { return successors(bb); }, bb, back_edges)) {
            if (succ != odd_one_out) {
                // Mark edges as instrumented
                analysis_result.insert({bb, succ});
                llvm::dbgs() << "\tInstr " << bb->getName() << ' '
                             << succ->getName() << '\n';
            }

            // Iterate over all its parents, if they have been processed then it
            // can be added in the work set
            bool may_be_processed = true;
            for (const auto& pred :
                 exclude_edges([](auto* bb) { return predecessors(bb); }, succ,
                               back_edges)) {
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

    print(llvm::errs(), F.getParent());
    return false;
}

std::set<const llvm::BasicBlock*>
OptimalTracingPassBase::getVertexTracingSet() const {
    std::set<const llvm::BasicBlock*> set;
    for (const auto& [in, out] : analysis_result) {
        set.insert(out);
    }
    return set;
}

void OptimalTracingPassLegacy::getAnalysisUsage(AnalysisUsage& Info) const {
    // Info.addRequired<UnifyFunctionExitNodesPass>();
    Info.setPreservesAll();
}

void OptimalTracingPassBase::print(raw_ostream& O, Module const*) const {
    O << "OptimalTracingPass\n";
    for (auto& [in, out] : analysis_result) {
        O << in->getName() << " -> " << out->getName() << '\n';
    }
}

llvm::AnalysisKey OptimalTracingPass::Key;

} // namespace hip
