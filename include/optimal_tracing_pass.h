/** \file Optimal_tracing_pass.h
 * \brief Optimal tracing analysis pass. Identify optimal placement of
 * instrumentation nodes
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include "llvm/IR/BasicBlock.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

#include <set>
#include <unordered_set>

namespace hip {

class OptimalTracingPassBase {
  public:
    using TracingSet =
        std::set<std::tuple<const llvm::BasicBlock*, const llvm::BasicBlock*>>;

    bool run(llvm::Function&);
    const TracingSet& getTracingSet() const { return analysis_result; }

    void print(llvm::raw_ostream& O, llvm::Module const*) const;

  protected:
    TracingSet analysis_result;
    const llvm::BasicBlock* exit_block;

    TracingSet
    dfs(const llvm::BasicBlock* bb,
        std::unordered_set<const llvm::BasicBlock*> explored_vertices = {});
};

/** \class OptimalTracingPass
 * \brief Returns a set of edges in the CFG that need to be instrumented. Legacy
 * pass manager version
 */
class OptimalTracingPassLegacy : public llvm::FunctionPass,
                                 public OptimalTracingPassBase {
  public:
    static char ID;

    OptimalTracingPassLegacy();

    void print(llvm::raw_ostream& O, llvm::Module const* M) const override {
        OptimalTracingPassBase::print(O, M);
    }

    void getAnalysisUsage(llvm::AnalysisUsage& Info) const override;
    bool runOnFunction(llvm::Function& f) override { return run(f); }
};

class OptimalTracingPass : public llvm::AnalysisInfoMixin<OptimalTracingPass>,
                           public OptimalTracingPassBase {
  public:
    using Result = TracingSet;
    OptimalTracingPass() {}
    Result run(llvm::Function& fn, llvm::FunctionAnalysisManager& fam) {
        OptimalTracingPassBase::run(fn);
        return getTracingSet();
    };

  private:
    static llvm::AnalysisKey Key;
    friend struct llvm::AnalysisInfoMixin<OptimalTracingPass>;
};

} // namespace hip
