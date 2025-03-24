/** \file Optimal_tracing_pass.h
 * \brief Optimal tracing analysis pass. Identify optimal placement of
 * instrumentation nodes
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "llvm/IR/BasicBlock.h"
#include "llvm/Pass.h"

#include <set>

namespace hip {

/** \class OptimalTracingPass
 * \brief Returns a set of edges in the CFG that need to be instrumented
 */
class OptimalTracingPass : llvm::FunctionPass {
  public:
    static char ID;
    using TracingSet =
        std::set<std::tuple<const llvm::BasicBlock*, const llvm::BasicBlock*>>;

    OptimalTracingPass();

    void getAnalysisUsage(llvm::AnalysisUsage& Info) const override;
    bool runOnFunction(llvm::Function&) override;

    const TracingSet& getTracingSet() const { return analysis_result; }

    void print(llvm::raw_ostream& O, llvm::Module const*) const override;

  private:
    TracingSet analysis_result;
    const llvm::BasicBlock* exit_block;

    TracingSet dfs(const llvm::BasicBlock* bb,
                   std::set<const llvm::BasicBlock*> explored_vertices = {});
};

} // namespace hip
