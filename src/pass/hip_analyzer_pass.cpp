/** \file hip_analyzer_pass
 * \brief Instrumentation pass
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

namespace {

struct HipAnalyzerPass : public llvm::ModulePass {
    static char ID;

    HipAnalyzerPass() : llvm::ModulePass(ID) {}

    virtual bool runOnModule(llvm::Module& mod) {
        llvm::errs() << "Module " << mod.getName() << '\n';
        mod.dump();
        return false;
    }
};

char HipAnalyzerPass::ID = 0;

static void registerHipAnalyzerPass(const llvm::PassManagerBuilder&,
                                    llvm::legacy::PassManagerBase& PM) {
    PM.add(new HipAnalyzerPass());
}

static llvm::RegisterPass<HipAnalyzerPass>
    Pass("hip-analyzer", "Hip-Analyzer instrumentation", false, false);

} // namespace

static llvm::RegisterStandardPasses
    registerPass(llvm::PassManagerBuilder::EP_EarlyAsPossible,
                 registerHipAnalyzerPass);
