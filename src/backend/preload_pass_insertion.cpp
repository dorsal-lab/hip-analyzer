/** \file preload_pass_insertion.cpp
 * \brief Intercept hooks to add instrumentation passes
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "backend/wave_basic_block_counters.h"
#include "hip_analyzer_pass.h"

#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Pass.h"

#include <iostream>

class PassPrinter : public llvm::PassRegistrationListener {
  public:
    void passEnumerate(const llvm::PassInfo* pass_info) override {
        std::cerr << pass_info->getPassName().str() << '\n';
    }
};

// ---- hooks definition ----- //

extern "C" {

void amdgcn_hooks_optimized_reg_alloc(llvm::TargetPassConfig* pass_config) {
    PassPrinter printer;

    llvm::PassRegistry* PR = llvm::PassRegistry::getPassRegistry();

    PR->enumerateWith(&printer);

    initializePrintFunctionPass(*PR);
    initializeWaveBasicBlockCountersInstrPass(*PR);
    initializeDuplicateKernelsPass(*PR);

    pass_config->insertPass(&llvm::SILowerControlFlowID,
                            &WaveBasicBlockCountersInstr::ID);
    pass_config->insertPass(&llvm::SILowerControlFlowID, &PrintFunction::ID);
}

llvm::SmallVector<llvm::Pass*> amdgcn_hooks_additional_ir_passes() {
    return {new DuplicateKernels()};
}
}
