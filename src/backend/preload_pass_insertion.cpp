/** \file preload_pass_insertion.cpp
 * \brief Intercept hooks to add instrumentation passes
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "backend/wave_basic_block_counters.h"

#include "llvm/CodeGen/TargetPassConfig.h"

#include <iostream>

// ---- hooks definition ----- //

extern "C" {

void amdgcn_hooks_optimized_reg_alloc(llvm::TargetPassConfig* pass_config) {
    llvm::PassRegistry* PR = llvm::PassRegistry::getPassRegistry();
    initializePrintFunctionPass(*PR);

    std::cerr << static_cast<int>(llvm::SILowerControlFlowID) << " ; "
              << static_cast<int>(PrintFunction::ID) << "\n";

    pass_config->insertPass(&llvm::SILowerControlFlowID, &PrintFunction::ID);
}
}
