/** \file preload_pass_insertion.cpp
 * \brief Intercept hooks to add instrumentation passes
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "backend/export_cfg.h"

#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Pass.h"

#include <iostream>

// ---- hooks definition ----- //

extern "C" {

void amdgcn_hooks_optimized_reg_alloc(llvm::TargetPassConfig* pass_config) {
    llvm::PassRegistry* PR = llvm::PassRegistry::getPassRegistry();

    initializeExportCFGPass(*PR);

    pass_config->insertPass(&llvm::SILowerControlFlowID, &ExportCFG::ID);
}
}
