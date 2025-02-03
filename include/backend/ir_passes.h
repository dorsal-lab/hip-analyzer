/** \file ir_passes.h
 * \brief Preliminary IR passes
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include "llvm/Pass.h"

namespace llvm {
void initializeDuplicateKernelsPass(PassRegistry&);
}

class DuplicateKernels : public llvm::ModulePass {
  public:
    static char ID;
    DuplicateKernels() : ModulePass(ID) {}

    bool runOnModule(llvm::Module& Mod) override;

    llvm::StringRef getPassName() const override {
        return "Duplicate AMDGPU kernels for intrumentation";
    }
};
