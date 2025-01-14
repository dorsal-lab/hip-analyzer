#include "AMDGPU.h"
#include "AMDGPUCallLowering.h"
#include "AMDGPUInstructionSelector.h"
#include "AMDGPULegalizerInfo.h"
#include "AMDGPURegisterBankInfo.h"
#include "AMDGPUSubtarget.h"
#include "AMDGPUTargetMachine.h"
#include "GCNSubtarget.h"
#include "R600Subtarget.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/GlobalISel/InlineAsmLowering.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/TargetFrameLowering.h"

#include "SIInstrInfo.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

namespace llvm {
void initializePrintFunctionPass(PassRegistry&);
void initializeWaveBasicBlockCountersInstrPass(PassRegistry&);
void initializeDuplicateKernelsPass(PassRegistry&);
} // namespace llvm

class PrintFunction : public llvm::MachineFunctionPass {
  public:
    static char ID;

    PrintFunction() : MachineFunctionPass(ID) {}

    bool runOnMachineFunction(llvm::MachineFunction& MF) override;

    llvm::StringRef getPassName() const override {
        return "Print function test pass";
    }
};

class WaveBasicBlockCountersInstr : public llvm::MachineFunctionPass {
  public:
    static char ID;
    WaveBasicBlockCountersInstr() : MachineFunctionPass(ID) {}

    bool runOnMachineFunction(llvm::MachineFunction& MF) override;

    llvm::StringRef getPassName() const override {
        return "Wavefront basic block counters instrumentation pas";
    }
};

class DuplicateKernels : public llvm::ModulePass {
  public:
    static char ID;
    DuplicateKernels() : ModulePass(ID) {}

    bool runOnModule(llvm::Module& Mod) override;

    llvm::StringRef getPassName() const override {
        return "Duplicate AMDGPU kernels for intrumentation";
    }
};
