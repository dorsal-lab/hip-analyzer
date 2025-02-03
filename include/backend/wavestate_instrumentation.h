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
void initializeWavestateTracingInstrPass(PassRegistry&);
} // namespace llvm

class WavestateTracingInstr : public llvm::MachineFunctionPass {
  public:
    static char ID;
    WavestateTracingInstr() : MachineFunctionPass(ID) {}

    bool runOnMachineFunction(llvm::MachineFunction& MF) override;

    llvm::StringRef getPassName() const override {
        return "Wavestate event tracing";
    }
};
