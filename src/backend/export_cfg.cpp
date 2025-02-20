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

#include "backend/backend_utils.h"
#include "backend/export_cfg.h"

#include <iostream>

using namespace llvm;

#define DEBUG_TYPE "export-cfg"

char ExportCFG::ID = 0;

INITIALIZE_PASS(ExportCFG, DEBUG_TYPE, "Export CFG Pass", false, false);

bool ExportCFG::runOnMachineFunction(MachineFunction& MF) {
    MF.dump();

    return true;
}
