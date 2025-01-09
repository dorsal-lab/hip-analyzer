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

#include "backend/wave_basic_block_counters.h"

#include <iostream>

using namespace llvm;

#define DEBUG_TYPE "print-function-test"

char PrintFunction::ID = 0;

INITIALIZE_PASS(PrintFunction, DEBUG_TYPE, "Print function test pass", false,
                false);

bool PrintFunction::runOnMachineFunction(MachineFunction& MF) {
    MF.dump();

    return false;
}
