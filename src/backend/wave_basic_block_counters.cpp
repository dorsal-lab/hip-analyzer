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

#define DEBUG_TYPE "wave-basic-block-counter"

char WaveBasicBlockCountersInstr::ID = 0;

INITIALIZE_PASS(WaveBasicBlockCountersInstr, DEBUG_TYPE,
                "Wave basic block counters pass", false, false);

bool WaveBasicBlockCountersInstr::runOnMachineFunction(MachineFunction& MF) {
    MF.dump();
    return false;
}
