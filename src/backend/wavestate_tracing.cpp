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
#include "backend/wavestate_tracing.h"

#include <iostream>

using namespace llvm;

#define DEBUG_TYPE "wavestate-tracing"

char WavestateTracingInstr::ID = 0;

INITIALIZE_PASS(WavestateTracingInstr, DEBUG_TYPE,
                "Wavestate tracing instrumentation pass", false, false);

bool WavestateTracingInstr::runOnMachineFunction(MachineFunction& MF) {
    MF.dump();

    if (!isInstrumentableFunction(MF)) {
        return false;
    }

    unsigned int block_id = 0;

    for (auto& mbb : MF) {
        if (mbb.isEntryBlock()) {
            initTracePointer(MF, &mbb.front());
        }

        // Instrument all basic blocks
        insertTracepoint(MF, &mbb.front());

        ++block_id;
    }

    return true;
}

Register
WavestateTracingInstr::initTracePointer(MachineFunction& MF,
                                        MachineInstr* insertion_point) {}

void WavestateTracingInstr::insertTracepoint(MachineFunction& MF,
                                             MachineInstr* insertion_point,
                                             uint32_t block_id) {
    const auto* TII = MF.getSubtarget().getInstrInfo();
    auto& MRI = MF.getRegInfo();

    // Increase ptr by event size
    increment64Register(MF, insertion_point, trace_pointer, getEventSize());

    // Prepare payload
    Register payload_lo = MRI.createVirtualRegister(&AMDGPU::SReg_128RegClass);
    Register payload_hi = MRI.createVirtualRegister(&AMDGPU::SReg_64RegClass);

    Register sub_timestamp = getSubReg(payload_lo, AMDGPU::SReg_64RegClass,
                                       AMDGPU::sub0_sub1, insertion_point);
    Register sub_exec = getSubReg(payload_lo, AMDGPU::SReg_64RegClass,
                                  AMDGPU::sub2_sub3, insertion_point);
    Register sub_hwid = getSubReg(payload_hi, AMDGPU::SReg_32RegClass,
                                  AMDGPU::sub0, insertion_point);
    Register sub_producer = getSubReg(payload_hi, AMDGPU::SReg_32RegClass,
                                      AMDGPU::sub0, insertion_point);

    BuildMI(insertion_point->getParent(), insertion_point, DebugLoc(),
            TII->get(AMDGPU::S_MEM_REALTIME), sub_timestamp);
    BuildMI(insertion_point->getParent(), insertion_point, DebugLoc(),
            TII->get(AMDGPU::S_GETREG))

    // Store
}
