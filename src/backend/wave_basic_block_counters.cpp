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

bool isInstrumentableFunction(MachineFunction& MF) {
    return MF.getName().contains("__hip_instr_dup");
}

bool WaveBasicBlockCountersInstr::runOnMachineFunction(MachineFunction& MF) {
    MF.dump();
    MF.getFunction().dump();

    if (!isInstrumentableFunction(MF)) {
        return false;
    }

    const auto* TII = MF.getSubtarget().getInstrInfo();
    auto& MRI = MF.getRegInfo();

    auto counter_reg = MRI.createVirtualRegister(&AMDGPU::SReg_32RegClass);
    Register counter_address;

    for (auto mbb = MF.begin(); mbb != MF.end(); ++mbb) {
        if (mbb->isEntryBlock()) {

            // Store input parameter address & initialize counter to 0

            for (auto& instr : *mbb) {
                if (!instr.isInlineAsm()) {
                    continue;
                }

                std::cerr << "INSTR : ";
                instr.dump();

                for (auto& op : instr.uses()) {
                    std::cerr << "Use ";
                    op.dump();
                }
                for (auto& op : instr.operands()) {
                    std::cerr << "Def ";
                    op.dump();
                }

                counter_address = instr.getOperand(3).getReg();

                auto builder =
                    BuildMI(*mbb, instr, DebugLoc(),
                            TII->get(AMDGPU::S_MOV_B32), counter_reg);
                builder.addImm(0);
            }
        } else if (mbb->isReturnBlock()) {
            // Store to output
            std::cerr << "End : ";
            mbb->dump();

            auto builder = BuildMI(*mbb, mbb->front(), DebugLoc(),
                                   TII->get(AMDGPU::S_STORE_DWORD_IMM));
            builder.addReg(counter_reg);
            builder.addReg(counter_address);
            builder.addImm(0);
            builder.addImm(0);
        } else {

            // For each basic block increment by 1
            auto builder = BuildMI(*mbb, mbb->begin(), DebugLoc(),
                                   TII->get(AMDGPU::S_ADD_U32), counter_reg);
            builder.addReg(counter_reg);
            builder.addImm(1);
        }
    }

    // Store in

    MF.dump();

    return true;
}
