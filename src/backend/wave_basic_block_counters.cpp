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

Register getSubReg(Register& reg, const TargetRegisterClass& SubRc,
                   unsigned subidx, MachineFunction& MF,
                   MachineInstr* insertion_point) {
    const auto* TII = MF.getSubtarget().getInstrInfo();
    Register dest = MF.getRegInfo().createVirtualRegister(&SubRc);

    unsigned composedId = reinterpret_cast<SIRegisterInfo&>(MF.getRegInfo())
                              .composeSubRegIndices(reg, subidx);

    BuildMI(*insertion_point->getParent(), insertion_point, DebugLoc(),
            TII->get(AMDGPU::COPY), dest)
        .addReg(reg, 0, composedId);
    return dest;
}

Register getFlatBlockId(MachineFunction& MF, MachineInstr* insertion_point) {
    SIMachineFunctionInfo* Info = MF.getInfo<SIMachineFunctionInfo>();
    const auto* TII = MF.getSubtarget().getInstrInfo();

    auto dim = 0;
    if (Info->hasWorkGroupIDX()) {
        ++dim;
    }
    if (Info->hasWorkGroupIDY()) {
        ++dim;
    }
    if (Info->hasWorkGroupIDZ()) {
        ++dim;
    }

    std::cerr << "dim : " << dim << '\n';

    Register sgpr_x = Info->getWorkGroupIDSGPR(0);

    Register block_id =
        MF.getRegInfo().createVirtualRegister(&AMDGPU::SReg_32RegClass);

    BuildMI(MF.front(), insertion_point, DebugLoc(), TII->get(AMDGPU::COPY),
            block_id)
        .addReg(sgpr_x);

    if (dim > 1) {
        // Get gridDim.x, blockDim.x from implicit arg pointer
        const auto* implicitArgArgInfo = std::get<0>(
            Info->getPreloadedValue(AMDGPUFunctionArgInfo::IMPLICIT_ARG_PTR));

        std::cerr << implicitArgArgInfo << '\n';

        Register implicit_arg = implicitArgArgInfo->getRegister();

        std::cerr << "84" << '\n';

        Register grid_dim_xy =
            MF.getRegInfo().createVirtualRegister(&AMDGPU::SReg_64RegClass);

        Register block_dim_xy =
            MF.getRegInfo().createVirtualRegister(&AMDGPU::SReg_32RegClass);

        auto* MMO = MF.getMachineMemOperand(
            MachinePointerInfo(AMDGPUAS::CONSTANT_ADDRESS),
            MachineMemOperand::MOLoad | MachineMemOperand::MOInvariant |
                MachineMemOperand::MODereferenceable,
            8, Align(4));

        unsigned grid_dim_offset = AMDGPU::convertSMRDOffsetUnits(
            MF.getSubtarget(), 0); // offset in implicit arg

        unsigned block_dim_offset = AMDGPU::convertSMRDOffsetUnits(
            MF.getSubtarget(), 12); // offset in implicit arg

        BuildMI(MF.front(), insertion_point, DebugLoc(),
                TII->get(AMDGPU::S_LOAD_DWORDX2_IMM), grid_dim_xy)
            .addReg(implicit_arg)
            .addImm(grid_dim_offset)
            .addImm(0)
            .addMemOperand(MMO);

        std::cerr << "112" << '\n';

        Register grid_dim_x = getSubReg(grid_dim_xy, AMDGPU::SReg_32RegClass,
                                        AMDGPU::sub0, MF, insertion_point);

        Register grid_dim_y = getSubReg(grid_dim_xy, AMDGPU::SReg_32RegClass,
                                        AMDGPU::sub1, MF, insertion_point);

        Register block_dim_x =
            MF.getRegInfo().createVirtualRegister(&AMDGPU::SReg_32RegClass);

        BuildMI(MF.front(), insertion_point, DebugLoc(), TII->get(AMDGPU::COPY),
                block_dim_x)
            .addReg(block_dim_xy);

        BuildMI(MF.front(), insertion_point, DebugLoc(),
                TII->get(AMDGPU::S_AND_B32), block_dim_x)
            .addReg(block_dim_x)
            .addImm(0xFFFF);

        // Now, we have loaded everything we need

        Register sgpr_y = Info->getWorkGroupIDSGPR(1);

        // Mul sgpr_y * sgpr_dx
        Register sgpr_y_dx =
            MF.getRegInfo().createVirtualRegister(&AMDGPU::SReg_32RegClass);
        BuildMI(MF.front(), insertion_point, DebugLoc(),
                TII->get(AMDGPU::S_MUL_I32), sgpr_y_dx)
            .addReg(sgpr_y)
            .addReg(grid_dim_x);

        // add to block_id
        BuildMI(MF.front(), insertion_point, DebugLoc(),
                TII->get(AMDGPU::S_ADD_U32), block_id)
            .addReg(block_id)
            .addReg(sgpr_y_dx);

        if (dim > 2) {
            Register sgpr_z = Info->getWorkGroupIDSGPR(2);
            Register sgpr_temp =
                MF.getRegInfo().createVirtualRegister(&AMDGPU::SReg_32RegClass);

            // Compute gridDim.x * gridDim.y
            BuildMI(MF.front(), insertion_point, DebugLoc(),
                    TII->get(AMDGPU::S_MUL_I32), sgpr_temp)
                .addReg(grid_dim_x)
                .addReg(grid_dim_y);

            BuildMI(MF.front(), insertion_point, DebugLoc(),
                    TII->get(AMDGPU::S_MUL_I32), sgpr_temp)
                .addReg(sgpr_temp)
                .addReg(sgpr_z);
            BuildMI(MF.front(), insertion_point, DebugLoc(),
                    TII->get(AMDGPU::S_ADD_U32), block_id)
                .addReg(block_id)
                .addReg(sgpr_temp);
        }
    }

    return block_id;
}

bool WaveBasicBlockCountersInstr::runOnMachineFunction(MachineFunction& MF) {
    MF.dump();
    MF.getFunction().dump();

    if (!isInstrumentableFunction(MF)) {
        return false;
    }

    const auto* TII = MF.getSubtarget().getInstrInfo();
    auto& MRI = MF.getRegInfo();

    getFlatBlockId(MF, &*MF.front().begin());

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
