/** \file backend_utils.cpp
 * \brief Useful backend manipulation functions
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "SIMachineFunctionInfo.h"

#include <iostream>

#include "backend/backend_utils.h"

using namespace llvm;

Register getSubReg(Register& reg, const TargetRegisterClass& SubRc,
                   unsigned subidx, MachineFunction& MF,
                   MachineInstr* insertion_point) {
    const auto* TII = MF.getSubtarget().getInstrInfo();
    Register dest = MF.getRegInfo().createVirtualRegister(&SubRc);

    BuildMI(*insertion_point->getParent(), insertion_point, DebugLoc(),
            TII->get(AMDGPU::COPY), dest)
        .addReg(reg, 0, subidx);

    return dest;
}

Register getFlatBlockId(MachineFunction& MF, MachineInstr* insertion_point) {
    SIMachineFunctionInfo* Info = MF.getInfo<SIMachineFunctionInfo>();
    const auto* TII = MF.getSubtarget().getInstrInfo();
    const auto& ST = static_cast<const GCNSubtarget&>(MF.getSubtarget());

    uint64_t implicit_arg_offset =
        ST.getTargetLowering()->getImplicitParameterOffset(
            MF, AMDGPUTargetLowering::FIRST_IMPLICIT);

    Register kernarg_ptr =
        std::get<0>(
            Info->getPreloadedValue(AMDGPUFunctionArgInfo::KERNARG_SEGMENT_PTR))
            ->getRegister();

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
        const AMDGPULegalizerInfo* LI = static_cast<const AMDGPULegalizerInfo*>(
            MF.getSubtarget().getLegalizerInfo());

        std::cerr << "83" << '\n';

        Register grid_dim_xy = MF.getRegInfo().createVirtualRegister(
            &AMDGPU::SReg_64_XEXECRegClass);

        auto* MMO = MF.getMachineMemOperand(
            MachinePointerInfo(AMDGPUAS::CONSTANT_ADDRESS),
            MachineMemOperand::MOLoad | MachineMemOperand::MOInvariant |
                MachineMemOperand::MODereferenceable,
            8, Align(4));

        unsigned grid_dim_offset = AMDGPU::convertSMRDOffsetUnits(
            MF.getSubtarget(),
            implicit_arg_offset + 0); // offset in implicit arg

        unsigned block_dim_offset = AMDGPU::convertSMRDOffsetUnits(
            MF.getSubtarget(),
            implicit_arg_offset + 12); // offset in implicit arg

        BuildMI(MF.front(), insertion_point, DebugLoc(),
                TII->get(AMDGPU::S_LOAD_DWORDX2_IMM), grid_dim_xy)
            .addReg(kernarg_ptr)
            .addImm(grid_dim_offset)
            .addImm(0)
            .addMemOperand(MMO);

        // Register block_dim_xy =
        //     MF.getRegInfo().createVirtualRegister(&AMDGPU::SReg_32RegClass);
        // BuildMI(MF.front(), insertion_point, DebugLoc(),
        //         TII->get(AMDGPU::S_LOAD_DWORDX2_IMM), block_dim_xy)
        //     .addReg(kernarg_ptr)
        //     .addImm(block_dim_offset)
        //     .addImm(0)
        //     .addMemOperand(MMO);

        std::cerr << "112 " << grid_dim_xy << '\n';

        Register grid_dim_x = getSubReg(grid_dim_xy, AMDGPU::SReg_32RegClass,
                                        AMDGPU::sub0, MF, insertion_point);

        Register grid_dim_y = getSubReg(grid_dim_xy, AMDGPU::SReg_32RegClass,
                                        AMDGPU::sub1, MF, insertion_point);

        // Register block_dim_x =
        //     MF.getRegInfo().createVirtualRegister(&AMDGPU::SReg_32RegClass);

        // BuildMI(MF.front(), insertion_point, DebugLoc(),
        // TII->get(AMDGPU::COPY),
        //         block_dim_x)
        //     .addReg(block_dim_xy);

        // BuildMI(MF.front(), insertion_point, DebugLoc(),
        //         TII->get(AMDGPU::S_AND_B32), block_dim_x)
        //     .addReg(block_dim_x)
        //     .addImm(0xFFFF);

        std::cerr << "142" << '\n';

        // Now, we have loaded everything we need

        Register sgpr_y = Info->getWorkGroupIDSGPR(1);

        std::cerr << "148" << '\n';

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

        std::cerr << "162" << '\n';

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
