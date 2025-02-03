/** \file backend_utils.h
 * \brief Useful backend manipulation functions
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include "AMDGPU.h"
#include "AMDGPUCallLowering.h"
#include "AMDGPUInstructionSelector.h"
#include "AMDGPULegalizerInfo.h"
#include "AMDGPURegisterBankInfo.h"
#include "AMDGPUSubtarget.h"
#include "AMDGPUTargetMachine.h"

/** \fn getSubReg
 * \brief
 */
llvm::Register getSubReg(llvm::Register& reg,
                         const llvm::TargetRegisterClass& SubRc,
                         unsigned subidx, llvm::MachineFunction& MF,
                         llvm::MachineInstr* insertion_point);

/** \fn getFlatBlockId
 * \brief
 */
llvm::Register getFlatBlockId(llvm::MachineFunction& MF,
                              llvm::MachineInstr* insertion_point);
