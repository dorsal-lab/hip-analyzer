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

#include <fstream>
#include <iostream>

using namespace llvm;

#define DEBUG_TYPE "extract-bb-id"

char ExtractBBId::ID = 0;

INITIALIZE_PASS(ExtractBBId, DEBUG_TYPE, "Extract Basic Block ID from CFG",
                false, false);

bool ExtractBBId::runOnMachineFunction(MachineFunction& MF) {
    if (!MF.getName().contains("__hip_instr_tracing")) {
        return true;
    }

    // MF.dump();

    std::string filename("/tmp/cfgs/");
    filename += MF.getName();
    filename += ".json";

    std::ofstream out_cfg(filename);

    out_cfg << "{";
    int id = 0;
    for (auto& block : MF) {
        auto name = block.getName().str();

        for (auto& instr : block) {
            if (!instr.isInlineAsm()) {
                continue;
            }

            if (instr.getNumOperands() > 6) {
                auto bb_id = instr.getOperand(5).getImm();
                std::cerr << name << " == " << bb_id << '\n';

                if (id > 0) {
                    out_cfg << ',';
                }
                out_cfg << "\"" << name << "\":" << bb_id;

                ++id;
            }
        }
    }

    out_cfg << "}\n";

    return true;
}
