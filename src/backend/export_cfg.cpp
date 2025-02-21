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

#define DEBUG_TYPE "export-cfg"

char ExportCFG::ID = 0;

INITIALIZE_PASS(ExportCFG, DEBUG_TYPE, "Export CFG Pass", false, false);

bool ExportCFG::runOnMachineFunction(MachineFunction& MF) {
    MF.dump();

    std::string filename("/tmp/kernels/");
    filename += MF.getName();
    filename += ".json";

    std::ofstream out_cfg(filename);

    out_cfg << "{";
    for (auto& block : MF) {
        auto name = block.getName().str();

        if (block.isReturnBlock()) {
            continue;
        }

        out_cfg << "\"" << name << "\":[";

        int j = 0;
        for (auto& succ : block.successors()) {
            std::string succ_name("");

            if (j > 0) {
                succ_name += ",";
            }

            if (succ->isReturnBlock()) {
                succ_name += "\"exit\"";
            } else {
                succ_name += "\"";
                succ_name += succ->getName();
                succ_name += "\"";
            }
            out_cfg << succ_name;
            ++j;
        }
        out_cfg << "],";
    }

    out_cfg << "\"exit\":[]}";

    return true;
}
