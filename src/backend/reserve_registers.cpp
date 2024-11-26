/** \file intercept.cpp
 * \brief Test for LLVM Backend
 *
 * \author Sebastien Darche <sebastien.darche@polymtl.ca>
 */

#include "AMDGPU.h"
#include "AMDGPUResourceUsageAnalysis.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIFrameLowering.h"
#include "SIRegisterInfo.h"

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/Target/TargetMachine.h"

#include <charconv>
#include <fstream>
#include <iostream>
#include <sstream>

#include <dlfcn.h>

/*

gcc -std=c++17 \
    -fno-rtti \
    -g -ggdb -fPIC -rdynamic -shared \
    ../src/intercept.cpp -o intercept.so \
    -L/scratch/sdarche/llvm-rocm-6.2.0-build/lib \
    -Wl,-rpath-link=/scratch/sdarche/llvm-rocm-6.2.0-build/lib \
    -I/scratch/sdarche/llvm-rocm-6.2.0-build/lib/Target/AMDGPU \
    -I/scratch/sdarche/llvm-rocm-6.2.0-build/include \
    -I/home/sdarche/rocm-llvm/llvm/include \
    -I/home/sdarche/rocm-llvm/llvm/lib/Target/AMDGPU \
    -lLLVMAMDGPUDesc \
    -lLLVMAMDGPUInfo \
    -lLLVMAMDGPUUtils \
    -lLLVMAMDGPUCodeGen \
    -lLLVMAnalysis \
    -lLLVMAsmPrinter \
    -lLLVMBinaryFormat \
    -lLLVMCodeGen \
    -lLLVMCodeGenTypes \
    -lLLVMCore \
    -lLLVMGlobalISel \
    -lLLVMHipStdPar \
    -lLLVMMC \
    -lLLVMMIRParser \
    -lLLVMPasses \
    -lLLVMSelectionDAG \
    -lLLVMSupport \
    -lLLVMTarget \
    -lLLVMTargetParser \
    -lLLVMTransformUtils \
    -lLLVMVectorize

*/

namespace llvm {

namespace AMDGPU {

llvm::SmallVector<unsigned int> HipAnalyzerRegs = []() {
    const char* env = std::getenv("HIPCC_RESERVED_REGS");

    llvm::SmallVector<unsigned int> vec;
    if (env == nullptr) {
        return vec;
    }

    std::string token;
    std::stringstream ss;
    ss << env;

    std::ofstream out("/tmp/hip_analyzer");

    int count = 0;

    while (std::getline(ss, token, ',')) {
        unsigned int val;
        if (std::from_chars(token.data(), token.data() + token.length(), val)
                .ec != std::errc()) {
            out << "HIPCC_RESERVED_REGS : could not parse " << token << '\n';
        }

        vec.push_back(AMDGPU::SGPR0 + val);
        out << "Reserved " << val << " (" << (AMDGPU::SGPR0 + val) << ")\n";
        ++count;
    }

    out << "Total reserved SGPRs : " << count << '\n';
    return vec;
}();

} // namespace AMDGPU

llvm::BitVector
SIRegisterInfo::getReservedRegs(const MachineFunction& MF) const {
    // llvm::BitVector (*original_handle)(void*, const MachineFunction&);

    auto* original_handle = reinterpret_cast<
        llvm::BitVector (*)(const void*, const MachineFunction&)>(dlsym(
        RTLD_NEXT,
        "_ZNK4llvm14SIRegisterInfo15getReservedRegsERKNS_15MachineFunctionE"));

    std::cout << "Found handle " << (void*)original_handle << '\n';

    auto regs = original_handle(this, MF);

    for (auto hip_analyzer_reg : AMDGPU::HipAnalyzerRegs) {
        regs.set(hip_analyzer_reg);
        markSuperRegs(regs, hip_analyzer_reg);
    }

    return regs;
}

// Shift down registers reserved for the scratch RSRC.
void SIFrameLowering::emitEntryFunctionPrologue(MachineFunction& MF,
                                                MachineBasicBlock& MBB) const {
    auto* original_handle = reinterpret_cast<void (*)(
        const void*, MachineFunction&, MachineBasicBlock&)>(
        dlsym(RTLD_NEXT, "_ZNK4llvm15SIFrameLowering38getEntryFunctionReservedS"
                         "cratchRsrcRegERNS_15MachineFunctionE"));

    original_handle(this, MF, MBB);

    // hip-analyzer reserved regs

    for (llvm::MachineBasicBlock& OtherBB : MF) {
        if (&OtherBB != &MBB) {
            for (auto hip_analyzer_reg : AMDGPU::HipAnalyzerRegs) {
                OtherBB.addLiveIn(hip_analyzer_reg);
            }
        }
    }
}

AMDGPUResourceUsageAnalysis::SIFunctionResourceInfo
AMDGPUResourceUsageAnalysis::analyzeResourceUsage(
    const MachineFunction& MF, const TargetMachine& TM) const {
    auto* original_handle = reinterpret_cast<
        AMDGPUResourceUsageAnalysis::SIFunctionResourceInfo (*)(
            const void*, const MachineFunction&, const TargetMachine&)>(
        dlsym(RTLD_NEXT,
              "_ZNK4llvm27AMDGPUResourceUsageAnalysis20analyzeResourceUsageERKN"
              "S_15MachineFunctionERKNS_13TargetMachineE"));

    auto Info = original_handle(this, MF, TM);

    const GCNSubtarget& ST = MF.getSubtarget<GCNSubtarget>();
    const SIInstrInfo* TII = ST.getInstrInfo();
    const SIRegisterInfo& TRI = TII->getRegisterInfo();

    auto HighestSGPRReg = Info.NumExplicitSGPR - 1;

    if (AMDGPU::HipAnalyzerRegs.size() != 0) {
        for (auto reg : AMDGPU::HipAnalyzerRegs) {
            auto hwreg = TRI.getHWRegIndex(reg);
            if (HighestSGPRReg < hwreg) {
                HighestSGPRReg = hwreg;
            }
        }
    }

    return Info;
}

} // namespace llvm
