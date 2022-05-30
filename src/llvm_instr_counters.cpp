/** \file llvm_instruction_counters.cpp
 * \brief LLVM Instruction counters for basic block static analysis
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "llvm_instr_counters.h"

namespace hip {

unsigned int FlopCounter::operator()(llvm::BasicBlock& bb) {
    using namespace llvm;

    for (const auto& instr : bb) {
        switch (instr.getOpcode()) {
        case Instruction::FAdd:
        case Instruction::FSub:
        case Instruction::FMul:
        case Instruction::FDiv:
        case Instruction::FRem:
        case Instruction::FPToUI:
        case Instruction::FPToSI:
        case Instruction::UIToFP:
        case Instruction::SIToFP:
        case Instruction::FPTrunc:
        case Instruction::FPExt:
        case Instruction::FCmp:
            ++count;
        default:;
        }
    }

    return getCount();
}

} // namespace hip
