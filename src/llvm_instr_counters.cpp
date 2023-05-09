/** \file llvm_instruction_counters.cpp
 * \brief LLVM Instruction counters for basic block static analysis
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "llvm_instr_counters.h"

#include "llvm/IR/Instructions.h"

namespace hip {

unsigned int FlopCounter::operator()(const llvm::BasicBlock& bb) {
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

constexpr auto bits_per_byte = 8u;

bool isTypeCounted(const llvm::Type* type, MemType::MemType type_filter) {
    if (type_filter == MemType::All) {
        return true;
    }

    if (type_filter & MemType::MemType::Floating) {
        return type->isFloatingPointTy();
    } else if (type_filter & MemType::MemType::Integer) {
        return type->isIntegerTy();
    } else {
        return false;
    }
}

unsigned int StoreCounter::count(const llvm::BasicBlock& bb,
                                 MemType::MemType type_filter) {
    for (const auto& instr : bb) {
        if (const auto* store_inst = llvm::dyn_cast<llvm::StoreInst>(&instr)) {
            // store_inst->print(llvm::errs());

            const auto* type = store_inst->getValueOperand()->getType();

            if (isTypeCounted(type, type_filter)) {
                counted += type->getPrimitiveSizeInBits() / bits_per_byte;
            }
        }
    }

    return getCount();
}

unsigned int LoadCounter::count(const llvm::BasicBlock& bb,
                                MemType::MemType type_filter) {
    // TODO : handle possible compound types
    for (const auto& instr : bb) {
        if (const auto* load_inst = llvm::dyn_cast<llvm::LoadInst>(&instr)) {
            // load_inst->print(llvm::errs());

            const auto* type = load_inst->getType();

            if (isTypeCounted(type, type_filter)) {
                counted += type->getPrimitiveSizeInBits() / bits_per_byte;
            }
        }
    }

    return getCount();
}

} // namespace hip
