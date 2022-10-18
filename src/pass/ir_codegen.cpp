/** \file ir_codegen.cpp
 * \brief LLVM IR instrumentation code generation
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "llvm/IR/Module.h"

#include "ir_codegen.h"
#include "llvm_instr_counters.h"

namespace hip {

bool isBlockInstrumentable(const llvm::BasicBlock& block) { return true; }

void insertInstrumentationLocals(InstrumentationContext& context) {}

void insertCounter(llvm::BasicBlock& bb) {}

InstrumentedBlock getBlockInfo(const llvm::BasicBlock& bb, unsigned int i) {
    FlopCounter flop_counter;
    auto flops = flop_counter(bb);

    llvm::errs() << '\n' << bb.getName() << " (" << i << ")\n";
    llvm::errs() << "Found " << flops << " flops\n";

    hip::StoreCounter store_counter;
    auto f_st = store_counter.count(bb, MemType::MemType::Floating);
    llvm::errs() << "Found " << f_st << " stores \n";

    hip::LoadCounter load_counter;
    auto f_ld = load_counter.count(bb, MemType::MemType::Floating);
    llvm::errs() << "Found " << f_ld << " loads \n";

    return {i, flops, f_ld, f_st, {}};
}

} // namespace hip
