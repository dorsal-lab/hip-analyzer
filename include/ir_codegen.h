/** \file ir_codegen.cpp
 * \brief LLVM IR instrumentation code generation
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include <unordered_map>

#include "llvm/IR/Module.h"

#include "hip_instrumentation/basic_block.hpp"

namespace hip {

enum class InstrumentationType { Counters, Tracing };

struct InstrumentationContext {
    InstrumentationType instrType;
    llvm::Module& mod;
    llvm::Function& fn;
};

struct InstrumentedBlock {
    unsigned int id;

    // Default counted values
    unsigned int flops;
    unsigned int ld_bytes;
    unsigned int st_bytes;

    std::unordered_map<std::string, unsigned int> extra_counters;
};

/** \struct InstrumentationFunctions
 * \brief Structure of pointers to instrumentation functions in the module
 */
struct InstrumentationFunctions {
    llvm::Function* _hip_store_ctr;
};

/** \fn isBlockInstrumentable
 * \brief Returns true if the block is to be analyzed (and thus instrumented)
 */
bool isBlockInstrumentable(const llvm::BasicBlock& block);

/** \fn getBlockInfo
 * \brief Extracts information from a basic block and returns a report
 */
InstrumentedBlock getBlockInfo(const llvm::BasicBlock& block, unsigned int i);

/** \fn declareInstrumentations
 * \brief Forward-declare instrumentation functions in the module, and returns
 * pointers to them
 */
InstrumentationFunctions declareInstrumentation(llvm::Module& mod);

} // namespace hip
