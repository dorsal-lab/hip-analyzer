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
    llvm::BasicBlock& bb;
    unsigned int id;

    // Default counted values
    unsigned int flops;
    unsigned int ld_bytes;
    unsigned int st_bytes;

    std::unordered_map<std::string, unsigned int> extra_counters;
};

} // namespace hip
