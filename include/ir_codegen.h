/** \file ir_codegen.cpp
 * \brief LLVM IR instrumentation code generation
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include <unordered_map>

#include "llvm/ADT/StringRef.h"
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

    // ----- C instrumentation API ----- //

    // hipInstrumenter
    llvm::Function *hipNewInstrumenter, *hipInstrumenterToDevice,
        *hipInstrumenterFromDevice, *hipInstrumenterRecord,
        *freeHipInstrumenter;

    // hipStateRecoverer
    llvm::Function *hipNewStateRecoverer, *hipStateRecovererRegisterPointer,
        *hipStateRecovererRollback, *freeHipStateRecoverer;
};

// ----- IR Utils ----- //

inline bool contains(const std::string& str, std::string_view substr) {
    return (str.find(substr) != std::string::npos);
};

llvm::Value* getIndex(uint64_t idx, llvm::LLVMContext& context);

int64_t valueToInt(llvm::Value* v);

llvm::BasicBlock::iterator
findInstruction(llvm::Function& f,
                std::function<bool(const llvm::Instruction*)> predicate);

llvm::BasicBlock::iterator
findInstruction(llvm::BasicBlock& bb,
                std::function<bool(const llvm::Instruction*)> predicate);

/** \fn recursiveGetUsePredicate
 * \brief Recursively search for a value that matches the predicate, propagates
 * the search through uses' sub-uses
 */
llvm::Value*
recursiveGetUsePredicate(llvm::Value* v,
                         std::function<bool(const llvm::Value*)> predicate);

template <typename T>
llvm::BasicBlock::iterator firstInstructionOf(llvm::Function& f) {
    return findInstruction(
        f, [](const llvm::Instruction* i) { return isa<T>(i); });
}

void setInsertPointPastAllocas(llvm::IRBuilderBase& builder, llvm::Function& f);

/** \fn firstCallToFunction
 * \brief Returns the first call to the given symbol
 */
llvm::CallInst* firstCallToFunction(llvm::Function& f,
                                    const std::string& function);

/** \fn hasFunctionCall
 * \brief Returns whether the function contains a call to the given symbol
 */
bool hasFunctionCall(llvm::Function& f, const std::string& function);

llvm::BasicBlock::iterator getFirstNonPHIOrDbgOrAlloca(llvm::BasicBlock& bb);

/** \fn isBlockInstrumentable
 * \brief Returns true if the block is to be analyzed (and thus
 * instrumented)
 */
bool isBlockInstrumentable(const llvm::BasicBlock& block);

/** \fn getBlockInfo
 * \brief Extracts information from a basic block and returns a report
 */
InstrumentedBlock getBlockInfo(const llvm::BasicBlock& block, unsigned int i);

// ----- IR Modifiers ----- //

/** \fn declareInstrumentations
 * \brief Forward-declare instrumentation functions in the module, and returns
 * pointers to them
 */
InstrumentationFunctions declareInstrumentation(llvm::Module& mod);

/** \fn cloneWithName
 * \brief Clones a function f with a different name, and eventually additional
 * args
 */
llvm::Function& cloneWithName(llvm::Function& f, const std::string& name,
                              llvm::ArrayRef<llvm::Type*> extra_args = {});

/** \brief Suffix to distinguish already cloned function, placeholder for a real
 * attribute
 */
constexpr auto cloned_suffix = "__hip_instr_";

inline std::string getClonedName(const std::string& f,
                                 const std::string& prefix) {
    return llvm::Twine(cloned_suffix).concat(prefix).concat(f).str();
}

inline std::string getClonedName(const llvm::Function& f,
                                 const std::string& prefix) {
    return getClonedName(f.getName().str(), prefix);
}

llvm::Function& cloneWithPrefix(llvm::Function& f, const std::string& prefix,
                                llvm::ArrayRef<llvm::Type*> extra_args);

/** \fn pushAdditionalArguments
 * \brief Add additional arguments to a kernel stub
 *
 * \param f Kernel device stub
 * \param kernel_args Additional values (expected to be arguments of the
 * function)
 */
void pushAdditionalArguments(llvm::Function& f,
                             llvm::ArrayRef<llvm::Value*> kernel_args);

} // namespace hip
