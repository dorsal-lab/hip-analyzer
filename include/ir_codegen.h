/** \file ir_codegen.cpp
 * \brief LLVM IR instrumentation code generation
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include <unordered_map>

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"

#include "hip_instrumentation/basic_block.hpp"

namespace hip {

/** \brief Different address spaces supported by the backend
 */
namespace addrspaces {

constexpr unsigned int global = 1u;   // Global memory
constexpr unsigned int region = 2u;   // Device-specific
constexpr unsigned int local = 3u;    // LDS
constexpr unsigned int constant = 4u; // Global, but constant
constexpr unsigned int thread = 5u;   // Thread-private data

} // namespace addrspaces

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

    BasicBlock toBasicBlock() const;
};

/** \struct InstrumentationFunctions
 * \brief Structure of pointers to instrumentation functions in the module
 */
struct InstrumentationFunctions {
    // ----- C instrumentation API ----- //

    llvm::Function* rocmStamp;

    // hipInstrumenter
    llvm::Function *hipNewInstrumenter, *hipInstrumenterToDevice,
        *hipInstrumenterFromDevice, *hipInstrumenterRecord,
        *hipGetNextInstrumenter, *freeHipInstrumenter;

    // hipStateRecoverer
    llvm::Function *hipNewStateRecoverer, *hipStateRecovererRegisterPointer,
        *hipStateRecovererRollback, *freeHipStateRecoverer;

    // hipQueueInfo
    llvm::Function *newHipQueueInfo, *freeHipQueueInfo,
        *hipQueueInfoAllocBuffer, *hipQueueInfoAllocOffsets,
        *hipQueueInfoRecord;

    // hipGlobalQueueInfo
    llvm::Function *newGlobalMemoryQueueInfo, *hipGlobalMemQueueInfoToDevice,
        *hipGlobalMemQueueInfoRecord, *freeHipGlobalMemoryQueueInfo;

    // hipCUMemQueueInfo
    llvm::Function *newCUMemQueueInfo, *hipCUMemQueueInfoToDevice,
        *hipCUMemQueueInfoRecord, *freeHipCUMemoryQueueInfo;

    // hipChunkAllocator
    llvm::Function *newHipChunkAllocator, *hipChunkAllocatorToDevice,
        *hipChunkAllocatorRecord, *freeChunkAllocator;

    // hipChunkAllocator
    llvm::Function *newHipCUChunkAllocator, *hipCUChunkAllocatorToDevice,
        *hipCUChunkAllocatorRecord, *freeCUChunkAllocator;

    /** ctor
     * \brief Forward-declare instrumentation functions in the module, and
     * returns pointers to them
     * */
    InstrumentationFunctions(llvm::Module& mod);
};

struct CfgFunctions {
    // ----- CfgCountersFunctions ----- //
    llvm::Function* _hip_store_ctr;

    llvm::Function* _hip_wave_ctr_get_offset;

    /** ctor
     * \brief Forward-declare cfg instrumentation functions in the module, and
     * returns pointers to them
     * */
    CfgFunctions(llvm::Module& mod);
};

struct TracingFunctions {
    // ----- Tracing functions ----- //

    // Utils
    llvm::Function* _hip_wave_id_1d;

    // ThreadQueue
    llvm::Function *_hip_get_trace_offset, *_hip_create_event;

    // WaveQueue
    llvm::Function *_hip_get_wave_trace_offset, *_hip_create_wave_event;

    // Managed Queue
    llvm::Function* _hip_get_global_memory_trace_ptr;

    llvm::Function* _hip_chunk_allocator_alloc;

    llvm::Function* _hip_get_cache_aligned_registry;

    /** ctor
     * \brief Forward-declare tracing functions in the module, and
     * returns pointers to them
     * */
    TracingFunctions(llvm::Module& mod);
};

// ----- IR Utils ----- //

/** \fn isDeviceModule
 * \brief Returns true if the module has a device target triple
 */
bool isDeviceModule(const llvm::Module& mod);

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

/** \fn hasUse
 * \brief Returns true is one of the users of the value matches the predicate
 */
bool hasUse(const llvm::Value* v,
            std::function<bool(const llvm::Value*)> predicate);

template <typename T>
llvm::BasicBlock::iterator firstInstructionOf(llvm::Function& f) {
    return findInstruction(
        f, [](const llvm::Instruction* i) { return llvm::isa<T>(i); });
}

void setInsertPointPastAllocas(llvm::IRBuilderBase& builder, llvm::Function& f);

/** \fn firstCallToFunction
 * \brief Returns the first call to the given symbol
 */
llvm::CallInst* firstCallToFunction(llvm::Function& f,
                                    const std::string& function);

/** \fn hasFunctionCall
 * \brief Returns whether the instruction is a call to the given symbol
 */
bool hasFunctionCall(const llvm::Instruction& f, const std::string& function);

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

/** \fn readFirstLane
 * \brief Converts a vector u32 to a scalar u32
 */
llvm::Value* readFirstLane(llvm::IRBuilder<>& builder, llvm::Value* i32_vgpr);

/** \fn readFirstLaneI64
 * \brief Converts an i64 (or ptr) VGPR value to an i64 that will be (hopefully)
 * constrained to be stored in a sgpr
 */
llvm::Value* readFirstLaneI64(llvm::IRBuilder<>& builder,
                              llvm::Value* i64_vgpr);

/** \fn readFirstLaneI64
 * \brief Converts an i64 (or ptr) VGPR value to an i64 that will be stored in
 * the given register name
 */
llvm::Value* readFirstLaneI64(llvm::IRBuilder<>& builder, llvm::Value* i64_vgpr,
                              uint8_t reg);

/** \fn initializeSGPR
 * \brief Initializes a i32 SGPR to the given value
 */
llvm::Value* initializeSGPR(llvm::IRBuilder<>& builder, uint32_t initializer,
                            std::string_view reg);

/** \fn initializeSGPR64
 * \brief Initializes a i64 SGPR to the given value
 */
llvm::Value* initializeSGPR64(llvm::IRBuilder<>& builder, uint64_t initializer,
                              std::string_view reg);

/** \fn getFunction
 * \brief Util to get the handle to a function in LLVM IR
 */
llvm::Function* getFunction(llvm::Module& mod, llvm::StringRef name,
                            llvm::FunctionType* type);

/** \fn incrementRegisterAsm
 * \brief Returns the inline asm to increment a given hardware register
 */
llvm::InlineAsm* incrementRegisterAsm(llvm::IRBuilder<>& builder,
                                      std::string_view reg, bool carry = false,
                                      std::string_view inc = "1");

/** \fn atomicIncrementAsm
 * \brief Perform an atomic add operation at the address contained by reg_addr,
 * returns it in reg_ret
 */
llvm::InlineAsm* atomicIncrementAsm(llvm::IRBuilder<>& builder,
                                    std::string_view reg_addr,
                                    std::string_view reg_ret,
                                    std::string_view inc = "1");

/** \fn getEventCtorType
 * \brief Return the generic event constructor type
 */
llvm::FunctionType* getEventCtorType(llvm::LLVMContext& context);

/** \fn cloneWithName
 * \brief Clones a function f with a different name, and eventually additional
 * args
 */
llvm::Function& cloneWithName(llvm::Function& f, std::string_view name,
                              llvm::ArrayRef<llvm::Type*> extra_args = {});

/** \fn optimizeFunction
 * \brief Run all optimizations on a given function
 */
void optimizeFunction(llvm::Function& f, llvm::FunctionAnalysisManager& fm);

/** \brief Suffix to distinguish already cloned function, placeholder for a real
 * attribute
 */
constexpr auto cloned_suffix = "__hip_instr_";

inline std::string getClonedName(const std::string& f,
                                 std::string_view prefix) {
    return llvm::Twine(cloned_suffix).concat(prefix).concat(f).str();
}

inline std::string getClonedName(const llvm::Function& f,
                                 std::string_view prefix) {
    return getClonedName(f.getName().str(), prefix);
}

llvm::Function& cloneWithPrefix(llvm::Function& f, std::string_view prefix,
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

/** \fn assertModuleIntegrity
 * \brief Throws if the module is broken
 */
void assertModuleIntegrity(llvm::Module& m);

/** \fn dumpMetadata
 * \brief Print function / instruction metadata
 */
void dumpMetadata(llvm::Function* f);
void dumpMetadata(llvm::Instruction* i);

/** \fn getSubroutineType
 * \brief Return debug subroutine type from function (if any)
 */
llvm::DISubroutineType* getSubroutineType(llvm::Function* f);

} // namespace hip
