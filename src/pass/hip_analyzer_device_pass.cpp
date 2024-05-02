/** \file hip_analyzer_device_pass
 * \brief Instrumentation pass
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_analyzer_pass.h"

#include <cstdint>
#include <fstream>

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"

#include <json/json.h>
#include <sstream>

#include "ir_codegen.h"

namespace hip {

// ----- hip::AnalysisPass ----- //

llvm::AnalysisKey AnalysisPass::Key;

AnalysisPass::Result AnalysisPass::run(llvm::Function& fn,
                                       llvm::FunctionAnalysisManager& fam) {
    Result blocks;

    std::vector<hip::BasicBlock> blocks_legacy;

    auto i = 0u;
    for (auto& bb : fn) {
        if (isBlockInstrumentable(bb)) {
            blocks.emplace_back(getBlockInfo(bb, i));
            blocks_legacy.push_back(blocks.back().toBasicBlock());
        }

        ++i;
    }

    Json::Value root{Json::objectValue};
    std::ifstream in("hip_analyzer.json");
    if (in.is_open()) {
        in >> root;
    }

    auto report = BasicBlock::jsonArray(blocks_legacy);
    std::stringstream ss;
    ss << report;

    Json::Value new_report;
    ss >> new_report;

    root[fn.getName().str()] = new_report;

    std::ofstream out("hip_analyzer.json");
    out << root;
    out.close();

    return blocks;
}

// ----- hip::KernelInstrumentationPass ----- //

const std::string KernelInstrumentationPass::utils_path = []() -> std::string {
    if (auto* env = std::getenv("HIP_ANALYZER_INSTR")) {
        return env;
    } else {
        return "gpu_pass_instr.ll";
    }
}();

llvm::PreservedAnalyses
KernelInstrumentationPass::run(llvm::Module& mod,
                               llvm::ModuleAnalysisManager& modm) {

    if (!isDeviceModule(mod)) {
        // DO NOT run on host code
        return llvm::PreservedAnalyses::all();
    }

    bool modified = false;
    for (auto& f_original : mod.functions()) {
        if (!isInstrumentableKernel(f_original)) {
            continue;
        }

        // getSubroutineType(f_original)->printTree(llvm::dbgs());

        auto& fm = modm.getResult<llvm::FunctionAnalysisManagerModuleProxy>(mod)
                       .getManager();

        optimizeFunction(f_original, fm);

        // Clone the kernel, with extra arguments
        auto& f = cloneWithPrefix(f_original, getInstrumentedKernelPrefix(),
                                  getExtraArguments(mod.getContext()));

        modified |= addParams(f, f_original);

        llvm::errs() << "Function " << f.getName() << '\n';

        auto blocks = fm.getResult<AnalysisPass>(f_original);

        modified |= instrumentFunction(f, f_original, blocks);
    }

    // If we instrumented a kernel, link the necessary utilities function
    if (modified) {
        linkModuleUtils(mod);
    }

    // assertModuleIntegrity(mod);

    return modified ? llvm::PreservedAnalyses::none()
                    : llvm::PreservedAnalyses::all();
}

bool KernelInstrumentationPass::isInstrumentableKernel(
    const llvm::Function& f) const {
    return !f.isDeclaration() && // Is it a function definition
           f.getCallingConv() ==
               llvm::CallingConv::AMDGPU_KERNEL && // Is it a kernel
           !contains(f.getName().str(),
                     cloned_suffix) && // Is it not already cloned
           !contains(f.getName().str(),
                     dummy_kernel_name); // Is it *not* a dummy kernel
}

bool KernelInstrumentationPass::addParams(llvm::Function& f,
                                          llvm::Function& original_function) {

    // To be quite honest, I am not really sure how this works. It might be
    // possibly buggy in some cases.

    llvm::ValueToValueMapTy vmap;

    for (auto it1 = original_function.arg_begin(), it2 = f.arg_begin();
         it1 != original_function.arg_end(); ++it1, ++it2) {
        vmap[&*it1] = &*it2;
    }
    llvm::SmallVector<llvm::ReturnInst*, 8> returns;

    llvm::CloneFunctionInto(&f, &original_function, vmap,
                            llvm::CloneFunctionChangeType::LocalChangesOnly,
                            returns);

    return true;
}

void KernelInstrumentationPass::linkModuleUtils(llvm::Module& mod) {
    llvm::Linker linker(mod);
    auto& context = mod.getContext();
    context.setDiscardValueNames(false);

    // Load compiled module
    llvm::SMDiagnostic diag;
    auto utils_mod = llvm::parseIRFile(utils_path, diag, context);
    if (!utils_mod) {
        llvm::errs() << diag.getMessage() << '\n';
        throw std::runtime_error("CfgInstrumentationPass::linkModuleUtils()"
                                 " : Could not load utils module");
    }

    auto reopt = [](auto& f) {
        f.removeFnAttr(llvm::Attribute::OptimizeNone);
        f.removeFnAttr(llvm::Attribute::NoInline);
        f.addFnAttr(llvm::Attribute::AlwaysInline);
    };

    // Mark all functions as always inline, at link time.
    for (llvm::Function& f : utils_mod->functions()) {
        reopt(f);
    }

    linker.linkInModule(std::move(utils_mod));

    // Remove [[clang::optnone]] and add [[clang::always_inline]]
    // attributes

    CfgFunctions instrumentation_handlers(mod);

    instrumentation_handlers._hip_store_ctr->removeFnAttr(
        llvm::Attribute::OptimizeNone);
    instrumentation_handlers._hip_store_ctr->removeFnAttr(
        llvm::Attribute::NoInline);
    instrumentation_handlers._hip_store_ctr->addFnAttr(
        llvm::Attribute::AlwaysInline);
}

// ----- hip::CfgInstrumentationPass ----- //

const std::string ThreadCountersInstrumentationPass::instrumented_prefix =
    "counters_";

bool ThreadCountersInstrumentationPass::instrumentFunction(
    llvm::Function& f, llvm::Function& original_function,
    AnalysisPass::Result& blocks) {

    auto& context = f.getContext();
    CfgFunctions instrumentation_handlers(*f.getParent());
    auto* instr_ptr = f.getArg(f.arg_size() - 1);

    // Add counters
    auto* counter_type = getCounterType(context);
    auto* array_type =
        llvm::ArrayType::get(getCounterType(context), blocks.size());

    llvm::IRBuilder<> builder(&f.getEntryBlock());
    setInsertPointPastAllocas(builder, f);

    auto* counters =
        builder.CreateAlloca(array_type, nullptr, llvm::Twine("_bb_counters"));

    // Initialize it!

    builder.CreateMemSet(counters, llvm::ConstantInt::get(counter_type, 0u),
                         blocks.size(),
                         llvm::MaybeAlign(llvm::Align::Constant<1>()));

    // Instrument each basic block

    auto curr_bb = f.begin();
    auto index = 0u;

    for (auto& bb_instr : blocks) {
        while (index < bb_instr.id) {
            ++index;
            ++curr_bb;
        }

        if (!curr_bb->isEntryBlock()) {
            // First block is already at the right position
            builder.SetInsertPoint(&(*curr_bb),
                                   getFirstNonPHIOrDbgOrAlloca(*curr_bb));
        }

        auto* counter_ptr = builder.CreateInBoundsGEP(
            array_type, counters,
            {builder.getInt32(0), builder.getInt32(bb_instr.id)});

        auto* curr_ptr = builder.CreateLoad(counter_type, counter_ptr);

        auto* incremented = builder.CreateAdd(
            curr_ptr, llvm::ConstantInt::get(counter_type, 1u));

        builder.CreateStore(incremented, counter_ptr);
    }

    // Call saving method

    for (auto& bb_instr : f) {
        auto terminator = bb_instr.getTerminator();

        // Only call saving method if the terminator is a return
        if (isa<llvm::ReturnInst>(terminator)) {
            builder.SetInsertPoint(terminator);

            llvm::dbgs() << *counters->getType() << '\n';
            // Add call
            llvm::dbgs() << *builder.CreateCall(
                                instrumentation_handlers._hip_store_ctr,
                                {builder.CreateAddrSpaceCast(
                                     counters,
                                     llvm::PointerType::getUnqual(context)),
                                 builder.getInt64(blocks.size()), instr_ptr})
                         << '\n';
        }
    }

    llvm::dbgs() << f;

    return true;
}

llvm::SmallVector<llvm::Type*>
ThreadCountersInstrumentationPass::getExtraArguments(
    llvm::LLVMContext& context) const {
    return {llvm::PointerType::getUnqual(context)};
}

// ----- hip::WaveCfgInstrumentationPass ----- //

const std::string WaveCountersInstrumentationPass::instrumented_prefix =
    "wave_counters_";

bool WaveCountersInstrumentationPass::instrumentFunction(
    llvm::Function& f, llvm::Function& original_function,
    AnalysisPass::Result& blocks) {

    CfgFunctions instrumentation_handlers(*f.getParent());

    auto* raw_storage_ptr = f.getArg(f.arg_size() - 1); // Last arg

    llvm::IRBuilder<> builder(&f.getEntryBlock());

    // First pass : create fake (zero) values and increment them
    unsigned int bb_id = 0u;
    for (auto& bb : f) {
        if (bb.isEntryBlock()) {
            builder.SetInsertPoint(&bb.front());
            initializeSGPR(builder, 0, index_reg_str);
        } else {
            builder.SetInsertPoint(&*bb.getFirstNonPHIOrDbgOrAlloca());
        }

        // Counting register is fixed as s20, being the first non-allocated
        // scalar in the kernel prelude
        getCounterAndIncrement(*f.getParent(), builder, bb_id, index_reg_str);

        ++bb_id;
    }

    // Fetch counter offset

    builder.SetInsertPointPastAllocas(&f);
    auto* storage_ptr_vgpr =
        builder.CreateCall(instrumentation_handlers._hip_wave_ctr_get_offset,
                           {raw_storage_ptr, builder.getInt32(blocks.size())});

    auto* storage_ptr = hip::readFirstLaneI64(builder, storage_ptr_vgpr);

    // Save for all terminating basic blocks
    for (auto& bb : f) {
        auto* terminator = bb.getTerminator();
        if (isa<llvm::ReturnInst>(terminator)) {
            builder.SetInsertPoint(terminator);

            storeCounter(builder, storage_ptr, 0, index_reg_str);
        }
    }

    llvm::dbgs() << f;
    return true;
}

llvm::Value* WaveCountersInstrumentationPass::getCounterAndIncrement(
    llvm::Module& mod, llvm::IRBuilder<>& builder, unsigned bb,
    std::string_view reg) {

    auto* inc_sgpr = incrementRegisterAsm(builder, reg);
    return builder.CreateCall(inc_sgpr, {});
}

void WaveCountersInstrumentationPass::storeCounter(llvm::IRBuilder<>& builder,
                                                   llvm::Value* ptr,
                                                   unsigned bb,
                                                   std::string_view reg) {
    constexpr unsigned dword_size = 4u; // 4 bytes for a 32 bit integer

    auto instr = llvm::Twine("s_store_dword  ")
                     .concat(reg)
                     .concat(", $0, ")
                     .concat(std::to_string(dword_size * bb))
                     .concat("\ns_waitcnt lgkmcnt(0)\ns_dcache_wb\n")
                     .str();
    auto constraints = "s";

    auto* ptr_ty = builder.getPtrTy();

    // store inline asm
    auto* f_ty = llvm::FunctionType::get(builder.getVoidTy(), {ptr_ty}, false);
    auto* store_sgpr = llvm::InlineAsm::get(f_ty, instr, constraints, true);

    builder.CreateCall(store_sgpr, {ptr});
}

llvm::SmallVector<llvm::Type*>
WaveCountersInstrumentationPass::getExtraArguments(
    llvm::LLVMContext& context) const {
    return {llvm::PointerType::getUnqual(context)};
}

llvm::VectorType* WaveCountersInstrumentationPass::getVectorCounterType(
    llvm::LLVMContext& context, uint64_t bb_count) {
    return llvm::VectorType::get(getScalarCounterType(context),
                                 llvm::ElementCount::getFixed(bb_count));
}

// ----- hip::TracingPass ----- //

const std::string TracingPass::utils_path = "gpu_pass_instr.ll";

bool TracingPass::instrumentFunction(llvm::Function& f,
                                     llvm::Function& original_function,
                                     AnalysisPass::Result& blocks) {

    auto& mod = *f.getParent();
    TracingFunctions instrumentation_handlers(mod);

    auto* offsets_ptr = f.getArg(f.arg_size() - 1);

    // Add counters

    llvm::IRBuilder<> builder(&f.getEntryBlock());

    auto* ptr_ty = builder.getPtrTy();
    setInsertPointPastAllocas(builder, f);

    // Create the local counter and initialize it to 0.
    event->initTracingIndices(f);

    auto* storage_ptr =
        builder.CreateBitCast(f.getArg(f.arg_size() - 2), ptr_ty);

    auto* thread_storage =
        event->getThreadStorage(mod, builder, storage_ptr, offsets_ptr);

    // Start at 1 because the first block is handled separately

    auto* terminator = f.begin()->getTerminator();
    if (llvm::isa<llvm::ReturnInst>(terminator)) {
        builder.SetInsertPoint(terminator);
        event->finalize(builder);
    }

    auto curr_bb = ++f.begin();
    auto index = 1u;

    for (auto& bb_instr : blocks) {
        if (bb_instr.id == 0) {
            // Already instrumented above
            continue;
        }
        while (index < bb_instr.id) {
            ++index;
            ++curr_bb;
        }

        builder.SetInsertPoint(&(*curr_bb),
                               getFirstNonPHIOrDbgOrAlloca(*curr_bb));

        auto* counter = event->traceIdxAtBlock(*curr_bb);

        // Create an event
        event->createEvent(mod, builder, thread_storage, counter, bb_instr.id);

        // For all terminating blocks, may need to add instructions to flush
        auto* terminator = curr_bb->getTerminator();
        if (llvm::isa<llvm::ReturnInst>(terminator)) {
            builder.SetInsertPoint(terminator);
            event->finalize(builder);
        }
    }

    event->finalizeTracingIndices(f);

    llvm::dbgs() << f;

    return true;
}

llvm::SmallVector<llvm::Type*>
TracingPass::getExtraArguments(llvm::LLVMContext& context) const {
    return {llvm::PointerType::getUnqual(context),
            llvm::PointerType::getUnqual(context)};
}

void TracingPass::linkModuleUtils(llvm::Module& mod) {}

} // namespace hip
