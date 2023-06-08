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
    llvm::errs() << "Function " << fn.getName() << '\n';
    fn.print(llvm::dbgs(), nullptr);

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

        auto& fm = modm.getResult<llvm::FunctionAnalysisManagerModuleProxy>(mod)
                       .getManager();

        optimizeFunction(f_original, fm);

        llvm::dbgs() << "Function " << f_original.getName() << '\n'
                     << f_original;

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

    // LLVM uses SSA, which means the vector has to be passed by value to each
    // basic block as a PHI node to maintain a global state throughout the
    // function. We construct this spaghetti of PHI nodes in two times, by first
    // creating bogey values (ssa_copy from 0) then through a second pass to add
    // the actual incremented value.
    std::map<llvm::BasicBlock*, std::pair<llvm::Value*, llvm::Value*>> idx_map;

    auto& context = f.getContext();
    CfgFunctions instrumentation_handlers(*f.getParent());

    auto* raw_storage_ptr = f.getArg(f.arg_size() - 1); // Last arg

    llvm::IRBuilder<> builder(&f.getEntryBlock());
    builder.SetInsertPointPastAllocas(&f);

    auto* zero = builder.getInt32(0);
    auto* vector_ty = getVectorCounterType(context, blocks.size());
    auto* init_vector =
        llvm::ConstantVector::getSplat(vector_ty->getElementCount(), zero);

    // First pass : create fake (zero) values and increment them
    unsigned int bb_id = 0u;
    for (auto& bb : f) {
        builder.SetInsertPoint(&*bb.getFirstNonPHIOrDbgOrAlloca());

        auto* dummy_idx = builder.CreateIntrinsic(llvm::Intrinsic::ssa_copy,
                                                  {vector_ty}, init_vector);
        auto* incremented =
            getCounterAndIncrement(*f.getParent(), builder, dummy_idx, bb_id);

        idx_map.insert({&bb, {dummy_idx, incremented}});
        ++bb_id;
    }

    // Second pass : replace all uses with PHI nodes from predecessors

    for (auto& bb : f) {
        if (bb.isEntryBlock()) {
            builder.SetInsertPointPastAllocas(&f);
            idx_map[&bb].first->replaceAllUsesWith(init_vector);
            continue;
        }
        builder.SetInsertPoint(&bb.front());

        auto* phi = builder.CreatePHI(vector_ty, 0);

        for (const auto& pred : llvm::predecessors(&bb)) {
            phi->addIncoming(idx_map[pred].second, pred);
        }

        idx_map[&bb].first->replaceAllUsesWith(phi);
        dyn_cast<llvm::Instruction>(idx_map[&bb].first)->eraseFromParent();
        idx_map[&bb].first = phi;
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

            for (auto i = 0u; i < bb_id; ++i) {
                storeCounter(builder, idx_map[&bb].second, storage_ptr, i);
            }
        }
    }

    llvm::dbgs() << f;
    return true;
}

llvm::Value* WaveCountersInstrumentationPass::getCounterAndIncrement(
    llvm::Module& mod, llvm::IRBuilder<>& builder, llvm::Value* vec,
    unsigned bb) {
    static constexpr auto* inline_asm = "s_add_u32 $0, $0, 1";
    static constexpr auto* inline_asm_constraints = "=s,s";

    auto* i32_ty = getScalarCounterType(mod.getContext());
    auto* f_ty = llvm::FunctionType::get(i32_ty, {i32_ty}, false);
    auto* inc_sgpr =
        llvm::InlineAsm::get(f_ty, inline_asm, inline_asm_constraints, true);

    // Extract counter from vector
    auto* extracted = builder.CreateExtractElement(vec, bb);
    // Increment as a SGPR (hopefully!)
    auto* incremented = builder.CreateCall(inc_sgpr, {extracted});
    // Return the vector with inserted value
    return builder.CreateInsertElement(vec, incremented, bb);
}

void WaveCountersInstrumentationPass::storeCounter(llvm::IRBuilder<>& builder,
                                                   llvm::Value* vec,
                                                   llvm::Value* ptr,
                                                   unsigned bb) {
    // We have to set an offset for each store, so construct a string each time!
    // (Hopefully we'll get std::format aaaanytime now)
    constexpr unsigned dword_size = 4u; // 4 bytes for a 32 bit integer
    static constexpr auto* inline_asm_constraints = "s,s";

    std::stringstream ss;
    ss << "s_store_dword $0, $1, " << dword_size * bb;

    auto* i32_ty = builder.getInt32Ty();
    auto* ptr_ty = builder.getPtrTy();

    // readfirstlane inline asm
    auto* readlane_ty = llvm::FunctionType::get(i32_ty, {i32_ty}, false);
    auto* readline_asm = llvm::InlineAsm::get(
        readlane_ty, "v_readfirstlane_b32 $0, $1", "=s,v", true);

    // store inline asm
    auto* f_ty =
        llvm::FunctionType::get(builder.getVoidTy(), {i32_ty, ptr_ty}, false);
    auto* store_sgpr =
        llvm::InlineAsm::get(f_ty, ss.str(), inline_asm_constraints, true);

    // Extract value from vector
    auto* extracted = builder.CreateExtractElement(vec, bb);
    // Store in SGPR
    auto* sgpr = builder.CreateCall(readline_asm, {extracted});
    // Create call to inline asm to store the value
    builder.CreateCall(store_sgpr, {sgpr, ptr});
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

const std::string TracingPass::instrumented_prefix = "tracing_";
const std::string TracingPass::utils_path = "gpu_pass_instr.ll";

bool TracingPass::instrumentFunction(llvm::Function& f,
                                     llvm::Function& original_function,
                                     AnalysisPass::Result& blocks) {

    auto& mod = *f.getParent();
    auto& context = f.getContext();
    TracingFunctions instrumentation_handlers(mod);

    auto* i8_ptr_ty = llvm::Type::getInt8PtrTy(context);
    auto* offsets_ptr = f.getArg(f.arg_size() - 1);

    // Add counters

    llvm::IRBuilder<> builder(&f.getEntryBlock());
    setInsertPointPastAllocas(builder, f);

    // Create the local counter and initialize it to 0.
    event->initTracingIndices(f);

    auto* storage_ptr =
        builder.CreateBitCast(f.getArg(f.arg_size() - 2), i8_ptr_ty);

    auto* thread_storage =
        event->getThreadStorage(mod, builder, storage_ptr, offsets_ptr);

    /*
    builder.CreateCall(event->getEventCreator(mod),
                              {thread_storage, idx, event->getEventSize(mod),
                               event->getEventCtor(mod), getIndex(0, context)});
    */

    // Start at 1 because the first block is handled separately

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

        event->createEvent(mod, builder, thread_storage, counter, bb_instr.id);
    }

    event->finalizeTracingIndices(f);

    llvm::dbgs() << f;

    return true;
}

llvm::SmallVector<llvm::Type*>
TracingPass::getExtraArguments(llvm::LLVMContext& context) const {
    return {event->getEventType(context)->getPointerTo(),
            llvm::Type::getInt64PtrTy(context)};
}

void TracingPass::linkModuleUtils(llvm::Module& mod) {}

} // namespace hip
