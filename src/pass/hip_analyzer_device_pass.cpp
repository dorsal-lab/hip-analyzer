/** \file hip_analyzer_device_pass
 * \brief Instrumentation pass
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_analyzer_pass.h"

#include <fstream>

#include "llvm/IR/IRBuilder.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <json/json.h>

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

        llvm::SimplifyCFGPass().run(f_original, fm);

        llvm::dbgs() << "Function " << f_original.getName() << '\n'
                     << f_original;

        // Clone the kernel, with extra arguments
        auto& f = cloneWithPrefix(f_original, getInstrumentedKernelPrefix(),
                                  getExtraArguments(mod.getContext()));

        modified |= addParams(f, f_original);

        llvm::errs() << "Function " << f.getName() << '\n';
        f.print(llvm::dbgs(), nullptr);

        auto blocks = fm.getResult<AnalysisPass>(f_original);

        modified |= instrumentFunction(f, f_original, blocks);
    }

    // If we instrumented a kernel, link the necessary utilities function
    if (modified) {
        linkModuleUtils(mod);
    }

    return modified ? llvm::PreservedAnalyses::none()
                    : llvm::PreservedAnalyses::all();
}

bool KernelInstrumentationPass::addParams(llvm::Function& f,
                                          llvm::Function& original_function) {

    // To be quite honest, I am not really sure how this works. It might be
    // possibly buggy in some cases.

    llvm::ValueToValueMapTy vmap;

    llvm::dbgs() << f << "\n#####\n" << original_function;

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

// ----- hip::CfgInstrumentationPass ----- //

const std::string CfgInstrumentationPass::instrumented_prefix = "counters_";
const std::string CfgInstrumentationPass::utils_path = "gpu_pass_instr.ll";

bool CfgInstrumentationPass::isInstrumentableKernel(
    const llvm::Function& f) const {
    return !f.isDeclaration() && !contains(f.getName().str(), cloned_suffix) &&
           f.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL;
}

bool CfgInstrumentationPass::instrumentFunction(
    llvm::Function& f, llvm::Function& original_function,
    AnalysisPass::Result& blocks) {

    auto& context = f.getContext();
    CfgFunctions instrumentation_handlers(*f.getParent());
    auto* instr_ptr = f.getArg(f.arg_size() - 1);

    // Add counters
    auto* counter_type = getCounterType(context);
    auto* array_type = llvm::ArrayType::get(counter_type, blocks.size());

    llvm::IRBuilder<> builder_locals(&f.getEntryBlock());
    setInsertPointPastAllocas(builder_locals, f);

    auto* counters = builder_locals.CreateAlloca(array_type, nullptr,
                                                 llvm::Twine("_bb_counters"));

    // Initialize it!

    builder_locals.CreateMemSet(
        builder_locals.CreatePointerCast(
            counters, llvm::Type::getInt8PtrTy(f.getContext())),
        llvm::ConstantInt::get(counter_type, 0u), blocks.size(),
        llvm::MaybeAlign(llvm::Align::Constant<1>()));

    // Instrument each basic block

    auto& function_block_list = f.getBasicBlockList();
    auto curr_bb = f.begin();
    auto index = 0u;

    for (auto& bb_instr : blocks) {
        while (index < bb_instr.id) {
            ++index;
            ++curr_bb;
        }

        builder_locals.SetInsertPoint(&(*curr_bb),
                                      getFirstNonPHIOrDbgOrAlloca(*curr_bb));

        auto* inbound_ptr = builder_locals.CreateInBoundsGEP(
            array_type, counters,
            {getIndex(0u, context), getIndex(bb_instr.id, context)});

        auto* curr_ptr = builder_locals.CreateLoad(counter_type, inbound_ptr);

        auto* incremented = builder_locals.CreateAdd(
            curr_ptr, llvm::ConstantInt::get(counter_type, 1u));

        auto* store = builder_locals.CreateStore(incremented, inbound_ptr);
    }

    // Call saving method

    for (auto& bb_instr : f) {
        auto terminator = bb_instr.getTerminator();

        // Only call saving method if the terminator is a return
        if (isa<llvm::ReturnInst>(terminator)) {
            builder_locals.SetInsertPoint(terminator);

            // Bitcast to ptr

            auto* array_ptr =
                builder_locals.CreatePointerBitCastOrAddrSpaceCast(
                    counters, counter_type->getPointerTo());

            // Add call
            builder_locals.CreateCall(
                instrumentation_handlers._hip_store_ctr,
                {array_ptr, getIndex(blocks.size(), context), instr_ptr});
        }
    }

    llvm::dbgs() << f;

    return true;
}

llvm::SmallVector<llvm::Type*>
CfgInstrumentationPass::getExtraArguments(llvm::LLVMContext& context) const {
    return {getCounterType(context)->getPointerTo()};
}

void CfgInstrumentationPass::linkModuleUtils(llvm::Module& mod) {
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

// ----- hip::TracingPass ----- //

const std::string TracingPass::instrumented_prefix = "tracing_";
const std::string TracingPass::utils_path = "gpu_pass_instr.ll";

bool TracingPass::isInstrumentableKernel(const llvm::Function& f) const {
    return !f.isDeclaration() && !contains(f.getName().str(), cloned_suffix) &&
           f.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL;
}

bool TracingPass::instrumentFunction(llvm::Function& f,
                                     llvm::Function& original_function,
                                     AnalysisPass::Result& blocks) {

    auto& mod = *f.getParent();
    auto& context = f.getContext();
    TracingFunctions instrumentation_handlers(mod);

    auto* i64_ty = llvm::Type::getInt64Ty(context);
    auto* i8_ptr_ty = llvm::Type::getInt8PtrTy(context);
    auto* offsets_ptr = f.getArg(f.arg_size() - 1);

    // Add counters

    llvm::IRBuilder<> builder_locals(&f.getEntryBlock());
    setInsertPointPastAllocas(builder_locals, f);

    // Create the local counter and initialize it to 0.
    auto* idx = builder_locals.CreateAlloca(i64_ty, 0, nullptr,
                                            llvm::Twine("_trace_idx"));
    builder_locals.CreateStore(llvm::ConstantInt::get(i64_ty, 0), idx);

    auto* storage_ptr =
        builder_locals.CreateBitCast(f.getArg(f.arg_size() - 2), i8_ptr_ty);

    auto* thread_storage = builder_locals.CreateCall(
        event->getOffsetGetter(mod),
        {storage_ptr, offsets_ptr, event->getEventSize(mod)});

    builder_locals.CreateCall(event->getEventCreator(mod),
                              {thread_storage, idx, event->getEventSize(mod),
                               event->getEventCtor(mod), getIndex(0, context)});

    // Start at 1 because the first block is handled separately

    auto& function_block_list = f.getBasicBlockList();
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

        builder_locals.SetInsertPoint(&(*curr_bb),
                                      getFirstNonPHIOrDbgOrAlloca(*curr_bb));

        builder_locals.CreateCall(
            event->getEventCreator(mod),
            {thread_storage, idx, event->getEventSize(mod),
             event->getEventCtor(mod), getIndex(bb_instr.id, context)});
    }

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
