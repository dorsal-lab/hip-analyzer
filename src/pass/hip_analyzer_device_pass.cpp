/** \file hip_analyzer_device_pass
 * \brief Instrumentation pass
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_analyzer_pass.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "ir_codegen.h"

namespace hip {

// ----- hip::AnalysisPass ----- //

llvm::AnalysisKey AnalysisPass::Key;

AnalysisPass::Result AnalysisPass::run(llvm::Function& fn,
                                       llvm::FunctionAnalysisManager& fam) {
    llvm::errs() << "Function " << fn.getName() << '\n';
    fn.print(llvm::dbgs(), nullptr);

    Result blocks;

    auto i = 0u;
    for (auto& bb : fn) {
        if (isBlockInstrumentable(bb)) {
            blocks.emplace_back(getBlockInfo(bb, i));
        }

        ++i;
    }

    return blocks;
}

// ----- hip::CfgInstrumentationPass ----- //

const std::string CfgInstrumentationPass::instrumented_prefix = "counters_";
const std::string CfgInstrumentationPass::utils_path = "gpu_pass_instr.ll";

llvm::PreservedAnalyses
CfgInstrumentationPass::run(llvm::Module& mod,
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

        llvm::errs() << "Function " << f_original.getName() << '\n';
        f_original.print(llvm::dbgs(), nullptr);

        auto& f =
            cloneWithPrefix(f_original, instrumented_prefix,
                            {getCounterType(mod.getContext())->getPointerTo()});

        modified |= addParams(f, f_original);

        llvm::errs() << "Function " << f.getName() << '\n';
        f.print(llvm::dbgs(), nullptr);

        auto& fm = modm.getResult<llvm::FunctionAnalysisManagerModuleProxy>(mod)
                       .getManager();

        modified |= instrumentFunction(f, f_original, fm);
    }

    // Add necessary functions
    linkModuleUtils(mod);

    return modified ? llvm::PreservedAnalyses::none()
                    : llvm::PreservedAnalyses::all();
}

bool CfgInstrumentationPass::isInstrumentableKernel(llvm::Function& f) {
    return !f.isDeclaration() && !contains(f.getName().str(), cloned_suffix) &&
           f.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL;
}

bool CfgInstrumentationPass::addParams(llvm::Function& f,
                                       llvm::Function& original_function) {

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

bool CfgInstrumentationPass::instrumentFunction(
    llvm::Function& f, llvm::Function& original_function,
    llvm::FunctionAnalysisManager& fm) {

    auto blocks = fm.getResult<AnalysisPass>(f);

    auto& context = f.getContext();
    auto instrumentation_handlers = declareInstrumentation(*f.getParent());
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

    f.print(llvm::dbgs(), nullptr);

    return true;
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

    linker.linkInModule(std::move(utils_mod));

    // Remove [[clang::optnone]] and add [[clang::always_inline]]
    // attributes

    auto instrumentation_handlers = declareInstrumentation(mod);

    instrumentation_handlers._hip_store_ctr->removeFnAttr(
        llvm::Attribute::OptimizeNone);
    instrumentation_handlers._hip_store_ctr->removeFnAttr(
        llvm::Attribute::NoInline);
    instrumentation_handlers._hip_store_ctr->addFnAttr(
        llvm::Attribute::AlwaysInline);
}

// ----- hip::TracingPass ----- //

llvm::PreservedAnalyses TracingPass::run(llvm::Module& mod,
                                         llvm::ModuleAnalysisManager& modm) {
    return llvm::PreservedAnalyses::all();
}

bool TracingPass::instrumentFunction(llvm::Function& f) {
    // auto& blocks = getAnalysis<AnalysisPass>(f).getBlocks();

    return false;
}

} // namespace hip
