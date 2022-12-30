/** \file hip_analyzer_pass
 * \brief Experimental kernel timer
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "ir_codegen.h"

#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

namespace hip {
/** \struct HostPass
 * \brief The Host pass is responsible for adding device stubs for the new
 * instrumented kernels.
 *
 */
struct KernelTimerPass : public llvm::PassInfoMixin<KernelTimerPass> {
    KernelTimerPass() {}

    llvm::PreservedAnalyses run(llvm::Module& mod,
                                llvm::ModuleAnalysisManager& modm);
};

struct KernelTimerFunctions {
    KernelTimerFunctions(llvm::Module& mod);

    llvm::Function *begin_kernel_timer, *end_kernel_timer;
};

} // namespace hip

// ---- Pass setup ----- /

llvm::PassPluginLibraryInfo getKernelTimerPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "kernel-timer", LLVM_VERSION_STRING,
        [](llvm::PassBuilder& pb) {
            pb.registerPipelineStartEPCallback(
                [](llvm::ModulePassManager& pm, llvm::OptimizationLevel Level) {
                    pm.addPass(hip::KernelTimerPass());
                });
        }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return getKernelTimerPluginInfo();
}

hip::KernelTimerFunctions::KernelTimerFunctions(llvm::Module& mod) {
    auto& context = mod.getContext();
    auto* void_type = llvm::Type::getVoidTy(context);
    auto* unqual_ptr_type = llvm::PointerType::getUnqual(context);

    auto void_from_ptr_type =
        llvm::FunctionType::get(void_type, {unqual_ptr_type}, false);

    auto void_from_void_type = llvm::FunctionType::get(void_type, {}, false);

    begin_kernel_timer =
        getFunction(mod, "begin_kernel_timer", void_from_ptr_type);

    end_kernel_timer =
        getFunction(mod, "end_kernel_timer", void_from_void_type);
}

llvm::PreservedAnalyses
hip::KernelTimerPass::run(llvm::Module& mod,
                          llvm::ModuleAnalysisManager& modm) {
    if (isDeviceModule(mod)) {
        // DO NOT run on device code
        return llvm::PreservedAnalyses::all();
    }

    KernelTimerFunctions functions(mod);

    auto* unqual_ptr_type = llvm::PointerType::getUnqual(mod.getContext());

    for (auto& f : mod.functions()) {
        for (auto& bb : f) {
            for (auto& inst : bb) {
                if (hasFunctionCall(inst, "hipLaunchKernel")) {
                    auto* call_to_launch = dyn_cast<llvm::CallInst>(&inst);
                    llvm::IRBuilder<> builder(&inst);
                    builder.CreateCall(
                        functions.begin_kernel_timer,
                        {builder.CreateBitCast(
                            builder.CreateGlobalStringPtr(
                                dyn_cast<llvm::ConstantExpr>(
                                    call_to_launch->getArgOperand(0))
                                    ->getOperand(0)
                                    ->getName()),
                            unqual_ptr_type)});

                    builder.SetInsertPoint(call_to_launch->getNextNode());

                    builder.CreateCall(functions.end_kernel_timer, {});

		    llvm::dbgs() << f << '\n';
                }
            }
        }
    }
    return llvm::PreservedAnalyses::none();
}
