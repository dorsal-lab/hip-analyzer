/** \file hip_analyzer_pass
 * \brief Instrumentation pass
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_analyzer_pass.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

static llvm::cl::opt<std::string>
    kernel_name("kernel-name", llvm::cl::desc("Specify kernel name"),
                llvm::cl::value_desc("kernel"));

llvm::PassPluginLibraryInfo getHipAnalyzerPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "hip-analyzer", LLVM_VERSION_STRING,
        [](llvm::PassBuilder& pb) {
            pb.registerPipelineParsingCallback(
                [&](llvm::StringRef name, llvm::ModulePassManager& mpm,
                    llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
                    if (name == "hip-analyzer-host") {
                        mpm.addPass(hip::HostPass());
                        return true;
                    } else if (name == "hip-analyzer-counters") {
                        mpm.addPass(hip::CfgInstrumentationPass());
                        return true;
                    }
                    return false;
                });

            pb.registerPipelineStartEPCallback(
                [](llvm::ModulePassManager& pm, llvm::OptimizationLevel Level) {
                    pm.addPass(hip::CfgInstrumentationPass());
                    pm.addPass(hip::HostPass());
                });
            pb.registerAnalysisRegistrationCallback(
                [](llvm::FunctionAnalysisManager& fam) {
                    fam.registerPass([&] { return hip::AnalysisPass(); });
                });
        }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return getHipAnalyzerPluginInfo();
}
