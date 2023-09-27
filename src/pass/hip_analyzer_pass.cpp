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

static llvm::cl::opt<bool>
    wave_counters("wave-counters", llvm::cl::desc("Use wavefront counters"),
                  llvm::cl::init(true));

static llvm::cl::opt<std::string>
    trace_type("trace_type", llvm::cl::desc("hip-analyzer trace type"),
               llvm::cl::init("trace-wavestate"));

static llvm::cl::opt<bool>
    do_trace("hip-trace",
             llvm::cl::desc("hip-analyzer add to trace kernel values"),
             llvm::cl::init(false));

static llvm::cl::opt<bool>
    do_replay("hip-replay",
              llvm::cl::desc("hip-aalyzer load existing counters trace"),
              llvm::cl::init(false));

llvm::PassPluginLibraryInfo getHipAnalyzerPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "hip-analyzer", LLVM_VERSION_STRING,
        [](llvm::PassBuilder& pb) {
            pb.registerPipelineParsingCallback(
                [&](llvm::StringRef name, llvm::ModulePassManager& mpm,
                    llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
                    if (name == "hip-analyzer-host") {
                        mpm.addPass(hip::FullInstrumentationHostPass());
                        return true;
                    } else if (name == "hip-analyzer-counters") {
                        mpm.addPass(hip::ThreadCountersInstrumentationPass());
                        return true;
                    } else if (name == "hip-analyzer-wave-counters") {
                        mpm.addPass(hip::WaveCountersInstrumentationPass());
                        return true;
                    } else if (name == "hip-analyzer-trace") {
                        mpm.addPass(hip::TracingPass(trace_type.getValue()));
                        return true;
                    }
                    return false;
                });

            pb.registerOptimizerLastEPCallback(
                [](llvm::ModulePassManager& pm, llvm::OptimizationLevel Level) {
                    if (wave_counters) {
                        pm.addPass(hip::WaveCountersInstrumentationPass());
                    } else {
                        pm.addPass(hip::ThreadCountersInstrumentationPass());
                    }

                    if (do_trace || do_replay) {
                        pm.addPass(hip::TracingPass(trace_type.getValue()));
                    }
                });

            pb.registerPipelineStartEPCallback([](llvm::ModulePassManager& pm,
                                                  llvm::OptimizationLevel
                                                      Level) {
                // Overloading device stubs is only possible before
                // optimizations (all calls would be inlined)
                const auto& cfg_prefix =
                    wave_counters
                        ? hip::WaveCountersInstrumentationPass::CounterType
                        : hip::ThreadCountersInstrumentationPass::CounterType;

                if (do_replay) {
                    pm.addPass(
                        hip::KernelReplayerHostPass(trace_type.getValue()));
                } else if (do_trace) {
                    pm.addPass(hip::FullInstrumentationHostPass(
                        cfg_prefix, trace_type.getValue()));
                } else {
                    pm.addPass(
                        hip::CounterKernelInstrumentationHostPass(cfg_prefix));
                }
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
