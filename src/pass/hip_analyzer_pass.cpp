/** \file hip_analyzer_pass
 * \brief Instrumentation pass
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_analyzer_pass.h"

#include "llvm/Support/CommandLine.h"

static llvm::cl::opt<bool>
    wave_counters("wave-counters", llvm::cl::desc("Use wavefront counters"),
                  llvm::cl::init(true));

static llvm::cl::opt<std::string>
    trace_type("trace-type", llvm::cl::desc("hip-analyzer trace type"),
               llvm::cl::init([]() -> std::string {
                   if (auto* env = std::getenv("HIP_ANALYZER_TRACE_TYPE")) {
                       return env;
                   } else {
                       return "trace-wavestate-cuchunkallocator";
                   }
               }()));

enum class TracingType {
    CountersOnly,
    LowOverheadTracing,
    CountersReplayer,
    GlobalMemory,
    ChunkAllocator,
    CUChunkAllocator
    // TODO : Add new tracing modes
};

static llvm::cl::opt<TracingType> hip_analyzer_mode(
    llvm::cl::desc("hip-analyzer tracing type"),
    llvm::cl::values(
        clEnumValN(TracingType::CountersOnly, "hip-counters",
                   "Basic blocks counters only"),
        clEnumValN(TracingType::LowOverheadTracing, "hip-trace",
                   "Low-overhead tracing. Separate counters & tracing kernels"),
        clEnumValN(TracingType::CountersReplayer, "hip-replay",
                   "Load existing counters trace"),
        clEnumValN(TracingType::GlobalMemory, "hip-global-mem",
                   "Concurrent global memory, atomics based tracing"),
        clEnumValN(TracingType::ChunkAllocator, "hip-chunk-allocator",
                   "Concurrent buffer based tracing"),
        clEnumValN(
            TracingType::CUChunkAllocator, "hip-cu-chunk-allocator",
            "Concurrent buffer based tracing, with additional CU locality")),
    llvm::cl::init(TracingType::CUChunkAllocator));

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

                    if (hip_analyzer_mode != TracingType::CountersOnly) {
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

                switch (hip_analyzer_mode) {
                case TracingType::CountersOnly:
                    pm.addPass(
                        hip::CounterKernelInstrumentationHostPass(cfg_prefix));
                    break;
                case TracingType::LowOverheadTracing:
                    pm.addPass(hip::FullInstrumentationHostPass(
                        cfg_prefix, trace_type.getValue()));
                    break;
                case TracingType::CountersReplayer:
                    pm.addPass(
                        hip::KernelReplayerHostPass(trace_type.getValue()));
                    break;
                case TracingType::GlobalMemory:
                    pm.addPass(hip::GlobalMemoryQueueHostPass());
                    break;
                case TracingType::ChunkAllocator:
                    pm.addPass(hip::ChunkAllocatorHostPass());
                    break;
                case TracingType::CUChunkAllocator:
                    pm.addPass(hip::CUChunkAllocatorHostPass());
                    break;
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
