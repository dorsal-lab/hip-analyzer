/** \file instrumentation_mode.h
 * \brief Utilities to get the selected instrumentation mode from env
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

#pragma once

namespace hip {

namespace env {

enum class HipAnalyzerMode {
    CountersOnly,
    LowOverheadTracing,
    CountersReplayer,
    GlobalMemory,
    CUMemory,
    ChunkAllocator,
    CUChunkAllocator,
    FromEnv,
    None
};

inline HipAnalyzerMode parseTracingType() {
    auto* env = std::getenv("HIP_ANALYZER_MODE");
    if (env == nullptr) {
        return HipAnalyzerMode::None;
    }

    auto str = std::string(env);

    if (str == "hip-counters") {
        return HipAnalyzerMode::CountersOnly;
    } else if (str == "hip-trace") {
        return HipAnalyzerMode::LowOverheadTracing;
    } else if (str == "hip-replay") {
        return HipAnalyzerMode::CountersReplayer;
    } else if (str == "hip-global-mem") {
        return HipAnalyzerMode::GlobalMemory;
    } else if (str == "hip-cu-mem") {
        return HipAnalyzerMode::CUMemory;
    } else if (str == "hip-chunk-allocator") {
        return HipAnalyzerMode::ChunkAllocator;
    } else if (str == "hip-cu-chunk-allocator") {
        return HipAnalyzerMode::CUChunkAllocator;
    } else {
        return HipAnalyzerMode::None;
    }
}

} // namespace env

struct KernelInstrumentation {
    std::string prefix;
    llvm::SmallVector<llvm::Type*> extra_args;
};

} // namespace hip
