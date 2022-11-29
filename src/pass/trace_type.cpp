/** \file trace_type
 * \brief Implementation of different trace types at instrumentation-time
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_analyzer_pass.h"

namespace hip {
namespace {

class Event : public TraceType {
  public:
    llvm::Type* getEventType(llvm::LLVMContext& context) const override {
        return nullptr;
    }

    llvm::Function* getEventCtor(llvm::LLVMContext& context) const override {
        return nullptr;
    }

    llvm::ConstantInt* getEventSize(llvm::LLVMContext& context) const override {
        return nullptr;
    }
};

class TaggedEvent : public TraceType {
  public:
    llvm::Type* getEventType(llvm::LLVMContext& context) const override {
        return nullptr;
    }

    llvm::Function* getEventCtor(llvm::LLVMContext& context) const override {
        return nullptr;
    }

    llvm::ConstantInt* getEventSize(llvm::LLVMContext& context) const override {
        return nullptr;
    }
};

class WaveState : public TraceType {
  public:
    llvm::Type* getEventType(llvm::LLVMContext& context) const override {
        return nullptr;
    }

    llvm::Function* getEventCtor(llvm::LLVMContext& context) const override {
        return nullptr;
    }

    llvm::ConstantInt* getEventSize(llvm::LLVMContext& context) const override {
        return nullptr;
    }
};

} // namespace

std::unique_ptr<TraceType> TraceType::create(const std::string& trace_type) {
    // Here goes
    if (trace_type == "trace-event") {
        return std::make_unique<Event>();
    } else if (trace_type == "trace-tagged") {
        return std::make_unique<TaggedEvent>();
    } else if (trace_type == "trace-wavestate") {
        return std::make_unique<WaveState>();
    } else {
        return {nullptr};
    }
}

} // namespace hip
