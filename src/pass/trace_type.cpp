/** \file trace_type
 * \brief Implementation of different trace types at instrumentation-time
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_analyzer_pass.h"

#include "ir_codegen.h"

namespace hip {
namespace {

class Event : public TraceType {
  public:
    llvm::Type* getEventType(llvm::LLVMContext& context) const override {
        return llvm::StructType::create(
            context, {llvm::Type::getInt64Ty(context)}, "hipEvent");
    }

    llvm::Function* getEventCtor(llvm::Module& mod) const override {
        return getFunction(mod, "_hip_event_ctor",
                           getEventCtorType(mod.getContext()));
    }
};

class TaggedEvent : public TraceType {
  public:
    llvm::Type* getEventType(llvm::LLVMContext& context) const override {
        auto* i64 = llvm::Type::getInt64Ty(context);
        return llvm::StructType::create(context, {i64, i64}, "hipEvent");
    }

    llvm::Function* getEventCtor(llvm::Module& mod) const override {
        return getFunction(mod, "_hip_tagged_event_ctor",
                           getEventCtorType(mod.getContext()));
    }
};

class WaveState : public TraceType {
  public:
    llvm::Type* getEventType(llvm::LLVMContext& context) const override {
        auto* i64 = llvm::Type::getInt64Ty(context);
        auto* i32 = llvm::Type::getInt32Ty(context);
        return llvm::StructType::create(context, {i64, i64, i64, i32},
                                        "hipEvent");
    }

    llvm::Function* getEventCtor(llvm::Module& mod) const override {
        return getFunction(mod, "_hip_wavestate_ctor",
                           getEventCtorType(mod.getContext()));
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
