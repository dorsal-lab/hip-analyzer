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

    std::pair<llvm::Value*, llvm::Value*>
    getQueueType(llvm::Module& mod) const override {
        return getPair(mod.getContext(), 0, 0);
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

    std::pair<llvm::Value*, llvm::Value*>
    getQueueType(llvm::Module& mod) const override {
        return getPair(mod.getContext(), 1, 0);
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

    virtual llvm::Function* getEventCreator(llvm::Module& mod) const override {
        return TracingFunctions{mod}._hip_create_wave_event;
    }

    llvm::Function* getOffsetGetter(llvm::Module& mod) const override {
        return TracingFunctions{mod}._hip_get_wave_trace_offset;
    }

    std::pair<llvm::Value*, llvm::Value*>
    getQueueType(llvm::Module& mod) const override {
        return getPair(mod.getContext(), 2, 1);
    }

    llvm::Value* getTracingIndex(llvm::Module& module,
                                 llvm::IRBuilder<>& builder) override {
        constexpr auto num_waves = 16u;

        auto* i8_ty = llvm::Type::getInt8Ty(module.getContext());
        auto* i64_ty = llvm::Type::getInt64Ty(module.getContext());
        auto* i64_ptr_ty = i64_ty->getPointerTo();

        auto* array_ty = llvm::ArrayType::get(i64_ty, num_waves);

        auto* array = builder.CreateAlloca(array_ty, 5, nullptr,
                                           llvm::Twine("_trace_idx"));
        auto* array_ptr = builder.CreatePointerCast(array, i64_ptr_ty);

        builder.CreateMemSet(array_ptr, llvm::ConstantInt::get(i8_ty, 0u),
                             num_waves * 8,
                             llvm::MaybeAlign(llvm::Align::Constant<8>()));

        auto* _hip_wave_get_index_in_block = getFunction(
            module, "_hip_wave_get_index_in_block",
            llvm::FunctionType::get(i64_ptr_ty, {i64_ptr_ty}, false));

        return builder.CreateCall(_hip_wave_get_index_in_block, {array_ptr});
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
