/** \file trace_type
 * \brief Implementation of different trace types at instrumentation-time
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_analyzer_pass.h"

#include "ir_codegen.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Value.h"

namespace hip {
namespace {

// ----- Thread or Wave event production mode ----- //

class ThreadTrace : public TraceType {
  public:
    const std::map<llvm::BasicBlock*, std::pair<llvm::Value*, llvm::Value*>>&
    initTracingIndices(llvm::Function& kernel) override final {
        idx_map.clear(); // Reset map

        auto& entry = kernel.getEntryBlock();
        llvm::IRBuilder<> builder_locals(&entry);
        setInsertPointPastAllocas(builder_locals, kernel);

        auto* i32_ty = getCounterType(*kernel.getParent());
        auto* trace_idx = builder_locals.CreateAlloca(
            i32_ty, 0, nullptr, llvm::Twine("_trace_idx"));

        for (auto& bb : kernel) {
            builder_locals.SetInsertPoint(&bb);
            auto* incremented = getCounterAndIncrement(
                *kernel.getParent(), builder_locals, trace_idx);

            idx_map.insert({&bb, {trace_idx, incremented}});
            // incremented will not actually be used as it is stored in the
            // allocated
        }

        return idx_map;
    }

    llvm::Value* getCounterAndIncrement(llvm::Module& mod,
                                        llvm::IRBuilder<>& builder,
                                        llvm::Value* counter) override final {
        if (counter->getType()->isPointerTy()) {
            // Load the value
            auto* loaded = builder.CreateLoad(
                getCounterType(mod),
                counter); // We'll have to deal with the deprecation warning
                          // until AMD decides to enable opaque pointers in
                          // device compilation

            // Increment the value
            auto* inced = builder.CreateAdd(loaded, builder.getInt32(1));

            // Store the old value
            builder.CreateStore(inced, counter);
            return inced;
        } else {
            // Increment & return the value
            return builder.CreateAdd(counter, builder.getInt32(1));
        }
    }

    void finalizeTracingIndices(llvm::Function& kernel) override {
        // Everything was already computed when initializing, do nothing
    }

    llvm::Value* traceIdxAtBlock(llvm::BasicBlock& bb) override {
        return idx_map.at(&bb).first;
    }

  private:
    std::map<llvm::BasicBlock*, std::pair<llvm::Value*, llvm::Value*>> idx_map;
};

/** \class WaveTrace
 * \brief A trace whose producer is a wavefront, and not a single thread. We
 * have to ensuire that the trace index is global to the whole wavefront and
 * that a single event is created no matter the number of active threads
 */
class WaveTrace : public TraceType {
  public:
    const std::map<llvm::BasicBlock*, std::pair<llvm::Value*, llvm::Value*>>&
    initTracingIndices(llvm::Function& kernel) override final {
        idx_map.clear(); // Reset map

        auto& entry = kernel.getEntryBlock();
        llvm::IRBuilder<> builder(&entry);

        for (auto& bb : kernel) {
            llvm::dbgs() << "BB : \n" << bb << '\n';
            builder.SetInsertPoint(&*bb.getFirstNonPHIOrDbgOrAlloca());

            if (bb.isEntryBlock()) {
                // Initialize s[20:21] to base offset (thread storage)
                // initializeSGPR(builder, 0, "s20");
            }

            auto* incremented =
                getCounterAndIncrement(*kernel.getParent(), builder, nullptr);

            idx_map.insert({&bb, {incremented, incremented}});
        }

        return idx_map;
    }

    llvm::Value* getThreadStorage(llvm::Module& mod, llvm::IRBuilder<>& builder,
                                  llvm::Value* storage_ptr,
                                  llvm::Value* offsets_ptr) override final {

        auto* i64_ty = builder.getInt64Ty();

        auto* vgpr = builder.CreatePtrToInt(
            builder.CreateCall(getOffsetGetter(mod),
                               {storage_ptr, offsets_ptr, getEventSize(mod)}),
            i64_ty);

        // Because readfirstlane is only for 32 bit integers, we have to perform
        // two readlanes then assemble the result after a zext

        thread_storage = readFirstLaneI64(builder, vgpr, 20);
        return thread_storage;
    }

    void finalizeTracingIndices(llvm::Function& kernel) override {}

    llvm::Value* getCounterAndIncrement(llvm::Module& mod,
                                        llvm::IRBuilder<>& builder,
                                        llvm::Value* counter) override final {
        auto* inc_lsb = incrementRegisterAsm(builder, "s20", false, "24");
        auto* inc_msb = incrementRegisterAsm(builder, "s21", true, "0");

        auto* incremented = builder.CreateCall(inc_lsb, {});
        builder.CreateCall(inc_msb, {});

        return incremented;
    }

    llvm::Value* traceIdxAtBlock(llvm::BasicBlock& bb) override {
        const auto [pre_inc, post_inc] = idx_map.at(&bb);
        return pre_inc;
    }

    virtual void createEvent(llvm::Module& mod, llvm::IRBuilder<>& builder,
                             llvm::Value* thread_storage, llvm::Value* counter,
                             uint64_t bb) override = 0;

  protected:
    llvm::Value* thread_storage = nullptr;

  private:
    std::map<llvm::BasicBlock*, std::pair<llvm::Value*, llvm::Value*>> idx_map;
};

// ----- Event types ----- //

class Event : public ThreadTrace {
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

class TaggedEvent : public ThreadTrace {
  public:
    llvm::Type* getEventType(llvm::LLVMContext& context) const override {
        auto* i64 = llvm::Type::getInt64Ty(context);
        return llvm::StructType::create(context, {i64, i64}, "hipTaggedEvent");
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

class WaveState : public WaveTrace {
  public:
    llvm::Type* getEventType(llvm::LLVMContext& context) const override {
        auto* i64 = llvm::Type::getInt64Ty(context);
        auto* i32 = llvm::Type::getInt32Ty(context);
        return llvm::StructType::create(context, {i64, i64, i32, i32},
                                        "hipWaveState");
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

    void createEvent(llvm::Module& mod, llvm::IRBuilder<>& builder,
                     llvm::Value* thread_storage, llvm::Value* counter,
                     uint64_t bb) override {
        // We are not calling a "simple" device function. This constructor is
        // handwritten in assembly.
        auto* int64_ty = builder.getInt64Ty();
        auto* ptr_ty = builder.getPtrTy();
        auto* ctor_ty = llvm::FunctionType::get(builder.getVoidTy(),
                                                {int64_ty, ptr_ty}, false);

        auto* ctor = llvm::InlineAsm::get(ctor_ty, wave_event_ctor_asm,
                                          wave_event_ctor_constraints, true);

        // llvm::dbgs() << "Base storage " << *thread_storage << '\n';
        // llvm::dbgs() << "Counter " << *counter << '\n';

        thread_storage = builder.CreateIntToPtr(thread_storage, ptr_ty);

        builder.CreateCall(ctor,
                           {getIndex(bb, mod.getContext()), thread_storage});
    }

  private:
    static constexpr auto* wave_event_ctor_asm =
        // Prepare payload
        "s_memrealtime s[24:25]\n"                // timestamp
        "s_mov_b64 s[26:27], exec\n"              // exec mask
        "s_getreg_b32 s22, hwreg(HW_REG_HW_ID)\n" // hw_id
        "s_mov_b32 s23, $0\n"                     // bb

        // Write to mem
        "s_store_dwordx4 s[24:27], s[20:21], 0\n"
        "s_store_dwordx2 s[22:23], s[20:21], 4\n";
    static constexpr auto* wave_event_ctor_constraints =
        "i,s,~{s22},~{s23},~{s24},~{s25},~{s26},~{s27}," // Temp values
        "~{s20},~{s21}";                                 // Address
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
