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

        auto* i32_ty = llvm::Type::getInt32Ty(kernel.getContext());
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
                counter->getType()->getPointerElementType(),
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
        llvm::IRBuilder<> builder_locals(&entry);
        setInsertPointPastAllocas(builder_locals, kernel);

        auto* zero = builder_locals.getInt32(0);
        auto* i32_ty = llvm::Type::getInt32Ty(kernel.getContext());

        alloca = builder_locals.CreateAlloca(i32_ty, 0, nullptr, "");

        for (auto& bb : kernel) {
            builder_locals.SetInsertPoint(&bb.front());

            auto* dummy_idx = builder_locals.CreateIntrinsic(
                llvm::Intrinsic::ssa_copy, {i32_ty}, {zero});

            auto* incremented = getCounterAndIncrement(
                *kernel.getParent(), builder_locals, dummy_idx);

            idx_map.insert({&bb, {dummy_idx, incremented}});
        }

        return idx_map;
    }

    llvm::Value* getThreadStorage(llvm::Module& mod, llvm::IRBuilder<>& builder,
                                  llvm::Value* storage_ptr,
                                  llvm::Value* offsets_ptr) override final {
        auto* int32_ty = builder.getInt32Ty();
        auto* f_ty = llvm::FunctionType::get(int32_ty, {int32_ty}, false);

        auto* readfirstlane = llvm::InlineAsm::get(f_ty, vgpr_inline_asm,
                                                   vgpr_asm_constraints, true);
        auto* vgpr =
            builder.CreateCall(getOffsetGetter(mod),
                               {storage_ptr, offsets_ptr, getEventSize(mod)});
        return builder.CreateCall(
            readfirstlane, {vgpr}); // Hopefully will be stored in a SGPR!
    }

    void finalizeTracingIndices(llvm::Function& kernel) override {
        // Construct phi nodes from predecessors of each bb, then replace all
        // uses of the original value with the phi node
        auto* i32_ty = llvm::Type::getInt32Ty(kernel.getContext());

        for (auto& bb : kernel) {
            llvm::IRBuilder<> builder_locals(&bb.front());

            if (bb.isEntryBlock()) {
                idx_map[&bb].first->replaceAllUsesWith(
                    builder_locals.getInt32(0));
                continue;
            }

            // TODO check NumReservedValues
            auto* phi = builder_locals.CreatePHI(i32_ty, 0);

            for (const auto& pred : llvm::predecessors(&bb)) {
                phi->addIncoming(idx_map[pred].second, pred);
            }

            // Replace the dummy value created in initTracingIndices by the Phi
            // node
            idx_map[&bb].first->replaceAllUsesWith(phi);
            dyn_cast<llvm::Instruction>(idx_map[&bb].first)->eraseFromParent();
            idx_map[&bb].first = phi;
        }
    }

    llvm::Value* getCounterAndIncrement(llvm::Module& mod,
                                        llvm::IRBuilder<>& builder,
                                        llvm::Value* counter) override final {
        if (counter->getType()->isPointerTy()) {
            throw std::runtime_error("Cannot store wave counters in an "
                                     "alloca-ted value (or is it in LDS?)");
        }
        // Increment the value, but with inline asm
        auto* int32_ty = builder.getInt32Ty();

        auto* f_ty = llvm::FunctionType::get(int32_ty, {int32_ty}, false);

        auto* inc_sgpr = llvm::InlineAsm::get(f_ty, sgpr_inline_asm,
                                              sgpr_asm_constraints, true);

        return builder.CreateCall(inc_sgpr, {counter});
    }

    llvm::Value* traceIdxAtBlock(llvm::BasicBlock& bb) override {

        const auto [pre_inc, post_inc] = idx_map.at(&bb);

        llvm::IRBuilder<> builder(dyn_cast<llvm::Instruction>(pre_inc));

        auto* int32_ty = builder.getInt32Ty();
        auto* f_ty = llvm::FunctionType::get(int32_ty, {int32_ty}, false);

        auto* copy_to_vgpr = llvm::InlineAsm::get(f_ty, vgpr_inline_asm,
                                                  vgpr_asm_constraints, true);

        auto* copy = builder.CreateCall(copy_to_vgpr, {pre_inc});

        builder.CreateStore(copy, alloca);

        llvm::dbgs() << bb;
        return alloca;
    }

  private:
    // Inline asm to increase a (supposedly) sgpr
    static constexpr auto* sgpr_inline_asm = "s_add_u32 $0, $0, 1";
    static constexpr auto* sgpr_asm_constraints = "={s0},{s0}";

    // Inline asm to copy a sgpr (the counter) to a vgpr
    static constexpr auto* vgpr_inline_asm = "v_mov_b32_e32 $0, $1";
    static constexpr auto* vgpr_asm_constraints = "=v,{s0}";

    // Readfirstlane to extract a vgpr value to a sgpr
    static constexpr auto* readfirstlane_asm = "v_readfirstlane $0, $1";
    static constexpr auto* readfirstlane_asm_constraints = "=s, v";

    std::map<llvm::BasicBlock*, std::pair<llvm::Value*, llvm::Value*>> idx_map;
    llvm::Value* alloca = nullptr;
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
        return llvm::StructType::create(context, {i64, i64, i64, i32},
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
