/** \file ir_codegen.cpp
 * \brief LLVM IR instrumentation code generation
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"

#include "ir_codegen.h"
#include "llvm_instr_counters.h"

namespace hip {

int64_t valueToInt(llvm::Value* v) {
    if (auto* constant = dyn_cast<llvm::ConstantInt>(v)) {
        return constant->getZExtValue();
    } else {
        return 0;
    }
}

llvm::BasicBlock::iterator
findInstruction(llvm::Function& f,
                std::function<bool(const llvm::Instruction*)> predicate) {
    for (auto& bb : f) {
        for (auto it = bb.begin(); it != bb.end(); ++it) {
            if (predicate(&(*it))) {
                return it;
            }
        }
    }

    return {};
}

llvm::CallInst* firstCallToFunction(llvm::Function& f,
                                    const std::string& function) {
    return dyn_cast<llvm::CallInst>(
        &(*findInstruction(f, [&function](auto* instr) {
            if (auto* call_inst = dyn_cast<llvm::CallInst>(instr)) {
                return call_inst->getCalledFunction()->getName() == function;
            } else {
                return false;
            }
        })));
}

llvm::Value* getIndex(uint64_t idx, llvm::LLVMContext& context) {
    return llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), idx);
}

llvm::BasicBlock::iterator getFirstNonPHIOrDbgOrAlloca(llvm::BasicBlock& bb) {
    llvm::Instruction* FirstNonPHI = bb.getFirstNonPHI();
    if (!FirstNonPHI)
        return bb.end();

    llvm::BasicBlock::iterator InsertPt = FirstNonPHI->getIterator();
    if (InsertPt->isEHPad())
        ++InsertPt;

    if (bb.isEntryBlock()) {
        llvm::BasicBlock::iterator End = bb.end();
        while (InsertPt != End && (isa<llvm::AllocaInst>(*InsertPt) ||
                                   isa<llvm::DbgInfoIntrinsic>(*InsertPt) ||
                                   isa<llvm::PseudoProbeInst>(*InsertPt))) {
            if (const llvm::AllocaInst* AI =
                    dyn_cast<llvm::AllocaInst>(&*InsertPt)) {
                if (!AI->isStaticAlloca())
                    break;
            }
            ++InsertPt;
        }
    }
    return InsertPt;
}

void setInsertPointPastAllocas(llvm::IRBuilderBase& builder,
                               llvm::Function& f) {
    auto& bb = f.getEntryBlock();
    builder.SetInsertPoint(bb.getFirstNonPHIOrDbg());
}

bool isBlockInstrumentable(const llvm::BasicBlock& block) { return true; }

void insertInstrumentationLocals(InstrumentationContext& context) {}

void insertCounter(llvm::BasicBlock& bb) {}

InstrumentedBlock getBlockInfo(const llvm::BasicBlock& bb, unsigned int i) {
    FlopCounter flop_counter;
    auto flops = flop_counter(bb);

    llvm::errs() << '\n' << bb.getName() << " (" << i << ")\n";
    llvm::errs() << "Found " << flops << " flops\n";

    hip::StoreCounter store_counter;
    auto f_st = store_counter.count(bb, MemType::MemType::Floating);
    llvm::errs() << "Found " << f_st << " stores \n";

    hip::LoadCounter load_counter;
    auto f_ld = load_counter.count(bb, MemType::MemType::Floating);
    llvm::errs() << "Found " << f_ld << " loads \n";

    return {i, flops, f_ld, f_st, {}};
}

InstrumentationFunctions declareInstrumentation(llvm::Module& mod) {
    InstrumentationFunctions funs;
    auto& context = mod.getContext();

    auto void_type = llvm::Type::getVoidTy(context);
    auto uint8_type = llvm::Type::getInt8Ty(context);
    auto uint8_ptr_type = uint8_type->getPointerTo();
    auto uint64_type = llvm::Type::getInt64Ty(context);

    auto _hip_store_ctr_type = llvm::FunctionType::get(
        void_type, {uint8_ptr_type, uint64_type, uint8_ptr_type}, false);
    auto callee =
        mod.getOrInsertFunction("_hip_store_ctr", _hip_store_ctr_type);

    if (isa<llvm::Function>(callee.getCallee())) {
        funs._hip_store_ctr = dyn_cast<llvm::Function>(&*callee.getCallee());
    } else {
        throw std::runtime_error("Could not get function \"_hip_store_ctr\"");
    }

    return funs;
}

llvm::Function& cloneWithName(llvm::Function& f, const std::string& name,
                              llvm::ArrayRef<llvm::Type*> extra_args) {
    auto& mod = *f.getParent();
    auto fun_type = f.getFunctionType();
    auto base_args = fun_type->params();

    auto new_args =
        llvm::SmallVector<llvm::Type*>(base_args.begin(), base_args.end());

    new_args.append(extra_args.begin(), extra_args.end());

    auto new_fun_type =
        llvm::FunctionType::get(fun_type->getReturnType(), new_args, false);

    auto callee = mod.getOrInsertFunction(name, new_fun_type);

    if (isa<llvm::Function>(callee.getCallee())) {
        return *dyn_cast<llvm::Function>(&*callee.getCallee());
    } else {
        throw std::runtime_error("Could not clone function");
    }
}

llvm::Function& cloneWithSuffix(llvm::Function& f, const std::string& suffix,
                                llvm::ArrayRef<llvm::Type*> extra_args) {
    auto name = f.getName() + suffix + cloned_suffix;

    return cloneWithName(f, name.str(), extra_args);
}

void pushAdditionalArguments(llvm::Function& f,
                             llvm::ArrayRef<llvm::Value*> kernel_args) {
    auto push_call = firstInstructionOf<llvm::CallInst>(f);
    --push_call;

    // Allocate memory for additional args
    llvm::IRBuilder<> builder(&f.getEntryBlock());

    setInsertPointPastAllocas(builder, f);

    std::vector<llvm::Value*> new_args;

    // Allocate new arguments & copy to stack
    for (auto new_arg : kernel_args) {
        auto* var_type = new_arg->getType();

        auto* local_store = builder.CreateAlloca(var_type);
        new_args.push_back(local_store);

        builder.CreateStore(new_arg, local_store);
    }

    // Replace the void* array with additional size, modify types for each
    auto alloca_array = findInstruction(f, [](auto* inst) {
        if (auto* alloca_inst = dyn_cast<llvm::AllocaInst>(inst)) {
            return alloca_inst->getAllocatedType()->isArrayTy();
        } else {
            return false;
        }
    });

    auto* alloca_array_inst = dyn_cast<llvm::AllocaInst>(&(*alloca_array));

    builder.SetInsertPoint(alloca_array_inst);
    auto array_size =
        alloca_array_inst->getAllocatedType()->getArrayNumElements();

    auto* new_array_type =
        llvm::ArrayType::get(llvm::Type::getInt8PtrTy(f.getContext()),
                             array_size + kernel_args.size());
    auto* new_alloca_array = builder.CreateAlloca(new_array_type);

    // Alloca + replaceInstWithInst

    auto* const_zero = getIndex(0u, f.getContext());

    std::vector<llvm::Instruction*> to_remove;

    for (auto* use : alloca_array_inst->users()) {
        if (auto* gep = dyn_cast<llvm::GetElementPtrInst>(use)) {
            // Replace with new GEP

            builder.SetInsertPoint(gep);

            auto it = gep->idx_begin() + 1;
            auto* value = dyn_cast<llvm::Value>(&(*it));

            auto* new_gep = builder.CreateInBoundsGEP(
                new_array_type, new_alloca_array, {const_zero, value});

            gep->replaceAllUsesWith(new_gep);
            to_remove.push_back(gep);
        } else if (auto* bitcast = dyn_cast<llvm::BitCastInst>(use)) {
            // Actually a bug in the front-end, references the first element
            // of the array. Replace by a GEP
            builder.SetInsertPoint(bitcast);

            auto* new_bitcast =
                builder.CreateBitCast(new_alloca_array, bitcast->getDestTy());

            bitcast->replaceAllUsesWith(new_bitcast);
            to_remove.push_back(bitcast);
        }
    }

    // Delete old uses

    for (auto instr : to_remove) {
        instr->eraseFromParent();
    }

    alloca_array_inst->eraseFromParent();

    // Insert new args

    auto i = array_size; // Insert at end
    builder.SetInsertPoint(&(*firstInstructionOf<llvm::GetElementPtrInst>(f)));

    for (auto new_arg : new_args) {
        auto* gep = builder.CreateInBoundsGEP(
            new_array_type, new_alloca_array,
            {const_zero, getIndex(i, f.getContext())});

        auto* bitcast =
            builder.CreateBitCast(gep, new_arg->getType()->getPointerTo());

        builder.CreateStore(new_arg, bitcast);

        ++i;
    }
}

} // namespace hip
