/** \file ir_codegen.cpp
 * \brief LLVM IR instrumentation code generation
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Debug.h"

#include "ir_codegen.h"
#include "llvm_instr_counters.h"

namespace hip {

BasicBlock InstrumentedBlock::toBasicBlock() const {
    return BasicBlock(id, id, flops, "", "", ld_bytes, st_bytes);
}

bool isDeviceModule(const llvm::Module& mod) {
    auto triple = mod.getTargetTriple();
    // TODO : Handle non-AMD devices
    return triple == "amdgcn-amd-amdhsa";
}

int64_t valueToInt(llvm::Value* v) {
    if (auto* constant = dyn_cast<llvm::ConstantInt>(v)) {
        return constant->getZExtValue();
    } else {
        return 0;
    }
}

llvm::BasicBlock::iterator
findInstruction(llvm::BasicBlock& bb,
                std::function<bool(const llvm::Instruction*)> predicate) {
    for (auto it = bb.begin(); it != bb.end(); ++it) {
        if (predicate(&(*it))) {
            return it;
        }
    }

    return {};
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

llvm::Value*
recursiveGetUsePredicate(llvm::Value* v,
                         std::function<bool(const llvm::Value*)> predicate) {
    if (predicate(v)) {
        return v;
    }

    for (auto* use : v->users()) {
        auto* maybe_v = recursiveGetUsePredicate(use, predicate);
        if (maybe_v) {
            return maybe_v;
        }
    }

    return nullptr;
}

bool hasUse(const llvm::Value* v,
            std::function<bool(const llvm::Value*)> predicate) {
    for (auto* use : v->users()) {
        if (predicate(use)) {
            return true;
        }
    }

    return false;
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

bool hasFunctionCall(const llvm::Instruction& instr,
                     const std::string& function) {
    if (auto* call_inst = dyn_cast<llvm::CallInst>(&instr)) {
        auto* callee = call_inst->getCalledFunction();
        if (callee == nullptr || !callee->hasName()) {
            return false;
        } else {
            return callee->getName() == function;
        }
    } else {
        return false;
    }
}

bool hasFunctionCall(llvm::Function& f, const std::string& function) {
    auto predicate = [&function](auto& instr) {
        return hasFunctionCall(instr, function);
    };

    for (auto& bb : f) {
        for (auto it = bb.begin(); it != bb.end(); ++it) {
            if (predicate(*it)) {
                return true;
            }
        }
    }

    return false;
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

uint64_t getArraySize(const llvm::AllocaInst* alloca) {
    auto* type = alloca->getAllocatedType();
    if (alloca->isArrayAllocation()) {
        return dyn_cast<llvm::ConstantInt>(alloca->getArraySize())
            ->getZExtValue();
    } else if (type->isArrayTy()) {
        return type->getArrayNumElements();
    } else {
        // Apparently not an array
        return 1;
    }
}

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

llvm::Function* getFunction(llvm::Module& mod, llvm::StringRef name,
                            llvm::FunctionType* type) {
    auto callee = mod.getOrInsertFunction(name, type);
    auto* fun = dyn_cast<llvm::Function>(&*callee.getCallee());

    if (fun == nullptr) {
        throw std::runtime_error(
            ("Could not get function \"" + name + "\"").str());
    }

    return fun;
}

InstrumentationFunctions declareInstrumentation(llvm::Module& mod) {
    InstrumentationFunctions funs;
    auto& context = mod.getContext();

    auto void_type = llvm::Type::getVoidTy(context);
    auto uint8_type = llvm::Type::getInt8Ty(context);
    auto uint8_ptr_type = uint8_type->getPointerTo();
    auto uint64_type = llvm::Type::getInt64Ty(context);
    auto unqual_ptr_type = llvm::PointerType::getUnqual(context);

    auto _hip_store_ctr_type = llvm::FunctionType::get(
        void_type, {uint8_ptr_type, uint64_type, uint8_ptr_type}, false);

    funs._hip_store_ctr =
        getFunction(mod, "_hip_store_ctr", _hip_store_ctr_type);

    auto void_from_ptr_type =
        llvm::FunctionType::get(void_type, {unqual_ptr_type}, false);
    auto recoverer_ctor_type =
        llvm::FunctionType::get(unqual_ptr_type, {}, false);
    auto ptr_from_ptr_type =
        llvm::FunctionType::get(unqual_ptr_type, {unqual_ptr_type}, false);

    // This is tedious, but now way around it

    funs.freeHipInstrumenter =
        getFunction(mod, "freeHipInstrumenter", void_from_ptr_type);

    funs.freeHipStateRecoverer =
        getFunction(mod, "freeHipStateRecoverer", void_from_ptr_type);

    funs.hipNewInstrumenter =
        getFunction(mod, "hipNewInstrumenter", ptr_from_ptr_type);

    funs.hipNewStateRecoverer =
        getFunction(mod, "hipNewStateRecoverer", recoverer_ctor_type);

    funs.hipInstrumenterToDevice =
        getFunction(mod, "hipInstrumenterToDevice", ptr_from_ptr_type);

    auto from_device_type = llvm::FunctionType::get(
        void_type, {unqual_ptr_type, unqual_ptr_type}, false);
    funs.hipInstrumenterFromDevice =
        getFunction(mod, "hipInstrumenterFromDevice", from_device_type);

    funs.hipInstrumenterRecord =
        getFunction(mod, "hipInstrumenterRecord", void_from_ptr_type);

    funs.hipStateRecovererRegisterPointer =
        getFunction(mod, "hipStateRecovererRegisterPointer", from_device_type);

    funs.hipStateRecovererRollback =
        getFunction(mod, "hipStateRecovererRollback", void_from_ptr_type);

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

llvm::Function& cloneWithPrefix(llvm::Function& f, const std::string& prefix,
                                llvm::ArrayRef<llvm::Type*> extra_args) {

    return cloneWithName(f, getClonedName(f, prefix), extra_args);
}

void pushAdditionalArguments(llvm::Function& f,
                             llvm::ArrayRef<llvm::Value*> kernel_args) {
    llvm::dbgs() << f;
    auto push_call = firstInstructionOf<llvm::CallInst>(f);
    --push_call;

    auto* i8_ptr = llvm::Type::getInt8PtrTy(f.getContext());

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
    auto alloca_array = findInstruction(f, [](const auto* inst) -> bool {
        if (auto* alloca_inst = dyn_cast<llvm::AllocaInst>(inst)) {
            return alloca_inst->getAllocatedType()->isArrayTy() ||
                   hasUse(inst, [alloca_inst](const auto* v) -> bool {
                       if (auto* call_inst = dyn_cast<llvm::CallInst>(v)) {
                           return hasFunctionCall(*call_inst,
                                                  "hipLaunchKernel") &&
                                  call_inst->getArgOperand(5) == alloca_inst;
                       }
                       return false;
                   });
        } else {
            return false;
        }
    });

    if (alloca_array.getNodePtr() == nullptr) {
        throw std::runtime_error("Could not find allocated args array");
    }

    auto* alloca_array_inst = dyn_cast<llvm::AllocaInst>(&(*alloca_array));

    builder.SetInsertPoint(alloca_array_inst);
    auto array_size = getArraySize(alloca_array_inst);

    auto* new_array_type =
        llvm::ArrayType::get(i8_ptr, array_size + kernel_args.size());
    auto* new_alloca_array = builder.CreateAlloca(new_array_type);
    new_alloca_array->setAlignment(alloca_array_inst->getAlign());

    // Alloca + replaceInstWithInst

    auto* const_zero = getIndex(0u, f.getContext());

    std::vector<llvm::Instruction*> to_remove;

    for (auto* use : alloca_array_inst->users()) {
        if (auto* gep = dyn_cast<llvm::GetElementPtrInst>(use)) {
            // Replace with new GEP

            builder.SetInsertPoint(gep);

            auto it = gep->idx_end() - 1;
            auto* value = dyn_cast<llvm::Value>(&(*it));

            auto* new_gep = builder.CreateInBoundsGEP(
                new_array_type, new_alloca_array, {const_zero, value});

            gep->replaceAllUsesWith(new_gep);
            to_remove.push_back(gep);
        } else if (auto* bitcast = dyn_cast<llvm::BitCastInst>(use)) {
            // Actually a bug (?) in the front-end, references the first element
            // of the array. Replace by a GEP
            builder.SetInsertPoint(bitcast);

            auto* new_bitcast =
                builder.CreateBitCast(new_alloca_array, bitcast->getDestTy());

            bitcast->replaceAllUsesWith(new_bitcast);
            to_remove.push_back(bitcast);
        }
    }

    alloca_array_inst->replaceAllUsesWith(
        builder.CreateBitCast(new_alloca_array, alloca_array_inst->getType()));

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

        // auto* bitcast = builder.CreateBitCast(gep, i8_ptr->getPointerTo());
        auto* stack_bitcast = builder.CreateBitCast(new_arg, i8_ptr);

        builder.CreateStore(stack_bitcast, gep);

        ++i;
    }
}

void assertModuleIntegrity(llvm::Module& mod) {
    std::string err;
    llvm::raw_string_ostream os(err);
    if (llvm::verifyModule(mod, &os)) {
        llvm::dbgs() << "##### FULL MODULE #####\n\n" << mod << "\n#####\n";
        llvm::dbgs() << "Error : \n" << err << "\n#####\n";
        throw std::runtime_error("Broken module!");
    }
}

} // namespace hip
