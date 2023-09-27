/** \file ir_codegen.cpp
 * \brief LLVM IR instrumentation code generation
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include <string>

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
            if (auto* call_inst = llvm::dyn_cast<llvm::CallInst>(instr)) {
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
        while (InsertPt != End &&
               (llvm::isa<llvm::AllocaInst>(*InsertPt) ||
                llvm::isa<llvm::DbgInfoIntrinsic>(*InsertPt) ||
                llvm::isa<llvm::PseudoProbeInst>(*InsertPt))) {
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

bool isBlockInstrumentable(const llvm::BasicBlock& block) {
    if (block.isEntryBlock() ||
        llvm::isa<llvm::ReturnInst>(block.getTerminator())) {
        return true;
    }

    for (const auto& parent : llvm::predecessors(&block)) {
        auto* term = parent->getTerminator();
        if (llvm::isa<llvm::BranchInst>(term) ||
            llvm::isa<llvm::SwitchInst>(term)) {
            return true;
        }
    }

    return false;
}

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

llvm::Value* readFirstLaneI64(llvm::IRBuilder<>& builder,
                              llvm::Value* i64_vgpr) {
    auto* ptr_ty = builder.getPtrTy();
    auto* i32_ty = builder.getInt32Ty();
    auto* i64_ty = builder.getInt64Ty();

    auto* double_i32_ty =
        llvm::VectorType::get(i32_ty, llvm::ElementCount::getFixed(2));

    auto* f_ty = llvm::FunctionType::get(i32_ty, {i32_ty}, false);

    auto* readfirstlane =
        llvm::InlineAsm::get(f_ty, "v_readfirstlane_b32 $0, $1", "=s,v", true);

    // Convert to i64 if the value is a pointer
    bool ptr = i64_vgpr->getType()->isPointerTy();
    if (ptr) {
        i64_vgpr = builder.CreatePtrToInt(i64_vgpr, i64_ty);
    }

    auto* vgpr_pair = builder.CreateBitCast(i64_vgpr, double_i32_ty);

    auto* lsb_vgpr = builder.CreateExtractElement(vgpr_pair, 0ul);
    auto* msb_vgpr = builder.CreateExtractElement(vgpr_pair, 1ul);

    auto* lsb_sgpr = builder.CreateCall(readfirstlane, lsb_vgpr);
    auto* msb_sgpr = builder.CreateCall(readfirstlane, msb_vgpr);

    auto* sgpr_pair = builder.CreateInsertElement(
        llvm::UndefValue::get(double_i32_ty), lsb_sgpr, 0ul);
    sgpr_pair = builder.CreateInsertElement(sgpr_pair, msb_sgpr, 1ul);

    auto* sgpr = builder.CreateBitCast(sgpr_pair, i64_ty);
    if (ptr) {
        return builder.CreateIntToPtr(sgpr, ptr_ty);
    } else {
        return sgpr;
    }
}

llvm::Value* readFirstLaneI64(llvm::IRBuilder<>& builder, llvm::Value* i64_vgpr,
                              uint8_t reg) {
    auto* ptr_ty = builder.getPtrTy();
    auto* i32_ty = builder.getInt32Ty();
    auto* i64_ty = builder.getInt64Ty();

    auto reg_num = static_cast<unsigned int>(reg);
    auto reg_lsb = llvm::Twine('s').concat(std::to_string(reg_num)).str();
    auto reg_msb = llvm::Twine('s').concat(std::to_string(reg_num + 1)).str();

    auto* double_i32_ty =
        llvm::VectorType::get(i32_ty, llvm::ElementCount::getFixed(2));

    auto* f_ty = llvm::FunctionType::get(i32_ty, {i32_ty}, false);

    auto instr = llvm::Twine("v_readfirstlane_b32 $0, $1").str();
    auto constraints_lsb =
        llvm::Twine("={").concat(reg_lsb).concat("},v").str();
    auto constraints_msb =
        llvm::Twine("={").concat(reg_msb).concat("},v").str();

    auto* readfirstlane_lsb =
        llvm::InlineAsm::get(f_ty, instr, constraints_lsb, true);
    auto* readfirstlane_msb =
        llvm::InlineAsm::get(f_ty, instr, constraints_msb, true);

    // Convert to i64 if the value is a pointer
    bool ptr = i64_vgpr->getType()->isPointerTy();
    if (ptr) {
        i64_vgpr = builder.CreatePtrToInt(i64_vgpr, i64_ty);
    }

    auto* vgpr_pair = builder.CreateBitCast(i64_vgpr, double_i32_ty);

    auto* lsb_vgpr = builder.CreateExtractElement(vgpr_pair, 0ul);
    auto* msb_vgpr = builder.CreateExtractElement(vgpr_pair, 1ul);

    auto* lsb_sgpr = builder.CreateCall(readfirstlane_lsb, lsb_vgpr);
    auto* msb_sgpr = builder.CreateCall(readfirstlane_msb, msb_vgpr);

    auto* sgpr_pair = builder.CreateInsertElement(
        llvm::UndefValue::get(double_i32_ty), lsb_sgpr, 0ul);
    sgpr_pair = builder.CreateInsertElement(sgpr_pair, msb_sgpr, 1ul);

    auto* sgpr = builder.CreateBitCast(sgpr_pair, i64_ty);
    if (ptr) {
        return builder.CreateIntToPtr(sgpr, ptr_ty);
    } else {
        return sgpr;
    }
}

llvm::Value* initializeSGPR(llvm::IRBuilder<>& builder, uint32_t initializer,
                            std::string_view reg) {
    auto* i32_ty = builder.getInt32Ty();
    auto init_val = builder.getInt32(initializer);

    auto* f_ty = llvm::FunctionType::get(builder.getVoidTy(), {i32_ty}, false);

    auto instr = llvm::Twine("s_mov_b32 ").concat(reg).concat(", $0").str();
    auto constraints = "i";

    auto* mov_b32 = llvm::InlineAsm::get(f_ty, instr, constraints, true);

    return builder.CreateCall(mov_b32, {init_val});
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

llvm::InlineAsm* incrementRegisterAsm(llvm::IRBuilder<>& builder,
                                      std::string_view reg, bool carry,
                                      std::string_view inc) {
    const char* opcode = carry ? "s_addc_u32 " : "s_add_u32 ";
    auto instr = llvm::Twine(opcode)
                     .concat(reg)
                     .concat(", ")
                     .concat(reg)
                     .concat(", ")
                     .concat(inc)
                     .str();
    auto constraints = "~{scc}";

    auto* void_ty = builder.getVoidTy();
    auto* f_ty = llvm::FunctionType::get(void_ty, {}, false);
    return llvm::InlineAsm::get(f_ty, instr, constraints, true);
}

InstrumentationFunctions::InstrumentationFunctions(llvm::Module& mod) {
    auto& context = mod.getContext();

    auto* void_type = llvm::Type::getVoidTy(context);
    auto* uint32_type = llvm::Type::getInt32Ty(context);
    auto* ptr_type = llvm::PointerType::getUnqual(context);

    auto void_from_ptr_type =
        llvm::FunctionType::get(void_type, {ptr_type}, false);
    auto recoverer_ctor_type = llvm::FunctionType::get(ptr_type, {}, false);
    auto ptr_from_ptr_type =
        llvm::FunctionType::get(ptr_type, {ptr_type}, false);

    // This is tedious, but now way around it

    freeHipInstrumenter =
        getFunction(mod, "freeHipInstrumenter", void_from_ptr_type);

    freeHipStateRecoverer =
        getFunction(mod, "freeHipStateRecoverer", void_from_ptr_type);

    hipNewInstrumenter = getFunction(
        mod, "hipNewInstrumenter",
        llvm::FunctionType::get(ptr_type, {ptr_type, uint32_type}, false));

    hipNewStateRecoverer =
        getFunction(mod, "hipNewStateRecoverer", recoverer_ctor_type);

    hipGetNextInstrumenter =
        getFunction(mod, "hipGetNextInstrumenter",
                    llvm::FunctionType::get(ptr_type, {}, false));

    hipInstrumenterToDevice =
        getFunction(mod, "hipInstrumenterToDevice", ptr_from_ptr_type);

    auto from_device_type =
        llvm::FunctionType::get(void_type, {ptr_type, ptr_type}, false);
    hipInstrumenterFromDevice =
        getFunction(mod, "hipInstrumenterFromDevice", from_device_type);

    hipInstrumenterRecord =
        getFunction(mod, "hipInstrumenterRecord", void_from_ptr_type);

    hipStateRecovererRegisterPointer = getFunction(
        mod, "hipStateRecovererRegisterPointer",
        llvm::FunctionType::get(ptr_type, {ptr_type, ptr_type}, false));

    hipStateRecovererRollback = getFunction(
        mod, "hipStateRecovererRollback",
        llvm::FunctionType::get(void_type, {ptr_type, ptr_type}, false));

    newHipQueueInfo =
        getFunction(mod, "newHipQueueInfo",
                    llvm::FunctionType::get(
                        ptr_type, {ptr_type, uint32_type, uint32_type}, false));

    hipQueueInfoAllocBuffer =
        getFunction(mod, "hipQueueInfoAllocBuffer", ptr_from_ptr_type);

    hipQueueInfoAllocOffsets =
        getFunction(mod, "hipQueueInfoAllocOffsets", ptr_from_ptr_type);

    hipQueueInfoRecord =
        getFunction(mod, "hipQueueInfoRecord",
                    llvm::FunctionType::get(
                        void_type, {ptr_type, ptr_type, ptr_type}, false));

    freeHipQueueInfo = getFunction(mod, "freeHipQueueInfo", void_from_ptr_type);
}

CfgFunctions::CfgFunctions(llvm::Module& mod) {
    auto& context = mod.getContext();

    auto* void_type = llvm::Type::getVoidTy(context);
    auto* ptr_ty = llvm::PointerType::getUnqual(context);
    auto* uint64_type = llvm::Type::getInt64Ty(context);
    auto* uint32_type = llvm::Type::getInt32Ty(context);

    auto* _hip_store_ctr_type = llvm::FunctionType::get(
        void_type, {ptr_ty, uint64_type, ptr_ty}, false);

    _hip_store_ctr = getFunction(mod, "_hip_store_ctr", _hip_store_ctr_type);
    _hip_wave_ctr_get_offset = getFunction(
        mod, "_hip_wave_ctr_get_offset",
        llvm::FunctionType::get(ptr_ty, {ptr_ty, uint32_type}, false));
}

llvm::FunctionType* getEventCtorType(llvm::LLVMContext& context) {
    auto* void_type = llvm::Type::getVoidTy(context);
    auto* ptr_type = llvm::PointerType::getUnqual(context);
    auto* uint64_type = llvm::Type::getInt64Ty(context);
    return llvm::FunctionType::get(void_type, {ptr_type, uint64_type}, false);
}

TracingFunctions::TracingFunctions(llvm::Module& mod) {
    auto& context = mod.getContext();

    auto* void_type = llvm::Type::getVoidTy(context);
    auto* ptr_type = llvm::PointerType::getUnqual(context);
    auto* uint64_type = llvm::Type::getInt64Ty(context);
    auto* uint32_type = llvm::Type::getInt32Ty(context);

    // Unqual ptr hides function pointers?
    // auto* _hip_event_ctor_type = getEventCtorType(context);

    auto* offset_getter_type = llvm::FunctionType::get(
        ptr_type, {ptr_type, ptr_type, uint64_type}, false);

    auto* event_creator_type = llvm::FunctionType::get(
        void_type, {ptr_type, ptr_type, uint64_type, ptr_type, uint32_type},
        false);

    _hip_get_trace_offset =
        getFunction(mod, "_hip_get_trace_offset", offset_getter_type);

    _hip_get_wave_trace_offset =
        getFunction(mod, "_hip_get_wave_trace_offset", offset_getter_type);

    _hip_create_event =
        getFunction(mod, "_hip_create_event", event_creator_type);

    _hip_create_wave_event =
        getFunction(mod, "_hip_create_wave_event", event_creator_type);
}

llvm::Function& cloneWithName(llvm::Function& f, std::string_view name,
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

    if (llvm::isa<llvm::Function>(callee.getCallee())) {
        return *dyn_cast<llvm::Function>(&*callee.getCallee());
    } else {
        throw std::runtime_error("Could not clone function");
    }
}

void optimizeFunction(llvm::Function& f, llvm::FunctionAnalysisManager& fm) {
    llvm::SimplifyCFGPass().run(f, fm);
}

llvm::Function& cloneWithPrefix(llvm::Function& f, std::string_view prefix,
                                llvm::ArrayRef<llvm::Type*> extra_args) {

    return cloneWithName(f, getClonedName(f, prefix), extra_args);
}

void pushAdditionalArguments(llvm::Function& f,
                             llvm::ArrayRef<llvm::Value*> kernel_args) {
    auto push_call = firstInstructionOf<llvm::CallInst>(f);
    --push_call;

    auto* ptr_ty = llvm::PointerType::getUnqual(f.getContext());

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
        if (auto* alloca_inst = llvm::dyn_cast<llvm::AllocaInst>(inst)) {
            return hasUse(inst, [alloca_inst](const auto* v) -> bool {
                if (auto* call_inst = llvm::dyn_cast<llvm::CallInst>(v)) {
                    return hasFunctionCall(*call_inst, "hipLaunchKernel") &&
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

    auto* alloca_array_inst =
        llvm::dyn_cast<llvm::AllocaInst>(&(*alloca_array));

    builder.SetInsertPoint(alloca_array_inst);
    auto array_size = getArraySize(alloca_array_inst);

    auto* new_alloca_array = builder.CreateAlloca(
        ptr_ty, builder.getInt64(array_size + kernel_args.size()),
        "new_kernel_args");
    new_alloca_array->setAlignment(alloca_array_inst->getAlign());

    // Alloca + replaceInstWithInst

    std::vector<llvm::Instruction*> to_remove;

    for (auto* use : alloca_array_inst->users()) {
        if (auto* gep = dyn_cast<llvm::GetElementPtrInst>(use)) {
            // Replace with new GEP

            builder.SetInsertPoint(gep);

            auto it = gep->idx_end() - 1;
            auto* value = dyn_cast<llvm::Value>(&(*it));

            auto* new_gep =
                builder.CreateGEP(ptr_ty, new_alloca_array, {value});

            gep->replaceAllUsesWith(new_gep);
            to_remove.push_back(gep);
        }
    }

    alloca_array_inst->replaceAllUsesWith(new_alloca_array);

    // Delete old uses

    for (auto instr : to_remove) {
        instr->eraseFromParent();
    }

    alloca_array_inst->eraseFromParent();

    // Insert new args

    auto i = array_size; // Insert at end
    builder.SetInsertPoint(new_alloca_array->getNextNode());

    for (auto new_arg : new_args) {
        auto* gep =
            builder.CreateGEP(ptr_ty, new_alloca_array, {builder.getInt32(i)});

        builder.CreateStore(new_arg, gep);

        ++i;
    }

    new_alloca_array->setName("kernel_args");
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

void dumpMetadata(llvm::Function* f) {
    llvm::SmallVector<std::pair<unsigned, llvm::MDNode*>> vec;
    f->getAllMetadata(vec);
    for (auto& [id, node] : vec) {
        if (node != nullptr) {
            llvm::dbgs() << "DBG " << id << '\n';
            node->printTree(llvm::dbgs());
            llvm::dbgs() << '\n';
        }
    }
}

void dumpMetadata(llvm::Instruction* i) {
    llvm::SmallVector<std::pair<unsigned, llvm::MDNode*>> vec;
    i->getAllMetadata(vec);
    for (auto& [id, node] : vec) {
        if (node != nullptr) {
            llvm::dbgs() << "DBG " << id << '\n';
            node->printTree(llvm::dbgs());
            llvm::dbgs() << '\n';
        }
    }
}

llvm::DISubroutineType* getSubroutineType(llvm::Function* f) {
    auto* di = f->getSubprogram();
    return di != nullptr
               ? dyn_cast<llvm::DISubroutineType>(di->getOperand(4).get())
               : nullptr;
}

} // namespace hip
