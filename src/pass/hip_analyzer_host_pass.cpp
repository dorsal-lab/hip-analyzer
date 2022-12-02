/** \file hip_analyzer_host_pass
 * \brief Instrumentation pass
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_analyzer_pass.h"

#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "ir_codegen.h"

namespace hip {

bool isDeviceStub(llvm::Function& f) {
    auto name = f.getName().str();

    return contains(name, device_stub_prefix) &&
           !contains(name, cloned_suffix +
                               CfgInstrumentationPass::instrumented_prefix) &&
           !contains(name, cloned_suffix + TracingPass::instrumented_prefix);
}

bool isKernelCallSite(llvm::Function& f) {
    // True if calls hipLaunchKernel and is not a stub (so an inlined
    // stub)
    return hasFunctionCall(f, "hipLaunchKernel") && !isDeviceStub(f) &&
           !contains(f.getName().str(),
                     cloned_suffix +
                         CfgInstrumentationPass::instrumented_prefix) &&
           !contains(f.getName().str(),
                     cloned_suffix + TracingPass::instrumented_prefix);
}

llvm::PreservedAnalyses HostPass::run(llvm::Module& mod,
                                      llvm::ModuleAnalysisManager& modm) {
    if (isDeviceModule(mod)) {
        // DO NOT run on device code
        return llvm::PreservedAnalyses::all();
    }

    bool modified = false;

    llvm::SmallVector<std::pair<llvm::Function*, llvm::Function*>, 8> to_delete;
    for (auto& f_original : mod.functions()) {
        if (isDeviceStub(f_original) && !f_original.isDeclaration()) {
            // Device stub

            // Duplicates the stub and calls the appropriate function
            auto* stub_counter = addCountersDeviceStub(f_original);
            llvm::dbgs() << *stub_counter;

            if (trace) {
                auto* stub_trace = addTracingDeviceStub(f_original);
                llvm::dbgs() << *stub_trace;
            }

            // Replaces all call to the original stub with tmp_<stub
            // name>, and add the old function to an erase list
            if (auto* new_stub = replaceStubCall(f_original)) {
                to_delete.push_back(std::make_pair(&f_original, new_stub));
            }

        } else if (isKernelCallSite(f_original)) {
            // Kernel calling site

            // Sadly the HIP Front-end inlines all kernel device stubs,
            // so we have to substitute the call ourselves

            addDeviceStubCall(f_original);
        }
    }

    for (auto& [old_stub, new_stub] : to_delete) {
        std::string name = old_stub->getName().str();

        llvm::dbgs() << "Removing stubs : " << name << '\n';

        // Just in case
        old_stub->replaceAllUsesWith(new_stub);

        old_stub->eraseFromParent();
        new_stub->setName(name);
    }

    assertModuleIntegrity(mod);

    return llvm::PreservedAnalyses::none();
    // TODO : link needed functions

    // TODO : remove `optnone` and `noinline` attributes, add
    // alwaysinline
}

llvm::Function*
HostPass::addCountersDeviceStub(llvm::Function& f_original) const {
    auto& context = f_original.getContext();

    return duplicateStubWithArgs(
        f_original, CfgInstrumentationPass::instrumented_prefix,
        {CfgInstrumentationPass::getCounterType(context)->getPointerTo()});
}

llvm::Function*
HostPass::addTracingDeviceStub(llvm::Function& f_original) const {
    auto& context = f_original.getContext();
    auto* i8_ptr = llvm::Type::getInt8PtrTy(context);

    return duplicateStubWithArgs(f_original, TracingPass::instrumented_prefix,
                                 {i8_ptr, i8_ptr});
}

llvm::Function*
HostPass::duplicateStubWithArgs(llvm::Function& f_original,
                                const std::string& prefix,
                                llvm::ArrayRef<llvm::Type*> new_args) const {
    auto& mod = *f_original.getParent();
    auto& context = mod.getContext();

    // Useful types
    auto* i8_ptr = llvm::Type::getInt8PtrTy(context);
    auto* i32_ptr = llvm::Type::getInt32PtrTy(context);

    auto& f = cloneWithPrefix(f_original, prefix, new_args);

    llvm::ValueToValueMapTy vmap;

    for (auto it1 = f_original.arg_begin(), it2 = f.arg_begin();
         it1 != f_original.arg_end(); ++it1, ++it2) {
        vmap[&*it1] = &*it2;
    }
    llvm::SmallVector<llvm::ReturnInst*, 8> returns;

    llvm::CloneFunctionInto(&f, &f_original, vmap,
                            llvm::CloneFunctionChangeType::LocalChangesOnly,
                            returns);

    llvm::SmallVector<llvm::Value*> new_vals;
    auto i = f_original.arg_size();

    for (auto* type : new_args) {
        if (type->isPointerTy()) {
            f.getArg(i)->addAttr(llvm::Attribute::NoUndef);
        }
        new_vals.push_back(f.getArg(i));
        ++i;
    }

    // Add additional arguments to the void* array : uint8_t* _counters;

    pushAdditionalArguments(f, new_vals);

    // Create global symbol

    auto* global_sym = createKernelSymbol(f_original, f, prefix);

    // Modify the call to hipLaunchKernel

    auto* call_to_launch = firstCallToFunction(f, "hipLaunchKernel");

    call_to_launch->setArgOperand(
        0, llvm::ConstantExpr::getPointerCast(global_sym, i8_ptr));

    // Modify __hip_module_ctor to register kernel

    auto* device_ctor = dyn_cast<llvm::Function>(
        mod.getOrInsertFunction(
               "__hip_module_ctor",
               llvm::FunctionType::get(llvm::Type::getVoidTy(context), {i8_ptr},
                                       false))
            .getCallee());

    llvm::CallInst* register_function;

    // The kernels are registered in the runtime by a function
    // `__hip_register_globals`, but it is marked alwaysinline and might be
    // absent from the module. Handle both cases here
    if (hasFunctionCall(*device_ctor, "__hip_register_globals")) {
        auto* globals_registerer =
            firstCallToFunction(*device_ctor, "__hip_register_globals")
                ->getCalledFunction();
        register_function =
            firstCallToFunction(*globals_registerer, "__hipRegisterFunction");
    } else {
        register_function =
            firstCallToFunction(*device_ctor, "__hipRegisterFunction");
    }

    llvm::IRBuilder<> builder(register_function);

    auto* name = builder.CreateGlobalStringPtr(global_sym->getName());

    builder.CreateCall(
        register_function->getCalledFunction(),
        {
            register_function->getArgOperand(0),
            llvm::ConstantExpr::getPointerCast(global_sym, i8_ptr),
            name,
            name,
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), -1, true),
            llvm::ConstantPointerNull::get(i8_ptr),
            llvm::ConstantPointerNull::get(i8_ptr),
            llvm::ConstantPointerNull::get(i8_ptr),
            llvm::ConstantPointerNull::get(i8_ptr),
            llvm::ConstantPointerNull::get(i32_ptr),
        });

    return &f;
}

llvm::Function* HostPass::replaceStubCall(llvm::Function& stub) const {
    auto& mod = *stub.getParent();

    auto fun_type = stub.getFunctionType();
    InstrumentationFunctions instr_handlers(mod);
    auto* call_to_launch = firstCallToFunction(stub, "hipLaunchKernel");
    auto* i8_ptr = llvm::Type::getInt8PtrTy(mod.getContext());

    auto* new_stub = dyn_cast<llvm::Function>(
        mod.getOrInsertFunction(
               getClonedName(stub, CfgInstrumentationPass::instrumented_prefix +
                                       "tmp_"),
               fun_type)
            .getCallee());

    auto* counters_stub = &cloneWithPrefix(
        stub, CfgInstrumentationPass::instrumented_prefix,
        {CfgInstrumentationPass::getCounterType(mod.getContext())
             ->getPointerTo()});

    auto* tracing_stub =
        trace ? &cloneWithPrefix(stub, TracingPass::instrumented_prefix,
                                 {i8_ptr, i8_ptr})
              : nullptr;

    auto* bb = llvm::BasicBlock::Create(mod.getContext(), "", new_stub);

    llvm::IRBuilder<> builder(bb);

    auto unqual_ptr_type = llvm::PointerType::getUnqual(mod.getContext());

    // Create instr object

    auto* instr = builder.CreateCall(
        instr_handlers.hipNewInstrumenter,
        {builder.CreateBitCast(
            builder.CreateGlobalStringPtr(
                dyn_cast<llvm::ConstantExpr>(call_to_launch->getArgOperand(0))
                    ->getOperand(0)
                    ->getName()),
            unqual_ptr_type)});

    auto* recoverer =
        builder.CreateCall(instr_handlers.hipNewStateRecoverer, {});

    // Create call to newly created stub
    llvm::SmallVector<llvm::Value*> args;
    for (llvm::Argument& arg : new_stub->args()) {
        // Save value if vector
        if (arg.getType()->isPointerTy()) {
            auto* bitcast = builder.CreateBitCast(&arg, unqual_ptr_type);
            builder.CreateCall(instr_handlers.hipStateRecovererRegisterPointer,
                               {recoverer, bitcast});
        }

        args.push_back(&arg);
    }

    auto* device_ptr =
        builder.CreateCall(instr_handlers.hipInstrumenterToDevice, {instr});

    args.push_back(builder.CreateBitCast(device_ptr, i8_ptr));

    builder.CreateCall(counters_stub->getFunctionType(), counters_stub, args);
    args.pop_back();

    llvm::Value *queue_info, *events_buffer, *events_offsets;

    if (trace) {
        // Launch right after the kernel so it executes in parallel
        auto* const_0 = llvm::ConstantInt::get(
            llvm::Type::getInt32Ty(mod.getContext()), 0ul);
        // 0, 0 -> thread<hip::Event>
        queue_info = builder.CreateCall(instr_handlers.newHipQueueInfo,
                                        {instr, const_0, const_0});
    }

    // Store counters (runtime)
    builder.CreateCall(instr_handlers.hipInstrumenterFromDevice,
                       {instr, device_ptr});

    if (trace) {
        builder.CreateCall(instr_handlers.hipStateRecovererRollback,
                           {recoverer, instr});

        events_buffer = builder.CreateCall(
            instr_handlers.hipQueueInfoAllocBuffer, {queue_info});
        events_offsets = builder.CreateCall(
            instr_handlers.hipQueueInfoAllocOffsets, {queue_info});

        // Launch tracing kernel

        args.push_back(builder.CreateBitCast(events_buffer, i8_ptr));
        args.push_back(builder.CreateBitCast(events_offsets, i8_ptr));
        builder.CreateCall(tracing_stub, args);
        args.pop_back_n(2);
    }

    builder.CreateCall(instr_handlers.hipInstrumenterRecord, {instr});

    if (trace) {
        // Store counters (runtime)
        builder.CreateCall(instr_handlers.hipQueueInfoFromDevice,
                           {queue_info, events_buffer});
        builder.CreateCall(instr_handlers.hipQueueInfoRecord, {queue_info});
    }

    // Free allocated instrumentation
    builder.CreateCall(instr_handlers.freeHipInstrumenter, {instr});
    builder.CreateCall(instr_handlers.freeHipStateRecoverer, {recoverer});

    builder.CreateRetVoid();

    // Replace all calls

    stub.replaceAllUsesWith(new_stub);

    llvm::dbgs() << *new_stub;

    return new_stub;
}

void HostPass::addDeviceStubCall(llvm::Function& f_original) const {
    auto* kernel_call = firstCallToFunction(f_original, "hipLaunchKernel");
    auto* inliner_bb = kernel_call->getParent();

    auto* gep_of_array =
        dyn_cast<llvm::GetElementPtrInst>(kernel_call->getArgOperand(5));

    // Get device stub

    // llvm::dbgs() << *kernel_call;
    auto* stub = getDeviceStub(dyn_cast<llvm::GlobalValue>(
        dyn_cast<llvm::ConstantExpr>(kernel_call->getArgOperand(0))
            ->getOperand(0)));

    if (stub == nullptr) {
        throw std::runtime_error("HostPass::addDeviceStubCall() : "
                                 "Could not find device stub");
    }

    auto* args_array = gep_of_array->getPointerOperand();

    // Build the value vector to pass to the new kernel stub

    llvm::SmallVector<llvm::Value*, 8> args{stub->arg_size(), nullptr};

    auto append_arg = [&args, &stub](size_t idx,
                                     llvm::AllocaInst* stack_storage) {
        for (auto* use_stack : stack_storage->users()) {
            if (auto* store_inst = dyn_cast<llvm::StoreInst>(use_stack)) {
                if (stack_storage == store_inst->getPointerOperand()) {
                    // llvm::dbgs() << *store_inst << '\n';

                    // Handle boolean values
                    auto* arg = stub->getArg(idx);
                    if (arg->getType()->isIntegerTy(1)) {
                        llvm::IRBuilder<> builder(store_inst);
                        auto* trunc = builder.CreateTrunc(
                            store_inst->getValueOperand(),
                            llvm::Type::getInt1Ty(arg->getContext()));
                        args[idx] = trunc;
                    } else {
                        args[idx] = store_inst->getValueOperand();
                    }
                }
            }
        }
    };

    // Forgive me
    for (auto* use : args_array->users()) {
        // llvm::dbgs() << *use << '\n';

        // Is it used directly in a bitcast ?
        if (auto* bitcast = dyn_cast<llvm::BitCastInst>(use)) {
            // llvm::dbgs() << "\t- > this is a bitcast\n";

            for (auto* use_bitcast : bitcast->users()) {

                // Is it used in a store ?
                if (auto* store = dyn_cast<llvm::StoreInst>(use_bitcast)) {
                    // Found arg
                    // llvm::dbgs() << "\t\t- > it is used in a store ("
                    //  << *store << " )\n";

                    append_arg(0, dyn_cast<llvm::AllocaInst>(
                                      store->getValueOperand()));
                }
            }
        } // Is it used in a GEP ?
        else if (auto* gep = dyn_cast<llvm::GetElementPtrInst>(use)) {
            // llvm::dbgs() << "\t- > this is a GEP\n";

            auto index = dyn_cast<llvm ::ConstantInt>(gep->getOperand(2))
                             ->getZExtValue();

            for (auto* use_gep : gep->users()) {
                if (auto* store = dyn_cast<llvm::StoreInst>(use_gep)) {
                    // Found arg
                    // llvm::dbgs() << "\t\t- > it is used in a store ("
                    //  << *store << " )\n";

                    append_arg(index, dyn_cast<llvm::AllocaInst>(
                                          store->getValueOperand()));
                } else if (auto* bitcast =
                               dyn_cast<llvm::BitCastInst>(use_gep)) {
                    // llvm::dbgs() << "\t\t- > it is used in a bitcast
                    // ("
                    //  << *bitcast << " )\n";
                    for (auto* use_bitcast : bitcast->users()) {
                        if (auto* store =
                                dyn_cast<llvm::StoreInst>(use_bitcast)) {
                            // Found arg
                            // llvm::dbgs()
                            // << "\t\t\t- > it is used in a store ("
                            // << *store << " )\n";
                            append_arg(index, dyn_cast<llvm::AllocaInst>(
                                                  store->getValueOperand()));
                        }
                    }
                }
            }
        }
    }

    llvm::dbgs() << "Found args : \n";
    int i = 0;
    for (auto* val : args) {
        if (val == nullptr) {
            throw std::runtime_error(
                llvm::Twine("hip::HostPass::addDeviceStubCall() : Could "
                            "not find argument ")
                    .concat(llvm::Twine(i))
                    .str());
        }
        llvm::dbgs() << i << ' ' << *val << '\n';
        ++i;
    }

    auto* split_point =
        firstCallToFunction(f_original, "__hipPopCallConfiguration");

    // This is going to leave a bunch of garbage but the optimizer will
    // take care of it

    split_point->getParent()->splitBasicBlockBefore(split_point);
    auto* bb_to_delete = split_point->getParent();

    auto* new_bb =
        llvm::BasicBlock::Create(f_original.getContext(), "", &f_original);

    llvm::IRBuilder<> builder(new_bb);

    // Replace basic block with new one calling the stub

    builder.CreateCall(stub, args);
    auto* terminator =
        dyn_cast<llvm::BranchInst>(bb_to_delete->getTerminator());
    builder.CreateBr(terminator->getSuccessor(0));

    bb_to_delete->replaceAllUsesWith(new_bb);
    bb_to_delete->eraseFromParent();
}

llvm::Function* HostPass::getDeviceStub(llvm::GlobalValue* fake_symbol) const {
    auto name = fake_symbol->getName().str();
    auto demangled =
        llvm::demangle(name); // THIS WILL NOT WORK WITH OVERLOADED KERNELS
    auto search = demangled.substr(0, demangled.find('('));

    // Ugly but works. Maybe find a better way ?
    for (auto& f : fake_symbol->getParent()->functions()) {
        auto fname = f.getName().str();
        if (contains(fname, search) && contains(fname, device_stub_prefix) &&
            !contains(fname, cloned_suffix)) {
            return &f;
        }
    }

    return nullptr;
}

llvm::Constant* HostPass::createKernelSymbol(llvm::Function& stub,
                                             llvm::Function& new_stub,
                                             const std::string& suffix) const {
    auto kernel_name = kernelNameFromStub(stub);
    auto suffixed = getClonedName(kernel_name, suffix);

    auto& mod = *stub.getParent();

    return mod.getOrInsertGlobal(
        suffixed, new_stub.getFunctionType()->getPointerTo(),
        [&mod, &new_stub, &suffixed]() {
            return new llvm::GlobalVariable(
                mod, new_stub.getFunctionType()->getPointerTo(), true,
                llvm::GlobalValue::ExternalLinkage, &new_stub, suffixed);
        });
}

std::string HostPass::kernelNameFromStub(llvm::Function& stub) const {
    auto* call_to_launch = firstCallToFunction(stub, "hipLaunchKernel");

    return call_to_launch
        ->getArgOperand(0)    // First argument to hipLaunchKernel
        ->stripPointerCasts() // the stub addrss is automatically casted
                              // to i8, need to remove it
        ->getName()
        .str();
}

} // namespace hip
