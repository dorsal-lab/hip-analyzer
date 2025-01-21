#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "backend/wave_basic_block_counters.h"
#include "ir_codegen.h"

#include <iostream>

using namespace llvm;

#define DEBUG_TYPE "duplicate-kernels"

char DuplicateKernels::ID = 0;

INITIALIZE_PASS(DuplicateKernels, DEBUG_TYPE,
                "Duplicate AMDGPU kernels for instrumentation", false, false);

bool addParams(llvm::Function& f, llvm::Function& original_function) {

    // To be quite honest, I am not really sure how this works. It might be
    // possibly buggy in some cases.

    llvm::ValueToValueMapTy vmap;

    for (auto it1 = original_function.arg_begin(), it2 = f.arg_begin();
         it1 != original_function.arg_end(); ++it1, ++it2) {
        vmap[&*it1] = &*it2;
        it2->setName(it1->getName());
    }
    llvm::SmallVector<llvm::ReturnInst*, 8> returns;

    llvm::CloneFunctionInto(&f, &original_function, vmap,
                            llvm::CloneFunctionChangeType::LocalChangesOnly,
                            returns);

    return true;
}

bool DuplicateKernels::runOnModule(Module& mod) {
    if (!hip::isDeviceModule(mod)) {
        // DO NOT run on host code
        return false;
    }

    constexpr std::string_view prefix = "dup_";

    auto* ptr_ty = llvm::PointerType::get(mod.getContext(), 1);

    bool modified = false;
    for (auto& f_original : mod.functions()) {
        if (!hip::isInstrumentableKernel(f_original)) {
            continue;
        }
        // Clone the kernel, with extra arguments
        auto& f = hip::cloneWithPrefix(f_original, prefix, {ptr_ty});

        modified |= addParams(f, f_original);

        // Fake uses. Use inline asm to prevent optimizing out the argument load

        auto* load = llvm::InlineAsm::get(
            llvm::FunctionType::get(ptr_ty, {ptr_ty}, false),
            "s_load_dwordx2 $0, $1//Test\n", "=s,s", true);

        llvm::IRBuilder<> builder(&(*f.begin()->begin()));
        for (auto arg = f.arg_begin() + f_original.arg_size();
             arg != f.arg_end(); ++arg) {
            auto* val = builder.CreateCall(load, {arg});
        }

        llvm::errs() << "Duplicated kernel : " << f.getName() << '\n';

        f.dump();
    }

    return modified;
}
