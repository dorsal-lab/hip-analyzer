#include "llvm/CodeGen/MachineFunctionPass.h"
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

    bool modified = false;
    for (auto& f_original : mod.functions()) {
        if (!hip::isInstrumentableKernel(f_original)) {
            continue;
        }
        // Clone the kernel, with extra arguments
        auto& f = hip::cloneWithPrefix(
            f_original, prefix,
            {llvm::PointerType::getUnqual(mod.getContext())});

        modified |= addParams(f, f_original);

        llvm::errs() << "Duplicated kernel : " << f.getName() << '\n';
    }

    return modified;
}
