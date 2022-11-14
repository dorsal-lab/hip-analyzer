/** \file hip_analyzer_pass
 * \brief Instrumentation pass
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "llvm/Support/CommandLine.h"

#include "hip_instrumentation/basic_block.hpp"
#include "ir_codegen.h"

static llvm::cl::opt<std::string>
    kernel_name("kernel-name", llvm::cl::desc("Specify kernel name"),
                llvm::cl::value_desc("kernel"));

namespace hip {

namespace {

/** \struct AnalysisPass
 * \brief CFG Analysis pass, read cfg and gather static analysis information
 */
class AnalysisPass : public llvm::FunctionPass {
  public:
    static char ID;

    AnalysisPass() : llvm::FunctionPass(ID) {}

    virtual bool runOnFunction(llvm::Function& fn) {
        llvm::errs() << "Function " << fn.getName() << '\n';
        fn.print(llvm::dbgs(), nullptr);

        auto i = 0u;
        for (auto& bb : fn) {
            if (isBlockInstrumentable(bb)) {
                blocks.emplace_back(getBlockInfo(bb, i));
            }

            ++i;
        }

        return false;
    }

    const std::vector<hip::InstrumentedBlock>& getBlocks() { return blocks; }

  private:
    /** \brief List of instrumented block, ordered by the llvm block id in the
     * function
     */
    std::vector<hip::InstrumentedBlock> blocks;
};

struct CfgInstrumentationPass : public llvm::ModulePass {
    static char ID;
    static const std::string instrumented_prefix;
    static const std::string utils_path;

    CfgInstrumentationPass() : llvm::ModulePass(ID) {}

    virtual bool runOnModule(llvm::Module& mod) override {
        bool modified = false;
        for (auto& f_original : mod.functions()) {
            if (!isInstrumentableKernel(f_original)) {
                continue;
            }

            llvm::errs() << "Function " << f_original.getName() << '\n';
            f_original.print(llvm::dbgs(), nullptr);

            auto& f = cloneWithPrefix(
                f_original, instrumented_prefix,
                {getCounterType(mod.getContext())->getPointerTo()});

            modified |= addParams(f, f_original);

            llvm::errs() << "Function " << f.getName() << '\n';
            f.print(llvm::dbgs(), nullptr);

            modified |= instrumentFunction(f, f_original);
        }

        // Add necessary functions
        linkModuleUtils(mod);

        return modified;
    }

    /** \fn isInstrumentableKernel
     * \brief Returns whether a function is a kernel that will be instrumented
     * (todo?)
     */
    bool isInstrumentableKernel(llvm::Function& f) {
        return !f.isDeclaration() &&
               !contains(f.getName().str(), cloned_suffix);
    }

    /** \fn addParams
     * \brief Clone function with new parameters
     */
    virtual bool addParams(llvm::Function& f,
                           llvm::Function& original_function) {

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

    /** \fn instrumentFunction
     * \brief Add CFG counters instrumentation to the compute kernel
     *
     * \param f Kernel
     * \param original_function Original (non-instrumented) kernel
     */
    virtual bool instrumentFunction(llvm::Function& f,
                                    llvm::Function& original_function) {
        auto& blocks = getAnalysis<AnalysisPass>(original_function).getBlocks();
        auto& context = f.getContext();
        auto instrumentation_handlers = declareInstrumentation(*f.getParent());
        auto* instr_ptr = f.getArg(f.arg_size() - 1);

        // Add counters
        auto* counter_type = getCounterType(context);
        auto* array_type = llvm::ArrayType::get(counter_type, blocks.size());

        llvm::IRBuilder<> builder_locals(&f.getEntryBlock());
        setInsertPointPastAllocas(builder_locals, f);

        auto* counters = builder_locals.CreateAlloca(
            array_type, nullptr, llvm::Twine("_bb_counters"));

        // Initialize it!

        builder_locals.CreateMemSet(
            builder_locals.CreatePointerCast(
                counters, llvm::Type::getInt8PtrTy(f.getContext())),
            llvm::ConstantInt::get(counter_type, 0u), blocks.size(),
            llvm::MaybeAlign(llvm::Align::Constant<1>()));

        // Instrument each basic block

        auto& function_block_list = f.getBasicBlockList();
        auto curr_bb = f.begin();
        auto index = 0u;

        for (auto& bb_instr : blocks) {
            while (index < bb_instr.id) {
                ++index;
                ++curr_bb;
            }

            builder_locals.SetInsertPoint(
                &(*curr_bb), getFirstNonPHIOrDbgOrAlloca(*curr_bb));

            auto* inbound_ptr = builder_locals.CreateInBoundsGEP(
                array_type, counters,
                {getIndex(0u, context), getIndex(bb_instr.id, context)});

            auto* curr_ptr =
                builder_locals.CreateLoad(counter_type, inbound_ptr);

            auto* incremented = builder_locals.CreateAdd(
                curr_ptr, llvm::ConstantInt::get(counter_type, 1u));

            auto* store = builder_locals.CreateStore(incremented, inbound_ptr);
        }

        // Call saving method

        for (auto& bb_instr : f) {
            auto terminator = bb_instr.getTerminator();

            // Only call saving method if the terminator is a return
            if (isa<llvm::ReturnInst>(terminator)) {
                builder_locals.SetInsertPoint(terminator);

                // Bitcast to ptr

                auto* array_ptr =
                    builder_locals.CreatePointerBitCastOrAddrSpaceCast(
                        counters, counter_type->getPointerTo());

                // Add call
                builder_locals.CreateCall(
                    instrumentation_handlers._hip_store_ctr,
                    {array_ptr, getIndex(blocks.size(), context), instr_ptr});
            }
        }

        f.print(llvm::dbgs(), nullptr);

        return true;
    }

    void linkModuleUtils(llvm::Module& mod) {
        llvm::Linker linker(mod);

        // Load compiled module
        llvm::SMDiagnostic diag;
        auto utils_mod = llvm::parseIRFile(utils_path, diag, mod.getContext());
        if (!utils_mod) {
            llvm::errs() << diag.getMessage() << '\n';
            throw std::runtime_error("CfgInstrumentationPass::linkModuleUtils()"
                                     " : Could not load utils module");
        }

        linker.linkInModule(std::move(utils_mod));

        // Remove [[clang::optnone]] and add [[clang::always_inline]] attributes

        auto instrumentation_handlers = declareInstrumentation(mod);

        instrumentation_handlers._hip_store_ctr->removeFnAttr(
            llvm::Attribute::OptimizeNone);
        instrumentation_handlers._hip_store_ctr->removeFnAttr(
            llvm::Attribute::NoInline);
        instrumentation_handlers._hip_store_ctr->addFnAttr(
            llvm::Attribute::AlwaysInline);
    }

    static llvm::Type* getCounterType(llvm::LLVMContext& context) {
        return llvm::Type::getInt8Ty(context);
    }

    virtual void getAnalysisUsage(llvm::AnalysisUsage& Info) const override {
        Info.addRequired<AnalysisPass>();
    }
};

struct TracingPass : public llvm::ModulePass {
    static char ID;

    TracingPass() : llvm::ModulePass(ID) {}

    virtual bool runOnModule(llvm::Module& fn) override { return false; }

    virtual bool instrumentFunction(llvm::Function& f) {
        auto& blocks = getAnalysis<AnalysisPass>(f).getBlocks();

        return false;
    }

    virtual void getAnalysisUsage(llvm::AnalysisUsage& Info) const override {
        Info.addRequired<AnalysisPass>();
    }
};

/** \brief The hipcc compiler inserts device stubs for each kernel call, which
 * in turns actually launches the kernel.
 */
constexpr std::string_view device_stub_prefix = "__device_stub__";

/** \struct HostPass
 * \brief The Host pass is responsible for adding device stubs for the new
 * instrumented kernels.
 *
 */
struct HostPass : public llvm::ModulePass {
    static char ID;

    HostPass() : llvm::ModulePass(ID) {}

    virtual bool runOnModule(llvm::Module& mod) override {
        bool modified = false;

        llvm::SmallVector<std::pair<llvm::Function*, llvm::Function*>, 8>
            to_delete;
        for (auto& f_original : mod.functions()) {
            if (isDeviceStub(f_original) && !f_original.isDeclaration()) {
                // Device stub
                llvm::errs() << "Function " << f_original.getName() << '\n';
                f_original.print(llvm::dbgs());

                // Duplicates the stub and calls the appropriate function
                auto& stub_counter = addCountersDeviceStub(f_original);

                // Replaces all call to the original stub with tmp_<stub name>,
                // and add the old function to an erase list
                if (auto* new_stub = replaceStubCall(f_original)) {
                    to_delete.push_back(std::make_pair(&f_original, new_stub));
                }

                // stub_counter.print(llvm::dbgs());
            } else if (isKernelCallSite(f_original)) {
                // Kernel calling site

                // Sadly the HIP Front-end inlines all kernel device stubs, so
                // we have to substitute the call ourselves

                addDeviceStubCall(f_original);

                f_original.print(llvm::dbgs());
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

        return true;
        // TODO : link needed functions

        // TODO : remove `optnone` and `noinline` attributes, add alwaysinline
    }

    /** \fn addCountersDeviceStub
     * \brief Copies the device stub and adds additional arguments for the
     * counters instrumentation
     *
     * \param f_original Original kernel device stub
     *
     * \returns Instrumented device stub
     */
    llvm::Function& addCountersDeviceStub(llvm::Function& f_original) const {
        auto& mod = *f_original.getParent();

        auto& f = cloneWithPrefix(
            f_original, CfgInstrumentationPass::instrumented_prefix,
            {CfgInstrumentationPass::getCounterType(mod.getContext())
                 ->getPointerTo()});

        llvm::ValueToValueMapTy vmap;

        for (auto it1 = f_original.arg_begin(), it2 = f.arg_begin();
             it1 != f_original.arg_end(); ++it1, ++it2) {
            vmap[&*it1] = &*it2;
        }
        llvm::SmallVector<llvm::ReturnInst*, 8> returns;

        llvm::CloneFunctionInto(&f, &f_original, vmap,
                                llvm::CloneFunctionChangeType::LocalChangesOnly,
                                returns);

        // Add additional arguments to the void* array : uint8_t* _counters;

        pushAdditionalArguments(f, {f.getArg(f.arg_size() - 1)});

        // Create global symbol

        auto* global_sym = createKernelSymbol(
            f_original, f, CfgInstrumentationPass::instrumented_prefix);

        // Modify the call to hipLaunchKernel

        auto* call_to_launch = firstCallToFunction(f, "hipLaunchKernel");

        call_to_launch->setArgOperand(
            0, llvm::ConstantExpr::getPointerCast(
                   global_sym, llvm::Type::getInt8PtrTy(mod.getContext())));

        // Modify __hip_module_ctor to register kernel

        return f;
    }

    llvm::Function* replaceStubCall(llvm::Function& stub) {
        auto& mod = *stub.getParent();

        auto fun_type = stub.getFunctionType();

        auto* new_stub = dyn_cast<llvm::Function>(
            mod.getOrInsertFunction(
                   getClonedName(stub,
                                 CfgInstrumentationPass::instrumented_prefix +
                                     "tmp_"),
                   fun_type)
                .getCallee());

        auto* counters_stub = &cloneWithPrefix(
            stub, CfgInstrumentationPass::instrumented_prefix,
            {CfgInstrumentationPass::getCounterType(mod.getContext())
                 ->getPointerTo()});

        auto* bb = llvm::BasicBlock::Create(mod.getContext(), "", new_stub);

        llvm::IRBuilder<> builder(bb);

        // Create call to newly created stub
        llvm::SmallVector<llvm::Value*> args;
        for (llvm::Argument& arg : stub.args()) {
            args.push_back(&arg);
        }

        // TODO : call instrumenation runtime to get pointer to instr
        args.push_back(llvm::ConstantPointerNull::get(
            CfgInstrumentationPass::getCounterType(mod.getContext())
                ->getPointerTo()));

        builder.CreateCall(counters_stub->getFunctionType(), counters_stub,
                           args);

        builder.CreateRetVoid();

        // Replace all calls

        stub.replaceAllUsesWith(new_stub);

        return new_stub;
    }

    /** \fn addCountersDeviceStub
     * \brief Replaces the (inlined) device stub call for the
     * counters-instrumented call
     *
     * \param f_original Original kernel device stub call site
     */
    void addDeviceStubCall(llvm::Function& f_original) const {
        auto* kernel_call = firstCallToFunction(f_original, "hipLaunchKernel");
        auto* inliner_bb = kernel_call->getParent();

        auto* gep_of_array =
            dyn_cast<llvm::GetElementPtrInst>(kernel_call->getArgOperand(5));

        // Get device stub

        auto* stub = getDeviceStub(dyn_cast<llvm::GlobalValue>(
            dyn_cast<llvm::ConstantExpr>(kernel_call->getArgOperand(0))
                ->getOperand(0)));

        if (stub == nullptr) {
            throw std::runtime_error(
                "HostPass::addDeviceStubCall() : Could not find device stub");
        }

        auto* args_array = gep_of_array->getPointerOperand();

        // Build the value vector to pass to the new kernel stub

        llvm::SmallVector<llvm::Value*, 8> args{stub->arg_size(), nullptr};

        auto append_arg = [&args, &stub](size_t idx,
                                         llvm::AllocaInst* stack_storage) {
            for (auto* use_stack : stack_storage->users()) {
                if (auto* store_inst = dyn_cast<llvm::StoreInst>(use_stack)) {
                    if (stack_storage == store_inst->getPointerOperand()) {
                        llvm::dbgs() << *store_inst << '\n';

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
                        // llvm::dbgs() << "\t\t- > it is used in a bitcast ("
                        //  << *bitcast << " )\n";
                        for (auto* use_bitcast : bitcast->users()) {
                            if (auto* store =
                                    dyn_cast<llvm::StoreInst>(use_bitcast)) {
                                // Found arg
                                // llvm::dbgs()
                                // << "\t\t\t- > it is used in a store ("
                                // << *store << " )\n";
                                append_arg(index,
                                           dyn_cast<llvm::AllocaInst>(
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

        // This is going to leave a bunch of garbage but the optimizer will take
        // care of it

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

    /** \fn getDeviceStub
     * \brief Returns the device stub of a given kernel symbol (kernel symbols
     * are created for the host but not defined)
     */
    llvm::Function* getDeviceStub(llvm::GlobalValue* fake_symbol) const {
        auto name = fake_symbol->getName().str();
        auto demangled =
            llvm::demangle(name); // THIS WILL NOT WORK WITH OVERLOADED KERNELS
        auto search = demangled.substr(0, demangled.find('('));

        // Ugly but works. Maybe find a better way ?
        for (auto& f : fake_symbol->getParent()->functions()) {
            auto fname = f.getName().str();
            if (contains(fname, search) &&
                contains(fname, device_stub_prefix) &&
                !contains(fname, cloned_suffix)) {
                return &f;
            }
        }

        return nullptr;
    }

    /** \fn createKernelSymbol
     * \brief Create the global kernel function symbol for the copied kernel
     *
     * \param stub original kernel host stub
     * \param new_stub new kernel stub with appropriate return type
     * \param suffix suffix added to the kernel name
     *
     */
    llvm::Constant* createKernelSymbol(llvm::Function& stub,
                                       llvm::Function& new_stub,
                                       const std::string& suffix) const {
        auto kernel_name = kernelNameFromStub(stub);
        auto suffixed = getClonedName(kernel_name, suffix);

        auto& mod = *stub.getParent();

        return mod.getOrInsertGlobal(suffixed, new_stub.getFunctionType());
    }

    /** \fn kernelNameFromStub
     * \brief Returns the kernel identifier from device stub function
     */
    std::string kernelNameFromStub(llvm::Function& stub) const {
        auto* call_to_launch = firstCallToFunction(stub, "hipLaunchKernel");

        return call_to_launch
            ->getArgOperand(0)    // First argument to hipLaunchKernel
            ->stripPointerCasts() // the stub addrss is automatically casted to
                                  // i8, need to remove it
            ->getName()
            .str();
    }

    bool isDeviceStub(llvm::Function& f) {
        auto name = f.getName().str();

        return contains(name, device_stub_prefix) &&
               !contains(name, cloned_suffix +
                                   CfgInstrumentationPass::instrumented_prefix);
    }

    bool isKernelCallSite(llvm::Function& f) {
        // True if calls hipLaunchKernel and is not a stub (so an inlined stub)
        return hasFunctionCall(f, "hipLaunchKernel") && !isDeviceStub(f) &&
               !contains(f.getName().str(),
                         cloned_suffix +
                             CfgInstrumentationPass::instrumented_prefix);
    }

    virtual void getAnalysisUsage(llvm::AnalysisUsage& Info) const override {
        // Info.addRequired<AnalysisPass>();
    }
};

char AnalysisPass::ID = 0;
const std::string CfgInstrumentationPass::instrumented_prefix = "counters_";
const std::string CfgInstrumentationPass::utils_path = "gpu_pass_instr.ll";
char CfgInstrumentationPass::ID = 1;
char TracingPass::ID = 2;
char HostPass::ID = 3;

static void registerAnalysisPass(const llvm::PassManagerBuilder&,
                                 llvm::legacy::PassManagerBase& PM) {
    PM.add(new AnalysisPass());
}

static void registerCfgPass(const llvm::PassManagerBuilder&,
                            llvm::legacy::PassManagerBase& PM) {
    PM.add(new CfgInstrumentationPass());
}
static void registerTracingPass(const llvm::PassManagerBuilder&,
                                llvm::legacy::PassManagerBase& PM) {
    PM.add(new TracingPass());
}

static void registerHostPass(const llvm::PassManagerBuilder&,
                             llvm::legacy::PassManagerBase& PM) {
    PM.add(new HostPass());
}

} // namespace

static llvm::RegisterPass<AnalysisPass>
    RegisterAnalysisPass("hip-analyzer", "Hip-Analyzer analysis pass", true,
                         true);

static llvm::RegisterPass<CfgInstrumentationPass>
    RegisterCfgCountersPass("hip-analyzer-counters",
                            "Hip-Analyzer cfg counters pass", false, false);

static llvm::RegisterPass<TracingPass>
    RegisterTracingPass("hip-analyzer-tracing", "Hip-Analyzer tracing pass",
                        false, false);

static llvm::RegisterPass<HostPass>
    RegisterHostPass("hip-analyzer-host",
                     "Hip-Analyzer host instrumentation pass", false, false);

} // namespace hip

static llvm::RegisterStandardPasses
    registerAnalysisPass(llvm::PassManagerBuilder::EP_EarlyAsPossible,
                         hip::registerAnalysisPass);

static llvm::RegisterStandardPasses
    registerCfgPass(llvm::PassManagerBuilder::EP_EarlyAsPossible,
                    hip::registerCfgPass);

static llvm::RegisterStandardPasses
    registerTracingPass(llvm::PassManagerBuilder::EP_EarlyAsPossible,
                        hip::registerTracingPass);

static llvm::RegisterStandardPasses
    registerHostPass(llvm::PassManagerBuilder::EP_EarlyAsPossible,
                     hip::registerHostPass);
