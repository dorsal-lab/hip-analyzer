/** \file hip_analyzer_pass
 * \brief Instrumentation pass
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
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
    static const char* instrumented_suffix;

    CfgInstrumentationPass() : llvm::ModulePass(ID) {}

    virtual bool runOnModule(llvm::Module& mod) override {
        bool modified = false;
        for (auto& f_original : mod.functions()) {
            if (!isInstrumentableKernel(f_original)) {
                continue;
            }

            llvm::errs() << "Function " << f_original.getName() << '\n';
            f_original.print(llvm::dbgs(), nullptr);

            auto& f = cloneWithSuffix(
                f_original, instrumented_suffix,
                {getCounterType(mod.getContext())->getPointerTo()});

            modified |= addParams(f, f_original);

            llvm::errs() << "Function " << f.getName() << '\n';
            f.print(llvm::dbgs(), nullptr);

            modified |= instrumentFunction(f, f_original);
        }

        return modified;
    }

    bool isInstrumentableKernel(llvm::Function& f) {
        return !f.isDeclaration() && !f.getName().endswith(cloned_suffix);
    }

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
            if (isa<llvm::ReturnInst>(terminator)) {
                builder_locals.SetInsertPoint(terminator);

                // Bitcast to ptr

                auto* array_ptr = builder_locals.CreateBitCast(
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
        for (auto& f_original : mod.functions()) {
            if (!isDeviceStub(f_original) || f_original.isDeclaration()) {
                continue;
            }

            llvm::errs() << "Function " << f_original.getName() << '\n';
            f_original.print(llvm::dbgs(), nullptr);

            auto& stub_counter = addCountersDeviceStub(f_original);

            stub_counter.print(llvm::dbgs(), nullptr);
        }

        return true;
        // TODO : link needed functions

        // TODO : remove `optnone` and `noinline` attributes, add alwaysinline
    }

    llvm::Function& addCountersDeviceStub(llvm::Function& f_original) const {
        auto& mod = *f_original.getParent();

        auto& f = cloneWithSuffix(
            f_original, CfgInstrumentationPass::instrumented_suffix,
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
            f_original, f, CfgInstrumentationPass::instrumented_suffix);

        // Modify the call to hipLaunchKernel

        auto* call_to_launch = firstCallToFunction(f, "hipLaunchKernel");

        auto counters_kernel_name = getClonedName(
            kernelNameFromStub(f), CfgInstrumentationPass::instrumented_suffix);

        llvm::errs() << counters_kernel_name << '\n';

        // Modify __hip_module_ctor to register kernel

        return f;
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
        auto suffixed = kernel_name + suffix;

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
        auto name = llvm::demangle(f.getName().str());
        return name.starts_with(device_stub_prefix);
    }

    virtual void getAnalysisUsage(llvm::AnalysisUsage& Info) const override {
        // Info.addRequired<AnalysisPass>();
    }
};

char AnalysisPass::ID = 0;
const char* CfgInstrumentationPass::instrumented_suffix = "_counters";
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
