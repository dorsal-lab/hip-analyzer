/** \file hip_analyzer_pass
 * \brief Instrumentation pass
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

#include "hip_instrumentation/basic_block.hpp"
#include "ir_codegen.h"

namespace hip {

// ----- Constants ----- //

/** \brief The hipcc compiler inserts device stubs for each kernel call,
 * which in turns actually launches the kernel.
 */
constexpr std::string_view device_stub_prefix = "__device_stub__";

// ----- Passes ----- //

/** \struct AnalysisPass
 * \brief CFG Analysis pass, read cfg and gather static analysis information
 */
class AnalysisPass : public llvm::AnalysisInfoMixin<AnalysisPass> {
  public:
    using Result = std::vector<hip::InstrumentedBlock>;

    AnalysisPass() {}

    Result run(llvm::Function& fn, llvm::FunctionAnalysisManager& fam);

  private:
    static llvm::AnalysisKey Key;
    friend struct llvm::AnalysisInfoMixin<AnalysisPass>;
};

struct KernelInstrumentationPass
    : public llvm::PassInfoMixin<KernelInstrumentationPass> {
    llvm::PreservedAnalyses run(llvm::Module& mod,
                                llvm::ModuleAnalysisManager& modm);

    /** \fn isInstrumentableKernel
     * \brief Returns whether a function is a kernel that will be
     * instrumented (todo?)
     */
    virtual bool isInstrumentableKernel(const llvm::Function& f) const = 0;

    /** \fn addParams
     * \brief Clone function with new parameters
     */
    bool addParams(llvm::Function& f, llvm::Function& original_function);

    /** \fn getExtraArguments
     * \brief Return the necessary extra arguments for the instrumentation type
     */
    virtual llvm::SmallVector<llvm::Type*>
    getExtraArguments(llvm::LLVMContext& context) const = 0;

    /** \fn instrumentFunction
     * \brief Add CFG counters instrumentation to the compute kernel
     *
     * \param kernel Kernel
     * \param original_kernel Original (non-instrumented) kernel
     */
    virtual bool instrumentFunction(llvm::Function& kernel,
                                    llvm::Function& original_kernel,
                                    AnalysisPass::Result& block_report) = 0;

    virtual const std::string& getInstrumentedKernelPrefix() const = 0;

    /** \fn linkModuleUtils
     * \brief Link (at IR-level) all necessary utility functions
     */
    virtual void linkModuleUtils(llvm::Module& mod) = 0;
};

struct CfgInstrumentationPass : public KernelInstrumentationPass {
    static const std::string instrumented_prefix;
    static const std::string utils_path;

    CfgInstrumentationPass() {}

    bool isInstrumentableKernel(const llvm::Function& f) const override;

    bool instrumentFunction(llvm::Function& f,
                            llvm::Function& original_function,
                            AnalysisPass::Result& block_report) override;

    llvm::SmallVector<llvm::Type*>
    getExtraArguments(llvm::LLVMContext& context) const override;

    void linkModuleUtils(llvm::Module& mod) override;

    const std::string& getInstrumentedKernelPrefix() const override {
        return instrumented_prefix;
    }

    static llvm::Type* getCounterType(llvm::LLVMContext& context) {
        return llvm::Type::getInt8Ty(context);
    }
};

/** \class TraceType
 * \brief Abstract trace type. Should contain necessary information to create
 * the LLVM struct type and its "constructor".
 */
class TraceType {
  public:
    /** \fn create
     * \brief Factory function to create a trace type
     */
    static std::unique_ptr<TraceType> create(const std::string& trace_type);
    virtual ~TraceType() = default;

    /** \fn getEventType
     * \brief Returns the event type (struct containing all the traced types)
     */
    virtual llvm::Type* getEventType(llvm::LLVMContext&) const = 0;

    /** \fn getEventCtor
     * \brief Returns the event constructor, which populates the aformentioned
     * struct. The event constructor calls the underlying C++ object constructor
     * using the placement-new syntax
     */
    virtual llvm::Function* getEventCtor(llvm::Module&) const = 0;

    /** \fn getEventSize
     * \brief Returns the size (in bytes) of the event type
     */
    virtual llvm::ConstantInt* getEventSize(llvm::Module& mod) const {
        auto& context = mod.getContext();
        return llvm::ConstantInt::get(
            llvm::Type::getInt64Ty(context),
            mod.getDataLayout()
                .getTypeAllocSize(getEventType(context))
                .getFixedSize());
    };

    /** \fn getOffsetGetter
     * \brief Returns the offset getter that is appropriate for this trace type
     * (thread or wave)
     */
    virtual llvm::Function* getOffsetGetter(llvm::Module& mod) const {
        return TracingFunctions{mod}._hip_get_trace_offset;
    }

    /** \fn getEventCreator
     * \brief Returns the event creator that is appropriate for this trace type.
     *
     * \details Called before the event *constructor*, the creator is
     * responsible for the management of the queue type. It then calls the event
     * constructor at the appropriate address
     */
    virtual llvm::Function* getEventCreator(llvm::Module& mod) const {
        return TracingFunctions{mod}._hip_create_event;
    }
};

struct TracingPass : public KernelInstrumentationPass {
    static const std::string instrumented_prefix;
    static const std::string utils_path;

    TracingPass(const std::string& trace_type)
        : event(TraceType::create(trace_type)) {
        if (!event) {
            throw std::runtime_error(
                "TracingPass::TracingPass() : Unknown trace \"" + trace_type +
                "\".");
        }
    }

    bool isInstrumentableKernel(const llvm::Function& f) const override;

    bool instrumentFunction(llvm::Function& f,
                            llvm::Function& original_function,
                            AnalysisPass::Result& block_report) override;

    llvm::SmallVector<llvm::Type*>
    getExtraArguments(llvm::LLVMContext& context) const override;

    void linkModuleUtils(llvm::Module& mod) override;

    const std::string& getInstrumentedKernelPrefix() const override {
        // TODO : Adapt to tracing name
        return instrumented_prefix;
    }

  private:
    std::unique_ptr<TraceType> event;
};

/** \struct HostPass
 * \brief The Host pass is responsible for adding device stubs for the new
 * instrumented kernels.
 *
 */
struct HostPass : public llvm::PassInfoMixin<HostPass> {
    HostPass(bool tracing = true) : trace(tracing) {}

    llvm::PreservedAnalyses run(llvm::Module& mod,
                                llvm::ModuleAnalysisManager& modm);

    /** \fn addCountersDeviceStub
     * \brief Copies the device stub and adds additional arguments for the
     * counters instrumentation
     *
     * \param f_original Original kernel device stub
     *
     * \returns Instrumented device stub
     */
    llvm::Function* addCountersDeviceStub(llvm::Function& f_original) const;

    /** \fn addTracingDeviceStub
     * \brief Copies the device stub and adds additional arguments for the
     * tracing instrumentation
     *
     * \param f_original Original kernel device stub
     *
     * \returns Instrumented device stub
     */
    llvm::Function* addTracingDeviceStub(llvm::Function& f_original) const;

    llvm::Function*
    duplicateStubWithArgs(llvm::Function& f_original, const std::string& prefix,
                          llvm::ArrayRef<llvm::Type*> new_args) const;

    llvm::Function* replaceStubCall(llvm::Function& stub) const;

    /** \fn addCountersDeviceStub
     * \brief Replaces the (inlined) device stub call for the
     * counters-instrumented call
     *
     * \param f_original Original kernel device stub call site
     */
    void addDeviceStubCall(llvm::Function& f_original) const;

    /** \fn getDeviceStub
     * \brief Returns the device stub of a given kernel symbol (kernel
     * symbols are created for the host but not defined)
     */
    llvm::Function* getDeviceStub(llvm::GlobalValue* fake_symbol) const;

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
                                       const std::string& suffix) const;

    /** \fn kernelNameFromStub
     * \brief Returns the kernel identifier from device stub function
     */
    std::string kernelNameFromStub(llvm::Function& stub) const;

  private:
    bool trace;
};

} // namespace hip
