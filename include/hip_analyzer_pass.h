/** \file hip_analyzer_pass
 * \brief Instrumentation pass
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

#include "hip_instrumentation/basic_block.hpp"
#include "ir_codegen.h"

#include <cstdint>
#include <utility>

namespace hip {

// ----- Constants ----- //

/** \brief The hipcc compiler inserts device stubs for each kernel call,
 * which in turns actually launches the kernel.
 */
constexpr std::string_view device_stub_prefix = "__device_stub__";

/** \brief We link a dummy kernel to prevent optimizing some functions away. Do
 * not instrument it as it would be a waste of time
 * */
constexpr std::string_view dummy_kernel_name =
    "_ZN12_GLOBAL__N_118dummy_kernel_nooptEmPh";

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
    virtual bool isInstrumentableKernel(const llvm::Function& f) const;

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
    virtual void linkModuleUtils(llvm::Module& mod);

    static const std::string utils_path;
};

struct CounterType {
  public:
    virtual ~CounterType() = default;
    /** \fn getInstrumentedPrefix
     * \brief Returns the instrumented kernel prefix corresponding to the
     * counters type
     */
    virtual const std::string& getInstrumentedPrefix() const = 0;

    /** \fn getInstrumenterType
     * \brief Returns the constant i32 to pass to hipNewInstrumenter to create
     * the proper CounterInstrumenter (defined in \ref
     * hip_instrumentation_cbindings.hpp)
     */
    virtual llvm::ConstantInt*
    getInstrumenterType(llvm::LLVMContext&) const = 0;

    static std::unique_ptr<CounterType> create(const std::string& type);
};

struct CountersInstrumentationPass : public KernelInstrumentationPass {};

/** \struct CfgInstrumentationpass
 * \brief Thread-level basic block counters insertion pass
 */
struct ThreadCountersInstrumentationPass : public CountersInstrumentationPass {
    static const std::string instrumented_prefix;
    static constexpr uint32_t CounterInstrumenterId = 0u;
    static constexpr auto CounterType = "thread-counters";

    ThreadCountersInstrumentationPass() {}

    bool instrumentFunction(llvm::Function& f,
                            llvm::Function& original_function,
                            AnalysisPass::Result& block_report) override;

    llvm::SmallVector<llvm::Type*>
    getExtraArguments(llvm::LLVMContext& context) const override;

    const std::string& getInstrumentedKernelPrefix() const override {
        return instrumented_prefix;
    }

    static llvm::Type* getCounterType(llvm::LLVMContext& context) {
        return llvm::Type::getInt8Ty(context);
    }
};

/** \struct CfgInstrumentationpass
 * \brief Wavefront-level basic block counters insertion pass
 */
struct WaveCountersInstrumentationPass : public CountersInstrumentationPass {
    static const std::string instrumented_prefix;
    static constexpr uint32_t CounterInstrumenterId = 1u;
    static constexpr auto CounterType = "wave-counters";

    WaveCountersInstrumentationPass() {}

    bool instrumentFunction(llvm::Function& f,
                            llvm::Function& original_function,
                            AnalysisPass::Result& block_report) override;

    llvm::SmallVector<llvm::Type*>
    getExtraArguments(llvm::LLVMContext& context) const override;

    const std::string& getInstrumentedKernelPrefix() const override {
        return instrumented_prefix;
    }

    static llvm::Type* getScalarCounterType(llvm::LLVMContext& context) {
        return llvm::Type::getInt32Ty(context);
    }

    static llvm::VectorType* getVectorCounterType(llvm::LLVMContext& context,
                                                  uint64_t bb_count);

    /** \fn getCounterAndIncrement
     * \brief Increment the `bb` value in the vector of counters, and returns
     * the value
     */
    llvm::Value* getCounterAndIncrement(llvm::Module& mod,
                                        llvm::IRBuilder<>& builder, unsigned bb,
                                        std::string_view reg);

    void storeCounter(llvm::IRBuilder<>& builder, llvm::Value* ptr, unsigned bb,
                      std::string_view reg);
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
    virtual llvm::ConstantInt* getEventSize(const llvm::Module& mod) const {
        auto& context = mod.getContext();
        return llvm::ConstantInt::get(
            llvm::Type::getInt64Ty(context),
            mod.getDataLayout()
                .getTypeAllocSize(getEventType(context))
                .getFixedValue());
    };

    /** \fn getOffsetGetter
     * \brief Returns the offset getter that is appropriate for this trace type
     * (thread or wave)
     */
    virtual llvm::Function* getOffsetGetter(llvm::Module& mod) const {
        return TracingFunctions{mod}._hip_get_trace_offset;
    }

    virtual llvm::Value* getThreadStorage(llvm::Module& mod,
                                          llvm::IRBuilder<>& builder,
                                          llvm::Value* storage_ptr,
                                          llvm::Value* offsets_ptr) {
        return builder.CreateCall(
            getOffsetGetter(mod),
            {storage_ptr, offsets_ptr, getEventSize(mod)});
    }

    /** \fn initTracingIndices
     * \brief Some tracing types require a much more involved infrastructure for
     * keeping counters (wave specifically). We have to create a map assigning
     * values of counters (pre-increment, post-increment) for each basic block.
     */
    virtual const std::map<llvm::BasicBlock*,
                           std::pair<llvm::Value*, llvm::Value*>>&
    initTracingIndices(llvm::Function& kernel) = 0;

    /** \fn finalizeTracingIndices
     * \brief
     */
    virtual void finalizeTracingIndices(llvm::Function& kernel) = 0;

    /** \fn traceIdxAtBlock
     * \brief Returns the trace index in a given basic block. Preferred way to
     * access the counter rather than indexing in the return index map.
     */
    virtual llvm::Value* traceIdxAtBlock(llvm::BasicBlock& bb) = 0;

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

    virtual std::pair<llvm::Value*, llvm::Value*>
    getQueueType(llvm::Module& mod) const = 0;

    /** \fn getCounterAndIncrement
     * \brief Increment a int32 counter and return the old value (effectively
     * returns counter++;). If \p counter is a pointer to int32, load the value
     * and store the incremented val.
     */
    virtual llvm::Value* getCounterAndIncrement(llvm::Module& mod,
                                                llvm::IRBuilder<>& builder,
                                                llvm::Value* counter) = 0;

    virtual void createEvent(llvm::Module& mod, llvm::IRBuilder<>& builder,
                             llvm::Value* thread_storage, llvm::Value* counter,
                             uint64_t bb) {
        builder.CreateCall(getEventCreator(mod),
                           {thread_storage, counter, getEventSize(mod),
                            getEventCtor(mod), getIndex(bb, mod.getContext())});
    }

    /** \fn getCounterType
     * \brief Returns the scalar counter type used as a trace index. Defaults to
     * a 32-bit integer
     */
    virtual llvm::Type* getCounterType(llvm::Module& mod) const {
        return llvm::Type::getInt32Ty(mod.getContext());
    }

  protected:
    static std::pair<llvm::Value*, llvm::Value*>
    getPair(llvm::LLVMContext& context, uint64_t event, uint64_t queue) {
        return {llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), event),
                llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), queue)};
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
    HostPass(bool tracing = false,
             const std::string& counters_ty =
                 ThreadCountersInstrumentationPass::CounterType,
             const std::string& trace_ty = "trace-wavestate")
        : trace(tracing), counters_type(CounterType::create(counters_ty)) {
        if (tracing) {
            trace_type = TraceType::create(trace_ty);
        }
    }

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
    std::unique_ptr<TraceType> trace_type;
    std::unique_ptr<CounterType> counters_type;
};

} // namespace hip
