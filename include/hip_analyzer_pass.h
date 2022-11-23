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

struct CfgInstrumentationPass
    : public llvm::PassInfoMixin<CfgInstrumentationPass> {
    static const std::string instrumented_prefix;
    static const std::string utils_path;

    CfgInstrumentationPass() {}

    llvm::PreservedAnalyses run(llvm::Module& mod,
                                llvm::ModuleAnalysisManager& modm);

    /** \fn isInstrumentableKernel
     * \brief Returns whether a function is a kernel that will be
     * instrumented (todo?)
     */
    bool isInstrumentableKernel(llvm::Function& f);

    /** \fn addParams
     * \brief Clone function with new parameters
     */
    bool addParams(llvm::Function& f, llvm::Function& original_function);

    /** \fn instrumentFunction
     * \brief Add CFG counters instrumentation to the compute kernel
     *
     * \param f Kernel
     * \param original_function Original (non-instrumented) kernel
     */
    bool instrumentFunction(llvm::Function& f,
                            llvm::Function& original_function,
                            llvm::FunctionAnalysisManager& fm);

    void linkModuleUtils(llvm::Module& mod);

    static llvm::Type* getCounterType(llvm::LLVMContext& context) {
        return llvm::Type::getInt8Ty(context);
    }
};

struct TracingPass : public llvm::PassInfoMixin<TracingPass> {
    TracingPass() {}

    llvm::PreservedAnalyses run(llvm::Module& mod,
                                llvm::ModuleAnalysisManager& modm);

    bool instrumentFunction(llvm::Function& f);
};

/** \struct HostPass
 * \brief The Host pass is responsible for adding device stubs for the new
 * instrumented kernels.
 *
 */
struct HostPass : public llvm::PassInfoMixin<HostPass> {
    HostPass() {}

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
    llvm::Function& addCountersDeviceStub(llvm::Function& f_original) const;

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
};

} // namespace hip
