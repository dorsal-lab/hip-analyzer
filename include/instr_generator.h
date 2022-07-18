/** \file instr_generator.cpp
 * \brief Kernel CFG Instrumentation code generation tools
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include "clang/AST/Expr.h"
#include "clang/Basic/SourceManager.h"

#include <string>

namespace hip {

/** \struct InstrGenerator
 * \brief Instrumentation code generator. Holds data about the
 * kernel (number of basic blocks, launch geometry, ...)
 */
struct InstrGenerator {

    void setGeometry(const clang::CallExpr& kernel_call,
                     const clang::SourceManager& source_manager);

    virtual void setKernelDecl(clang::FunctionDecl* decl) {}

    // ----- Device-side instrumentation ----- //

    /** \brief Instrumentation for each basic block
     */
    virtual std::string generateBlockCode(unsigned int id) const;

    /** \brief  Additional includes for the runtime
     */
    virtual std::string generateIncludes() const;

    /** \brief  Additional parameters for the instrumented kernel
     */
    virtual std::string generateInstrumentationParms() const;

    /** \brief Local variables for instrumentation (e.g. local counters)
     */
    virtual std::string generateInstrumentationLocals() const;

    /** \brief Final code to be executed by the kernel : commit to device memory
     */
    virtual std::string generateInstrumentationCommit() const;

    // ----- Host-side instrumentation ----- //

    /** \brief Device-side allocation & init of variables
     */
    virtual std::string generateInstrumentationInit() const;

    /** \brief Additional kernel launch parameters
     */
    virtual std::string generateInstrumentationLaunchParms() const;

    /** \brief Final code to be executed, right after execution
     */
    virtual std::string generateInstrumentationFinalize() const;

    /** \brief Code to be added after the kernel definition, e.g. a second
     * kernel
     */
    virtual std::string generatePostKernel() const { return ""; }

    // ----- Members ----- //

    /** \brief Number of instrumented basic blocks
     */
    unsigned int bb_count = 0u;

    /** \brief Geometry expressions
     */
    std::string threads, blocks;

    std::string kernel_name;
};

struct MultipleExecutionInstrGenerator : public InstrGenerator {
    virtual std::string generatePostKernel() const override;
};

} // namespace hip
