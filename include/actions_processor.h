/** \file actions_processor.h
 * \brief Processor to chain different compiler actions and produce a single
 * output
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include "clang/Tooling/Tooling.h"

#include <functional>
#include <string>

#include "hip_instrumentation/basic_block.hpp"

namespace hip {

/** \class ActionsProcessor
 * \brief Utility class to chain different compiler actions, using the
 * intermediate result as an input to the next. The result is stored when the
 * ActionProcessor is destroyed.
 */
class ActionsProcessor {
  public:
    /**
     * \brief Constructor. Initializes the buffers.
     * \param input_file Path to the input file
     * \param output_file
     */
    ActionsProcessor(const std::string& input_file,
                     const clang::tooling::CompilationDatabase& db,
                     const std::string& output_file);

    /**
     * \brief Destructor. Commits the changes to the output file specified in
     * the constructor
     */
    ~ActionsProcessor();

    /** \fn process
     * \brief Executes an action and modifies the buffer. Does not modify the
     * buffer if the return value is an empty string.
     */
    ActionsProcessor&
    process(std::function<std::string(clang::tooling::ClangTool&)> action);

    /** \fn observe
     * \brief Performs an action, but does not modify the buffer
     */
    ActionsProcessor&
    observe(std::function<void(clang::tooling::ClangTool&)> action);

    /** \fn observe
     * \brief Performs an action on the original file, but does not modify the
     * buffer
     */
    ActionsProcessor&
    observeOriginal(std::function<void(clang::tooling::ClangTool&)> action);

  private:
    std::string input_file_path;
    std::string output_file;
    std::string buffer;

    const clang::tooling::CompilationDatabase& db;
};

namespace actions {

/** \class Action
 * \brief A generic action to be performed on the source code. The output of the
 * call to its operator() must produce a valid, compilable code to ensure that
 * following actions will work.
 */
class Action {
  public:
    virtual std::string operator()(clang::tooling::ClangTool& tool) = 0;
};

/** \class DuplicateKernel
 * \brief Duplicate a function declaration which has the name original_name. The
 * rollback argument indicates to the Instrumentation Generator that alternate
 * HIP memory allocations method will be used ( \see hip::HipMemoryManager)
 */
class DuplicateKernel : public Action {
  public:
    DuplicateKernel(const std::string& original_name,
                    const std::string& new_name, int& err,
                    bool rollback = false)
        : original(original_name), new_kernel(new_name), err(err),
          rollback(rollback) {}

    virtual std::string operator()(clang::tooling::ClangTool& tool) override;

  private:
    const std::string& original;
    const std::string& new_kernel;
    int& err;
    bool rollback;
};

/** \class InstrumentBasicBlocks
 * \brief Perform a CFG analysis to add instrumentation points for basic blocks
 */
class InstrumentBasicBlocks : public Action {
  public:
    InstrumentBasicBlocks(const std::string& kernel_name,
                          std::vector<hip::BasicBlock>& blocks, int& err)
        : kernel(kernel_name), blocks(blocks), err(err),
          instrumented_kernel(getInstrumentedKernelName(kernel_name)) {}

    virtual std::string operator()(clang::tooling::ClangTool& tool) override;

    /** \fn getInstrumentedKernelName
     * \brief Returns the name of the kernel with instrumented bblocks
     */
    static std::string
    getInstrumentedKernelName(const std::string& kernel_name);

  private:
    const std::string& kernel;
    std::string instrumented_kernel;
    std::vector<hip::BasicBlock>& blocks;
    int& err;
};

class TraceBasicBlocks : public Action {
  public:
    TraceBasicBlocks(const std::string& kernel_name,
                     std::vector<hip::BasicBlock>& blocks, int& err)
        : kernel(kernel_name), blocks(blocks), err(err) {}

    virtual std::string operator()(clang::tooling::ClangTool& tool) override;

    /** \fn getInstrumentedKernelName
     * \brief Returns the name of the kernel with traced bblocks
     */
    static std::string
    getInstrumentedKernelName(const std::string& kernel_name);

  private:
    const std::string& kernel;
    std::vector<hip::BasicBlock>& blocks;
    int& err;
};

/** \class AnalyzeIR
 * \brief Compiles the kernel to LLVM IR and performs a more fine-grained
 * analysis of the basic blocks, to extract more precise informations (flops,
 * ld, st). Adjusts the basic block database
 */
class AnalyzeIR : public Action {
  public:
    AnalyzeIR(const std::string& kernel_name,
              std::vector<hip::BasicBlock>& blocks, int& err)
        : kernel(kernel_name), blocks(blocks), err(err) {}

    virtual std::string operator()(clang::tooling::ClangTool& tool) override;

  private:
    const std::string& kernel;
    std::vector<hip::BasicBlock>& blocks;
    int& err;
};

/** \class ReplaceKernelCall
 * \brief Replaces calls to the original kernel to a new one `new_kernel`
 */
class ReplaceKernelCall : public Action {
  public:
    ReplaceKernelCall(const std::string& original,
                      const std::string& new_kernel, int& err)
        : original(original), new_kernel(new_kernel), err(err) {}

    virtual std::string operator()(clang::tooling::ClangTool& tool) override;

  private:
    const std::string& original;
    const std::string& new_kernel;
    int& err;
};

/** \class DuplicateKernelCall
 * \brief Duplicates each call to the kernel `original` with a second call to
 * `new_kernel`. The original one will be executed after the new one.
 */
class DuplicateKernelCall : public Action {
  public:
    DuplicateKernelCall(const std::string& original,
                        const std::string& new_kernel, int& err)
        : original(original), new_kernel(new_kernel), err(err) {}

    virtual std::string operator()(clang::tooling::ClangTool& tool) override;

  private:
    const std::string& original;
    const std::string& new_kernel;
    int& err;
};

/** \class InstrumentKernelCall
 * \brief Add all the necessary hiptrace instrumentation to an instrumented
 * kernel call
 *
 * \details The rollback parameter in the constructor indicates to
 * the instrumenter that the memory has to be reset to its initial value before
 * the start of the kernel, which is necessary to ensure that each execution is
 * deterministic
 */
class InstrumentKernelCall : public Action {
  public:
    InstrumentKernelCall(const std::string& kernel,
                         const std::vector<hip::BasicBlock>& blocks, int& err,
                         bool rollback = false)
        : kernel(kernel), blocks(blocks), err(err), do_rollback(rollback),
          instrumented_kernel(
              InstrumentBasicBlocks::getInstrumentedKernelName(kernel)) {}

    virtual std::string operator()(clang::tooling::ClangTool& tool) override;

  private:
    const std::string& kernel;
    std::string instrumented_kernel;
    const std::vector<hip::BasicBlock>& blocks;
    int& err;
    bool do_rollback = false;
};

class TraceKernelCall : public Action {
  public:
    TraceKernelCall(const std::string& kernel,
                    const std::vector<hip::BasicBlock>& blocks, int& err)
        : kernel(kernel), blocks(blocks), err(err),
          instrumented_kernel(
              TraceBasicBlocks::getInstrumentedKernelName(kernel)) {}

    virtual std::string operator()(clang::tooling::ClangTool& tool) override;

  private:
    const std::string& kernel;
    std::string instrumented_kernel;
    const std::vector<hip::BasicBlock>& blocks;
    int& err;
};

} // namespace actions

} // namespace hip
