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
 * \brief A generic action to be performed
 */
class Action {
  public:
    virtual std::string operator()(clang::tooling::ClangTool& tool) = 0;
};

/** \class DuplicateKernel
 * \brief Duplicate a function declaration which has the name original_name
 */
class DuplicateKernel : public Action {
  public:
    DuplicateKernel(const std::string& original_name,
                    const std::string& new_name, int& err)
        : original(original_name), new_kernel(new_name), err(err) {}

    virtual std::string operator()(clang::tooling::ClangTool& tool) override;

  private:
    const std::string& original;
    const std::string& new_kernel;
    int& err;
};

/** \class InstrumentBasicBlocks
 * \brief Perform a CFG analysis to add instrumentation points for basic blocks
 */
class InstrumentBasicBlocks : public Action {
  public:
    InstrumentBasicBlocks(const std::string& kernel_name,
                          std::vector<hip::BasicBlock>& blocks, int& err)
        : kernel(kernel_name), blocks(blocks), err(err) {}

    virtual std::string operator()(clang::tooling::ClangTool& tool) override;

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

} // namespace actions

} // namespace hip
