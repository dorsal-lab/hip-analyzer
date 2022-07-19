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
