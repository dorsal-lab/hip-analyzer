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
                     clang::tooling::ClangTool& tool,
                     const std::string& output_file);

    /**
     * \brief Destructor. Commits the changes to the output file specified in
     * the constructor
     */
    ~ActionsProcessor();

    /** \fn process
     * \brief Executes an action and modifies the buffer
     */
    ActionsProcessor&
    process(std::function<std::string(clang::tooling::ClangTool&)>& action);

  private:
    std::string input_file_path;
    std::string output_file;
    std::string buffer;

    clang::tooling::ClangTool& tool
};
