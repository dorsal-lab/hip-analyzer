/** \file actions_processor.h
 * \brief Processor to chain different compiler actions and produce a single
 * output
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "actions_processor.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace hip {

ActionsProcessor::ActionsProcessor(const std::string& input_file,
                                   const clang::tooling::CompilationDatabase& d,
                                   const std::string& output_file_path)
    : input_file_path(clang::tooling::getAbsolutePath(input_file)), db(d),
      output_file(output_file_path) {

    std::stringstream ss;
    ss << std::ifstream(input_file).rdbuf();
    buffer = ss.str();
}

ActionsProcessor::~ActionsProcessor() {
    std::ofstream out(output_file);
    out << buffer;
    out.close();
}

ActionsProcessor& ActionsProcessor::process(
    std::function<std::string(clang::tooling::ClangTool&)> action) {
    clang::tooling::ClangTool tool(db, {input_file_path});

    tool.mapVirtualFile(input_file_path, buffer);

    auto ret = action(tool);

    if (!ret.empty()) {
        buffer = std::move(ret);
    }

    return *this;
}

ActionsProcessor& ActionsProcessor::observe(
    std::function<void(clang::tooling::ClangTool&)> action) {
    clang::tooling::ClangTool tool(db, {input_file_path});

    tool.mapVirtualFile(input_file_path, buffer);

    action(tool);

    return *this;
}

ActionsProcessor& ActionsProcessor::observeOriginal(
    std::function<void(clang::tooling::ClangTool&)> action) {
    clang::tooling::ClangTool tool(db, {input_file_path});

    action(tool);

    return *this;
}

namespace actions {

std::string DuplicateKernel::operator()(clang::tooling::ClangTool& tool) {
    // TODO
}

} // namespace actions

} // namespace hip
