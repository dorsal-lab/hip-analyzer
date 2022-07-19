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

    std::cout << buffer << "\n\n";

    return *this;
}
