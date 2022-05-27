/** \file matchers.h
 * \brief AST Matcher callbacks
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include "hip_instrumentation/basic_block.hpp"

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

#include <memory>

namespace hip {

// Printers

std::unique_ptr<clang::ast_matchers::MatchFinder::MatchCallback>
makeFunPrinter();

std::unique_ptr<clang::ast_matchers::MatchFinder::MatchCallback>
makeCfgInstrumenter(const std::string& name, const std::string& output_file,
                    std::vector<hip::BasicBlock>& blocks);

std::unique_ptr<clang::ast_matchers::MatchFinder::MatchCallback>
makeCudaCallInstrumenter(const std::string& kernel,
                         const std::string& output_file);

} // namespace hip
