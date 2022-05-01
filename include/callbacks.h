/** \file matchers.h
 * \brief AST Matcher callbacks
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

#include <memory>

namespace hip {

// Printers

std::unique_ptr<clang::ast_matchers::MatchFinder::MatchCallback>
makeFunPrinter();

std::unique_ptr<clang::ast_matchers::MatchFinder::MatchCallback>
makeCfgInstrumenter(const std::string& name, const std::string& output_file);

std::unique_ptr<clang::ast_matchers::MatchFinder::MatchCallback>
makeBaseInstrumenter(const std::string& kernel);

} // namespace hip
