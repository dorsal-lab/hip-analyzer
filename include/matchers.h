/** \file matchers.h
 * \brief AST Matcher definition
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include "clang/ASTMatchers/ASTMatchers.h"

namespace hip {

/** \brief Static matchers
 * \details These matchers do not depend on parameters, typically hip
 * intrinsics.
 */
// Geometry matchers
extern clang::ast_matchers::StatementMatcher function_call_matcher,
    geometry_matcher;

/** \brief Matcher factories
 * \details These matchers are generated based on some parameters (eg. kernel
 * name)
 */
clang::ast_matchers::DeclarationMatcher
kernelMatcher(const std::string& kernel_name);

// Kernel launch matchers

clang::ast_matchers::StatementMatcher
kernelCallMatcher(const std::string& kernel_name);

} // namespace hip
