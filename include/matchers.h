/** \file matchers.h
 * \brief AST Matcher definition
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include "clang/ASTMatchers/ASTMatchers.h"

namespace hip {

extern clang::ast_matchers::StatementMatcher function_call_matcher,
    geometry_matcher;

}
