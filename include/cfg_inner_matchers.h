/** \file matchers.h
 * \brief AST Matcher definition
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

#include "clang/Analysis/CFG.h"

#include <memory>

namespace hip {

// Matchers

/** \brief Matches any node performing a floating point binary operator
 */
extern clang::ast_matchers::StatementMatcher flopMatcher;

// Runners

/** \fn countFlops
 * \brief Returns the flops in a CFG block
 */
unsigned int countFlops(const clang::CFGBlock* block,
                        clang::ASTContext& context);

/** \fn countDerefs
 * \brief Returns the number of dereferences in a CFG block
 */
unsigned int countDerefs(const clang::CFGBlock* block,
                         clang::ASTContext& context);

} // namespace hip
