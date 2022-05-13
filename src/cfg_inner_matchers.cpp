/** \file cfg_inner_matchers.cpp
 * \brief Matchers & callbacks for AST analysis inside a CFG Block
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "cfg_inner_matchers.h"

#include <iostream>

using namespace clang;
using namespace clang::ast_matchers;

// Finders

class BasicCounter : public clang::ast_matchers::MatchFinder::MatchCallback {
  public:
    virtual void
    run(const clang::ast_matchers::MatchFinder::MatchResult& Result) override {
        // Should work, right ?

        if (const auto* match =
                Result.Nodes.getNodeAs<clang::Stmt>("cxxOperatorCallExpr")) {

            std::cout << "Counter match (cxxOperatorCallExpr): \n";
            ++count;
            match->dump();
        } else if (const auto* match =
                       Result.Nodes.getNodeAs<clang::Stmt>("binaryOperator")) {

            std::cout << "Counter match (binary op): \n";
            ++count;
            match->dump();
        }
    }

    unsigned int getCount() const { return count; }

  private:
    unsigned int count = 0u;
};

namespace hip {

clang::ast_matchers::StatementMatcher flopMatcher =
    anyOf(cxxOperatorCallExpr(hasAnyOperatorName("+", "-", "*", "/"))
              .bind("cxxOperatorCallExpr"),
          binaryOperator(hasAnyOperatorName("+", "-", "*", "/"))
              .bind("binaryOperator"));

unsigned int countFlops(const clang::CFGBlock* block,
                        clang::ASTContext& context) {
    BasicCounter counter;
    MatchFinder finder;

    finder.addMatcher(hip::flopMatcher, &counter);

    for (const auto& node : *block) {
        if (const auto cfg_stmt = node.getAs<clang::CFGStmt>()) {
            finder.match(clang::DynTypedNode::create(*cfg_stmt->getStmt()),
                         context);
        }
    }

    return counter.getCount();
}

} // namespace hip
