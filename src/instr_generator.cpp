/** \file instr_generator.cpp
 * \brief Kernel CFG Instrumentation code generation tools
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "instr_generator.h"

#include "clang/AST/ExprCXX.h"
#include "clang/Lex/Lexer.h"

#include <sstream>

namespace hip {

std::string getExprText(const clang::Expr* expr,
                        const clang::SourceManager& sm) {
    // TODO : This is unsafe if the expression is not a functional cast. To be
    // investigated
    auto cast = static_cast<const clang::CXXFunctionalCastExpr*>(expr);
    auto begin_loc = cast->getBeginLoc();
    auto end_loc = cast->getEndLoc().getLocWithOffset(1);

    return clang::Lexer::getSourceText(
               clang::CharSourceRange::getCharRange({begin_loc, end_loc}), sm,
               clang::LangOptions())
        .str();
}

void InstrGenerator::setGeometry(const clang::CallExpr& kernel_call,
                                 const clang::SourceManager& source_manager) {
    auto thread_expr = kernel_call.getArg(0);
    thread_expr->dump();

    threads_expr = getExprText(thread_expr, source_manager);

    auto blocks_expr = kernel_call.getArg(1);
    blocks_expr->dump();

    blocks_expr = getExprText(blocks_expr, source_manager);
}

std::string InstrGenerator::generateBlockCode(unsigned int id) const {
    std::stringstream ss;
    ss << "/* BB " << id << " (" << bb_count << ") */" << '\n';

    ss << "_bb_counters[" << bb_count << "][threadIdx.x] += 1;\n";

    return ss.str();
}

std::string InstrGenerator::generateInstrumentationParms() const {
    std::stringstream ss;
    ss << "/* Extra params */";
    return ss.str();
}

std::string InstrGenerator::generateInstrumentationLocals() const {
    std::stringstream ss;

    // The opening brace needs to be added to the code, in order to get "inside"
    // the kernel body. I agree that this feels like a kind of hack, but adding
    // an offset to a SourceLocation sounds tedious
    ss << "{\n/* Instrumentation locals */\n";

    ss << "__shared__ uint32_t _bb_counters[" << bb_count << "][64];\n"
       << "unsigned int _bb_count = " << bb_count << ";\n";

    // TODO (maybe) : Lexer::getIndentationForLine

    // TODO : init counters to 0

    return ss.str();
}

std::string InstrGenerator::generateInstrumentationCommit() const {
    std::stringstream ss;

    ss << "/* Finalize instrumentation */\n";

    // Print output
    ss << "   int id = threadIdx.x;\n"
          "for (auto i = 0u; i < _bb_count; ++i) {\n"
          "    printf(\" %d %d : %d\\n \", id, i, "
          "_bb_counters[i][threadIdx.x]);"
          "}\n";

    return ss.str();
}

std::string InstrGenerator::generateInstrumentationInit() const {
    std::stringstream ss;

    // Probably best to link a library etc;

    ss << "/* Instrumentation variables, hipMalloc, etc. */\n\n";

    return ss.str();
}

std::string InstrGenerator::generateInstrumentationLaunchParms() const {
    std::stringstream ss;

    ss << "/* Extra parameters for kernel launch ( " << bb_count << " )*/";

    return ss.str();
}

std::string InstrGenerator::generateInstrumentationFinalize() const {
    std::stringstream ss;

    ss << "\n\n/* Finalize instrumentation : copy back data */\n";

    return ss.str();
}

}; // namespace hip
