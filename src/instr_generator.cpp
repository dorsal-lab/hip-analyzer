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

    auto begin_loc = expr->getBeginLoc();
    auto end_loc = expr->getEndLoc().getLocWithOffset(1);

    if (begin_loc.isInvalid() || end_loc.isInvalid()) {

        throw std::runtime_error("getExprText : Could not cast expr, unhandled "
                                 "geometry declaration");
    }

    auto text = clang::Lexer::getSourceText(
                    clang::CharSourceRange::getTokenRange({begin_loc, end_loc}),
                    sm, clang::LangOptions())
                    .str();

    // Handle strange edge case where the following ',' could be included in the
    // token
    if (text.back() == ',') {
        text.pop_back();
    }
    return text;
}

void InstrGenerator::setGeometry(const clang::CallExpr& kernel_call,
                                 const clang::SourceManager& source_manager) {
    auto blocks_expr = kernel_call.getArg(0);
    blocks_expr->dump();

    blocks = getExprText(blocks_expr, source_manager);
    llvm::errs() << blocks << '\n';

    auto threads_expr = kernel_call.getArg(1);
    threads_expr->dump();

    threads = getExprText(threads_expr, source_manager);
    llvm::errs() << threads << '\n';
}

std::string InstrGenerator::generateBlockCode(unsigned int id) const {
    std::stringstream ss;
    ss << "/* BB " << id << " (" << bb_count << ") */" << '\n';

    ss << "_bb_counters[" << bb_count << "][threadIdx.x] += 1;\n";

    return ss.str();
}

std::string InstrGenerator::generateIncludes() const {
    return "#include \"hip_instrumentation/hip_instrumentation.hpp\"\n";
}

std::string InstrGenerator::generateInstrumentationParms() const {
    std::stringstream ss;
    ss << ",/* Extra params */ uint8_t* _instr_ptr";

    return ss.str();
}

std::string InstrGenerator::generateInstrumentationLocals() const {
    std::stringstream ss;

    ss << "\n/* Instrumentation locals */\n";

    ss << "__shared__ uint8_t _bb_counters[" << bb_count << "][64];\n"
       << "unsigned int _bb_count = " << bb_count << ";\n"
       << "#pragma unroll"
          "\nfor(auto i = 0u; i < _bb_count; ++i) { "
          "_bb_counters[i][threadIdx.x] = 0; }\n";

    return ss.str();
}

std::string InstrGenerator::generateInstrumentationCommit() const {
    std::stringstream ss;

    ss << "/* Finalize instrumentation */\n";

    // Print output

    ss << "    int id = threadIdx.x;\n"
          "    for (auto i = 0u; i < _bb_count; ++i) {\n"
          "        _instr_ptr[blockIdx.x * blockDim.x * _bb_count + "
          "threadIdx.x * _bb_count + i] = _bb_counters[i][threadIdx.x]\n;"
          "    }\n";

    return ss.str();
}

std::string InstrGenerator::generateInstrumentationInit() const {
    std::stringstream ss;

    ss << "/* Instrumentation variables, hipMalloc, etc. */\n\n";

    ss << "hip::KernelInfo _" << kernel_name << "_info(\"" << kernel_name
       << "\", " << bb_count << ", " << blocks << ", " << threads << ");\n";

    ss << "hip::Instrumenter _" << kernel_name << "_instr(_" << kernel_name
       << "_info);\n";

    ss << "auto _" << kernel_name << "_ptr = _" << kernel_name
       << "_instr.toDevice();\n\n";

    return ss.str();
}

std::string InstrGenerator::generateInstrumentationLaunchParms() const {
    std::stringstream ss;

    ss << ",/* Extra parameters for kernel launch ( " << bb_count
       << " )*/ (uint8_t*) _" << kernel_name << "_ptr";

    return ss.str();
}

std::string InstrGenerator::generateInstrumentationFinalize() const {
    std::stringstream ss;

    ss << "\n\n/* Finalize instrumentation : copy back data */\n";

    ss << "hip::check(hipDeviceSynchronize());\n"
       << "_" << kernel_name << "_instr.fromDevice(_" << kernel_name
       << "_ptr);\n";

    return ss.str();
}

// ----- MultipleExecutionInstrGenerator ----- //

std::string MultipleExecutionInstrGenerator::generatePostKernel() const {
    return "";
}

}; // namespace hip
