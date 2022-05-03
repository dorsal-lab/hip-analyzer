/** \file instr_generator.cpp
 * \brief Kernel CFG Instrumentation code generation tools
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "instr_generator.h"

#include <sstream>

namespace hip {

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
