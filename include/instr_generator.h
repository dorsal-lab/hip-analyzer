/** \file instr_generator.cpp
 * \brief Kernel CFG Instrumentation code generation tools
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include <string>

namespace hip {

struct InstrGenerator {

    std::string generateBlockCode(unsigned int id) const;

    std::string generateInstrumentationParms() const;

    std::string generateInstrumentationLocals() const;

    std::string generateInstrumentationCommit() const;

    std::string generateInstrumentationInit() const;

    std::string generateInstrumentationLaunchParms() const;

    std::string generateInstrumentationFinalize() const;

    unsigned int bb_count = 0u;
    std::string threads_expr, block_expr;
};

} // namespace hip
