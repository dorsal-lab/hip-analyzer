/** \file block_db.cpp
 * \brief Load a database and prints it in normalized form
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation.hpp"

#include <iostream>

int main() {
    hip::KernelInfo ki("test", 4, dim3(1), dim3(1));
    hip::Instrumenter instrumenter(ki);

    auto blocks = instrumenter.loadDatabase();

    if (blocks.empty()) {
        throw std::runtime_error("No database found");
    }

    std::cout << "Non-normalized\n";
    for (const auto& block : blocks) {
        std::cout << block.id << " :\n  " << block.flops << '\n';
    }

    std::cout << "\nNormalized\n";

    for (const auto& block : hip::BasicBlock::normalized(blocks)) {
        std::cout << block.id << " :\n  " << block.flops << '\n';
    }
}
