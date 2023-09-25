/** \file read_trace.cpp
 * \brief Load a binary hiptrace file, and displays the contents
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/hip_instrumentation.hpp"
#include "hip_instrumentation/hip_trace_manager.hpp"

#include <iostream>

#include "llvm/Support/CommandLine.h"

static llvm::cl::opt<std::string> input_file(llvm::cl::Positional,
                                             llvm::cl::desc("Hiptrace file"),
                                             llvm::cl::value_desc("hiptrace"),
                                             llvm::cl::Required);

template <class> inline constexpr bool always_false_v = false;

template <typename T> void visitor(T&&);

template <>
void visitor(hip::HipTraceManager::CountersQueuePayload<
             hip::HipTraceManager::ThreadCounters>&& payload) {
    std::cout << "Thread counters\n";

    auto& [vec, ki, stamp, interval] = payload;
    hip::ThreadCounterInstrumenter instr(std::move(vec), ki);
}

template <>
void visitor(hip::HipTraceManager::CountersQueuePayload<
             hip::HipTraceManager::WaveCounters>&& payload) {
    std::cout << "Wave counters\n";

    auto& [vec, ki, stamp, interval] = payload;
    hip::WaveCounterInstrumenter instr(std::move(vec), ki);
}

template <> void visitor(hip::HipTraceManager::EventsQueuePayload&& payload) {
    std::cout << "Events\n";
}

int main(int argc, char** argv) {
    llvm::cl::ParseCommandLineOptions(argc, argv);

    std::cout << "Reading trace file " << input_file.getValue() << '\n';

    hip::HipTraceFile trace{input_file.getValue()};

    while (!trace.done()) {
        auto payload = trace.getNext();

        std::visit([](auto&& var) { visitor(std::move(var)); }, payload);
    }
}
