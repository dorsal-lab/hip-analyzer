/** \file read_trace.cpp
 * \brief Load a binary hiptrace file, and displays the contents
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/gpu_queue.hpp"
#include "hip_instrumentation/hip_instrumentation.hpp"
#include "hip_instrumentation/hip_trace_manager.hpp"

#include <iostream>
#include <map>
#include <optional>

#include "llvm/Support/CommandLine.h"

static llvm::cl::opt<std::string> input_file(llvm::cl::Positional,
                                             llvm::cl::desc("Hiptrace file"),
                                             llvm::cl::value_desc("hiptrace"),
                                             llvm::cl::Required);

template <typename T> void visitor(T&&);

template <>
void visitor(hip::HipTraceManager::CountersQueuePayload<
             hip::HipTraceManager::ThreadCounters>&& payload) {
    throw std::runtime_error("Unexpected thread counters");
}

template <>
void visitor(hip::HipTraceManager::CountersQueuePayload<
             hip::HipTraceManager::WaveCounters>&& payload) {
    std::cout << "Wave counters\n";

    auto& [vec, ki, stamp, interval] = payload;
    hip::WaveCounterInstrumenter instr(std::move(vec), ki);
}

template <> void visitor(hip::HipTraceManager::EventsQueuePayload&& payload) {
    auto& [buffer, queue_info] = payload;

    auto* events =
        reinterpret_cast<const hip::WaveState*>(queue_info.events().data());
    auto& offsets = queue_info.offsets();

    auto num_producers = queue_info.parallelism();

    std::map<uint32_t, std::tuple<uint64_t, uint64_t>> occurences;

    for (auto i = 0u; i < num_producers; ++i) {
        std::optional<std::reference_wrapper<const hip::WaveState>> last;

        for (auto j = offsets[i]; j < offsets[i + 1]; ++j) {
            auto& event = events[j];

            if (event.bb == 0) {
                break;
            } else if (last) {
                // Not the first event
                uint64_t duration = event.stamp - last->get().stamp;

                if (duration > (1ull << 32)) {
                    // Most probably an error : ignore
                    std::cout << "BB : " << event.bb << "Ignored, duration "
                              << duration << '\n';
                } else {
                    auto& bb = occurences[last->get().bb];

                    std::get<0>(bb) += duration;
                    std::get<1>(bb) += std::popcount(last->get().exec);
                }
            }

            last.emplace(event);
        }
    }

    for (auto& [bb, contents] : occurences) {
        std::cout << bb << " -> " << std::get<0>(contents) << ';'
                  << std::get<1>(contents) << '\n';
    }
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
