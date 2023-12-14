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

std::map<uint32_t, std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>>
    occurences;

template <typename T> void visitor(T&&);

template <>
void visitor(hip::HipTraceManager::CountersQueuePayload<
             hip::HipTraceManager::ThreadCounters>&& payload) {
    throw std::runtime_error("Unexpected thread counters");
}

template <>
void visitor(hip::HipTraceManager::CountersQueuePayload<
             hip::HipTraceManager::WaveCounters>&& payload) {
    std::cerr << "Wave counters\n";

    auto& [vec, ki, stamp, interval] = payload;
    hip::WaveCounterInstrumenter instr(std::move(vec), ki);
}

template <> void visitor(hip::HipTraceManager::EventsQueuePayload&& payload) {
    auto& [buffer, queue_info] = payload;

    auto* events =
        reinterpret_cast<const hip::WaveState*>(queue_info.events().data());
    auto& offsets = queue_info.offsets();

    auto num_producers = queue_info.parallelism();

    for (auto i = 0u; i < num_producers; ++i) {
        std::optional<std::reference_wrapper<const hip::WaveState>> last;

        for (auto j = offsets[i]; j < offsets[i + 1]; ++j) {
            auto& event = events[j];

            if (event.bb == 0) {
                break;
            } else if (last) {
                // Not the first event
                uint64_t duration = event.stamp - last->get().stamp;

                // std::cout << last->get().bb << ' ' << last->get().stamp << '
                // '
                //           << std::popcount(last->get().exec) << '\n';

                if (duration > (1ull << 32)) {
                    // Most probably an error : ignore
                    std::cerr << "BB : " << event.bb << "Ignored, duration "
                              << duration << '\n';
                } else {
                    auto& bb = occurences[last->get().bb];

                    auto active_threads =
                        static_cast<uint64_t>(std::popcount(last->get().exec));

                    double weight = 64. / static_cast<double>(active_threads);

                    std::get<0>(bb) += duration;
                    std::get<1>(bb) += static_cast<uint64_t>(
                        static_cast<double>(duration) * weight);
                    std::get<2>(bb) += duration * (65 - active_threads);
                    std::get<3>(bb) += active_threads;
                }
            }

            last.emplace(event);
        }
    }
}

template <>
void visitor(hip::HipTraceManager::GlobalMemoryEventsQueuePayload&& payload) {
    throw std::runtime_error("Unexpected global memory events");
}

int main(int argc, char** argv) {
    llvm::cl::ParseCommandLineOptions(argc, argv);

    std::cerr << "Reading trace file " << input_file.getValue() << '\n';

    hip::HipTraceFile trace{input_file.getValue()};

    while (!trace.done()) {
        auto payload = trace.getNext();

        std::visit([](auto&& var) { visitor(std::move(var)); }, payload);
    }

    std::cout
        << "bb,duration,weighted_duration,linear_weighted,active_threads\n";
    for (auto& [bb, contents] : occurences) {
        std::cout << bb << ',' << std::get<0>(contents) << ','
                  << std::get<1>(contents) << ',' << std::get<2>(contents)
                  << ',' << std::get<3>(contents) << '\n';
    }
}
