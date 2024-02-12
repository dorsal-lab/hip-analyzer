/** \file read_trace.cpp
 * \brief Load a binary hiptrace file, and displays the contents
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/gpu_queue.hpp"
#include "hip_instrumentation/hip_instrumentation.hpp"
#include "hip_instrumentation/hip_trace_manager.hpp"

#include <iostream>

#include "llvm/Support/CommandLine.h"

static llvm::cl::opt<std::string> input_file(llvm::cl::Positional,
                                             llvm::cl::desc("Hiptrace file"),
                                             llvm::cl::value_desc("hiptrace"),
                                             llvm::cl::Required);

static llvm::cl::opt<bool> do_dump("dump-counters",
                                   llvm::cl::desc("Print counters"));

static llvm::cl::opt<bool>
    try_alloc("try-alloc",
              llvm::cl::desc("Try to allocate a queue when given counters"));

template <typename T> void visitor(T&&);

template <>
void visitor(hip::HipTraceManager::CountersQueuePayload<
             hip::HipTraceManager::ThreadCounters>&& payload) {
    std::cout << "Thread counters\n";

    auto& [vec, ki, stamp, interval] = payload;
    hip::ThreadCounterInstrumenter instr(std::move(vec), ki);

    if (do_dump) {
        for (auto& counter : instr.getVec()) {
            std::cout << '\t' << static_cast<uint32_t>(counter) << '\n';
        }
    }

    if (try_alloc) {
        auto queue = hip::QueueInfo::thread<hip::Event>(instr);

        auto* offsets = queue.allocOffsets();
        auto* buffer = queue.allocBuffer();

        std::cout << "Successfully allocated buffer, size " << queue.totalSize()
                  << '\n';

        hip::check(hipFree(offsets));
        hip::check(hipFree(buffer));
    }
}

template <>
void visitor(hip::HipTraceManager::CountersQueuePayload<
             hip::HipTraceManager::WaveCounters>&& payload) {
    std::cout << "Wave counters\n";

    auto& [vec, ki, stamp, interval] = payload;
    hip::WaveCounterInstrumenter instr(std::move(vec), ki);

    if (do_dump) {
        for (auto& counter : instr.getVec()) {
            std::cout << '\t' << static_cast<uint32_t>(counter) << '\n';
        }
    }

    if (try_alloc) {
        auto queue = hip::QueueInfo::wave<hip::WaveState>(instr);
        auto* offsets = queue.allocOffsets();
        auto* buffer = queue.allocBuffer();

        std::cout << "Successfully allocated buffer, size " << queue.totalSize()
                  << '\n';

        hip::check(hipFree(offsets));
        hip::check(hipFree(buffer));
    }
}

template <> void visitor(hip::HipTraceManager::EventsQueuePayload&& payload) {
    std::cout << "Events\n";
}

template <>
void visitor(hip::HipTraceManager::GlobalMemoryEventsQueuePayload&& payload) {
    std::cout << "Global memory events\n";
}

template <>
void visitor(hip::HipTraceManager::ChunkAllocatorEventsQueuePayload&& payload) {
    auto& [alloc, stamp, registry, curr_id] = payload;

    std::cout << "Chunk allocator events, " << registry.buffer_count
              << " buffers of " << registry.buffer_size << " bytes\n";

    delete alloc;
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
