/** \file count_allocations.cpp
 * \brief Load a binary hiptrace file, and for each trace packet compute the
 * number of allocation performed by each wavefront
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

static llvm::cl::opt<std::string> output_file(llvm::cl::desc("Output file"),
                                              llvm::cl::value_desc("output"),
                                              llvm::cl::init("out.csv"));

class Counter {
  public:
    Counter(hip::HipTraceFile& trace, std::ofstream& out)
        : trace(trace), out(out) {
        out << "input_trace,run_id,wave,count\n";
    }

    void process();

    template <typename T> void visitor(T&&) {
        // Basic behaviour : nothing to do
        std::cout << "Ignoring event\n";
    }

  private:
    hip::HipTraceFile& trace;
    std::ofstream& out;

    unsigned int payload_id = 0u;
};

// Specialisations
template <>
void Counter::visitor(
    hip::HipTraceManager::GlobalMemoryEventsQueuePayload&& payload) {
    std::cout << "Global memory events\n";
}

template <>
void Counter::visitor(
    hip::HipTraceManager::ChunkAllocatorEventsQueuePayload&& payload) {
    auto& [alloc, stamp, registry, curr_id] = payload;

    std::cout << "Chunk allocator events, " << registry.buffer_count
              << " buffers of " << registry.buffer_size << " bytes\n";

    delete alloc;
}

template <>
void Counter::visitor(
    hip::HipTraceManager::CUChunkAllocatorEventsQueuePayload&& payload) {
    auto& [alloc, stamp, begin_reg, end_reg] = payload;

    auto& registries = alloc->getRegistries();

    std::cout << "CU Chunk Allocators event, " << registries.size()
              << " CU-buffers of size " << registries[0].reg.buffer_size
              << '\n';

    for (auto& ca_reg : registries) {
        auto& reg = ca_reg.reg;
        for (auto subbuffer_id = 0ull; subbuffer_id < reg.buffer_count;
             ++subbuffer_id) {

            auto* ptr = reinterpret_cast<hip::ChunkAllocator::SubBuffer*>(
                reinterpret_cast<std::byte*>(reg.begin) +
                subbuffer_id * reg.buffer_size);

            std::cout << ptr->owner << ' ';
        }

        std::cout << "\n";
    }

    delete alloc;
}

void Counter::process() {
    while (!trace.done()) {
        auto payload = trace.getNext();

        std::visit([this](auto&& var) { visitor(std::move(var)); }, payload);

        ++payload_id;
    }
}

int main(int argc, char** argv) {
    llvm::cl::ParseCommandLineOptions(argc, argv);

    std::cout << "Reading trace file " << input_file.getValue() << '\n';

    hip::HipTraceFile trace{input_file.getValue()};
    std::ofstream out(output_file.getValue());

    Counter counter(trace, out);
    counter.process();
}
