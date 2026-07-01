/** \file count_allocations.cpp
 * \brief Load a binary hiptrace file, and for each trace packet compute the
 * number of allocation performed by each wavefront
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/gpu_queue.hpp"
#include "hip_instrumentation/hip_instrumentation.hpp"
#include "hip_instrumentation/hip_trace_manager.hpp"

#include <bit>
#include <iostream>
#include <map>

#include "llvm/Support/CommandLine.h"

static llvm::cl::opt<std::string> input_file(llvm::cl::Positional,
                                             llvm::cl::desc("Hiptrace file"),
                                             llvm::cl::value_desc("hiptrace"),
                                             llvm::cl::Required);

static llvm::cl::opt<std::string> output_file(llvm::cl::Positional,
                                              llvm::cl::desc("Output file"),
                                              llvm::cl::value_desc("output"),
                                              llvm::cl::init("out.csv"));

class Counter {
  public:
    Counter(hip::HipTraceFile& trace, std::ofstream& out)
        : trace(trace), out(out) {}

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

    auto i = 0u;


    auto instance_id = 0u;

    for (auto& ca_reg : registries) {
        auto& reg = ca_reg.reg;

        std::cout << "\tCU " << i << ", " << reg.buffer_count << " buffers\n";

        for (auto subbuffer_id = 0ull; subbuffer_id < reg.buffer_count;
             ++subbuffer_id) {

            auto* ptr = reinterpret_cast<hip::SubBuffer*>(
                reinterpret_cast<std::byte*>(reg.begin) +
                subbuffer_id * reg.buffer_size);

            uint32_t owner = (ptr->owner & 0xffffffff);

            auto* event = reinterpret_cast<hip::WaveState*>(ptr->data);
            auto n_iter = 0u;

            while (reinterpret_cast<void*>(event) <
                       reinterpret_cast<void*>(ptr + reg.buffer_size) &&
                   (event->hw_id != 0xbebebebe)) {
                ++event;

                auto active_threads = std::popcount(event->exec);

                if (event->bb == 3) {
                    ++n_iter;
                    out << instance_id << ',' << n_iter << ',' << active_threads << '\n';
                } else {
                    n_iter = 0;
                    ++instance_id;
                }

            }

            ++instance_id;
        }

        std::cout << "\n";

        ++i;
    }

    delete alloc;
}

void Counter::process() {

    out << "instance_id,iter,active_threads" << '\n';
    while (!trace.done()) {
        auto payload = trace.getNext();

        if (payload_id % 3 != 1) {
            ++payload_id;
            continue;
        }

        std::visit(
            [this](auto&& var) { visitor(std::move(var)); },
            payload);

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
