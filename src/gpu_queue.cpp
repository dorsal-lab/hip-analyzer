/** \file qpu_queue.cpp
 * \brief Data-parallel execution queue implementation
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/gpu_queue.hpp"

namespace hip {

QueueInfo::QueueInfo(Instrumenter& instr, size_t elem_size, bool is_thread)
    : is_thread(is_thread), elem_size(elem_size), instr(instr) {
    computeSize();
}

void QueueInfo::computeSize() {
    auto& kernel = instr.kernelInfo();
    auto& counters = instr.data();

    auto thread_stride = kernel.basic_blocks;
    auto blocks_stride = kernel.threads_per_blocks;

    offsets_vec.reserve(thread_stride * blocks_stride);
    offsets_vec.push_back(0u);

    size_t nb_events = 0u;

    if (is_thread) {
        // Handle thread queues
        for (auto block = 0u; block < kernel.blocks; ++block) {
            for (auto thread = 0u; thread < kernel.threads_per_blocks;
                 ++thread) {

                // For each thread, compute the number of events to record
                for (auto bb = 0u; bb < kernel.basic_blocks; ++b) {
                    nb_events += counters[block * blocks_stride +
                                          thread * thread_stride + bb];
                }

                // Append to the vector of offsets
                offsets_vec.push_back(nb_events);
            }
        }
    }

    // The last element is the total size of the events, won't be used by the
    // last thread of the last block
}

size_t QueueInfo::queueSize() const { return offsets.back(); }

} // namespace hip
