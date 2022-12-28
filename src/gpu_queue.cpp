/** \file qpu_queue.cpp
 * \brief Data-parallel execution queue implementation
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/gpu_queue.hpp"

#include <numeric>

namespace hip {

QueueInfo::QueueInfo(Instrumenter& instr, size_t elem_size, bool is_thread,
                     const std::string& type_desc, const std::string& type_name,
                     size_t extra_size)
    : is_thread(is_thread), elem_size(elem_size), instr(instr),
      type_desc(type_desc), type_name(type_name), extra_size(extra_size) {
    computeSize();
}

constexpr auto warpSize = 64u;

void QueueInfo::computeSize() {
    auto& kernel = instr.kernelInfo();
    auto& counters = instr.data();

    if (counters.empty()) {
        throw std::runtime_error("hip::QueueInfo::computeSize() : Empty "
                                 "counters, have they been moved out ?");
    }

    auto thread_stride = kernel.basic_blocks;
    auto blocks_stride = kernel.total_threads_per_blocks;

    offsets_vec.reserve(thread_stride * blocks_stride * kernel.basic_blocks);
    offsets_vec.push_back(0u);

    size_t nb_events = 0u;

    if (is_thread) {
        // Handle thread queues
        for (auto block = 0u; block < kernel.total_blocks; ++block) {
            for (auto thread = 0u; thread < kernel.total_threads_per_blocks;
                 ++thread) {

                // For each thread, compute the number of events to record
                for (auto bb = 0u; bb < kernel.basic_blocks; ++bb) {
                    nb_events += counters[block * blocks_stride +
                                          thread * thread_stride + bb];
                }

                nb_events += extra_size;

                // Append to the vector of offsets
                offsets_vec.push_back(nb_events);
            }
        }
    } else {
        // It's trickier for Wavefronts : there is a wavefront for every group
        // of 64 threads in the same block, so we have to separate each
        // wavefront when iterating through the threads. A new block is always a
        // new wavefront.

        std::vector<size_t> max_per_bblock{kernel.basic_blocks, 0u};

        auto wave = 0u;

        auto commit_wave = [&]() {
            nb_events += std::accumulate(max_per_bblock.begin(),
                                         max_per_bblock.end(), extra_size);
            offsets_vec.push_back(nb_events);

            max_per_bblock.assign(kernel.basic_blocks, 0u);
            ++wave;
        };

        for (auto block = 0u; block < kernel.total_blocks; ++block) {

            for (auto thread = 0u; thread < kernel.total_threads_per_blocks;
                 ++thread) {
                if (thread % warpSize == 0 && thread != 0) {
                    commit_wave();
                }

                // Go through every basic block and register if it is a new
                // maximum
                for (auto bb = 0u; bb < kernel.basic_blocks; ++bb) {
                    auto counter_bb = counters[block * blocks_stride +
                                               thread * thread_stride + bb];
                    if (counter_bb > max_per_bblock[bb]) {
                        max_per_bblock[bb] = counter_bb;
                    }
                }
            }

            commit_wave();
        }
    }

    // The last element is the total size of the events, won't be used by the
    // last thread of the last block
}

void QueueInfo::fromDevice(void* ptr) {
    cpu_queue.resize(totalSize());
    hip::check(
        hipMemcpy(cpu_queue.data(), ptr, totalSize(), hipMemcpyDeviceToHost));
}

size_t QueueInfo::queueLength() const { return offsets_vec.back(); }

// Standard events

std::string Event::description = HipEventFields<decltype(Event::bb)>();

std::string Event::name = "hip::Event";

std::string TaggedEvent::description =
    HipEventFields<decltype(TaggedEvent::bb), decltype(TaggedEvent::stamp)>();

std::string TaggedEvent::name = "hip::TaggedEvent";

std::string WaveState::description =
    HipEventFields<decltype(WaveState::bb), decltype(WaveState::stamp),
                   decltype(WaveState::exec), decltype(WaveState::hw_id)>();

std::string WaveState::name = "hip::WaveState";

} // namespace hip
