/** \file qpu_queue.cpp
 * \brief Data-parallel execution queue implementation
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/gpu_queue.hpp"
#include "hip_instrumentation/hip_trace_manager.hpp"
#include "hip_instrumentation/managed_queue_info.hpp"

#include <iomanip>
#include <numeric>
#include <stdexcept>

#include "hip_analyzer_tracepoints.h"

namespace hip {

/** \fn getSizeFactor
 * \brief Determine the size factor to be used from environnment variables
 */
float getSizeFactor(float passed) {
    if (passed != 1.f) {
        return passed;
    }

    auto* env = std::getenv("HIP_ANALYZER_PADDING");

    if (env == nullptr) {
        return 1.f;
    }

    char* end;
    passed = std::strtof(env, &end);
    if (end == env) {
        std::cerr << "Could not parse HIP_ANALYZER_PADDING " << env << '\n';
        return 1.f;
    } else {
        return passed >= 1.f ? passed : 1.f;
    }
}

QueueInfo::QueueInfo(ThreadCounterInstrumenter& instr, size_t elem_size,
                     bool is_thread, const std::string& type_desc,
                     const std::string& type_name, size_t extra_size,
                     float size_factor)
    : instr(&instr), is_thread(is_thread), elem_size(elem_size),
      extra_size(extra_size), size_factor(getSizeFactor(size_factor)),
      type_desc(type_desc), type_name(type_name) {

    if (instr.getVec().empty()) {
        throw std::runtime_error("hip::QueueInfo::QueueInfo() : Empty "
                                 "counters, have they been moved out ?");
    }

    if (is_thread) {
        computeSizeThreadFromThread();
    } else {
        computeSizeWaveFromThread();
    }
}

QueueInfo::QueueInfo(WaveCounterInstrumenter& instr, size_t elem_size,
                     const std::string& type_desc, const std::string& type_name,
                     size_t extra_size, float size_factor)
    : instr(&instr), is_thread(false), elem_size(elem_size),
      extra_size(extra_size), size_factor(getSizeFactor(size_factor)),
      type_desc(type_desc), type_name(type_name) {
    if (instr.getVec().empty()) {
        throw std::runtime_error("hip::QueueInfo::QueueInfo() : Empty "
                                 "counters, have they been moved out ?");
    }

    computeSizeWaveFromWave();
}

constexpr auto warpSize = 64u;

void QueueInfo::computeSizeThreadFromThread() {
    auto& t_instr = *std::get<ThreadCounterInstrumenter*>(instr);

    lttng_ust_tracepoint(hip_instrumentation, queue_compute_begin, this,
                         &t_instr);

    auto& kernel = t_instr.kernelInfo();
    auto& counters = t_instr.getVec();

    auto thread_stride = kernel.basic_blocks;
    auto blocks_stride = kernel.total_threads_per_blocks;

    offsets_vec.reserve(thread_stride * blocks_stride * kernel.basic_blocks);
    offsets_vec.push_back(0u);

    size_t nb_events = 0u;

    // Handle thread queues
    for (auto block = 0u; block < kernel.total_blocks; ++block) {
        for (auto thread = 0u; thread < kernel.total_threads_per_blocks;
             ++thread) {

            size_t events_thread = 0u;

            // For each thread, compute the number of events to record
            for (auto bb = 0u; bb < kernel.basic_blocks; ++bb) {
                events_thread += counters[block * blocks_stride +
                                          thread * thread_stride + bb];
            }

            events_thread =
                static_cast<size_t>(static_cast<float>(events_thread) *
                                    size_factor) +
                extra_size;

            nb_events += events_thread;

            // Append to the vector of offsets
            offsets_vec.push_back(nb_events);
        }
    }

    lttng_ust_tracepoint(hip_instrumentation, queue_compute_end, this);

    // The last element is the total size of the events, won't be used by the
    // last thread of the last block
}

void QueueInfo::computeSizeWaveFromThread() {
    auto& t_instr = *std::get<ThreadCounterInstrumenter*>(instr);

    lttng_ust_tracepoint(hip_instrumentation, queue_compute_begin, this,
                         &t_instr);

    auto& kernel = t_instr.kernelInfo();
    auto& counters = t_instr.getVec();

    auto thread_stride = kernel.basic_blocks;
    auto blocks_stride = kernel.total_threads_per_blocks;

    offsets_vec.reserve(thread_stride * blocks_stride * kernel.basic_blocks);
    offsets_vec.push_back(0u);

    size_t nb_events = 0u;

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
                    max_per_bblock[bb] = counter_bb * extra_size;
                }
            }
        }

        commit_wave();
    }

    lttng_ust_tracepoint(hip_instrumentation, queue_compute_end, this);
}

void QueueInfo::computeSizeWaveFromWave() {
    auto& w_instr = *std::get<WaveCounterInstrumenter*>(instr);

    lttng_ust_tracepoint(hip_instrumentation, queue_compute_begin, this,
                         &w_instr);

    auto& kernel = w_instr.kernelInfo();
    auto& counters = w_instr.getVec();

    auto wavefront_count = kernel.wavefrontCount();

    offsets_vec.reserve(wavefront_count);
    offsets_vec.push_back(0u);

    size_t nb_events = 1u;

    for (auto wave = 0u; wave < wavefront_count; ++wave) {
        // Compute all events to be generated by the wavefront (so far, a single
        // one per wave)
        auto count = counters[wave];

        count = static_cast<size_t>(static_cast<float>(count) * size_factor) +
                extra_size;

        nb_events += count;

        // Add to the offset vector
        offsets_vec.push_back(nb_events);
    }

    // std::cerr << '\n';
    // std::cerr << "Total : " << offsets_vec.back() << '\n';

    lttng_ust_tracepoint(hip_instrumentation, queue_compute_end, this);
}

void QueueInfo::fromDevice(void* ptr) {
    cpu_queue.resize(totalSize());
    hip::check(
        hipMemcpy(cpu_queue.data(), ptr, totalSize(), hipMemcpyDeviceToHost));
}

size_t QueueInfo::queueLength() const { return offsets_vec.back(); }

void QueueInfo::record(void* gpu_events) {
    auto& trace_manager = HipTraceManager::getInstance();

    trace_manager.registerQueue(*this, gpu_events);
}

// Standard events

std::string Event::description = HipEventFields<decltype(Event::bb)>();

std::string Event::name = "hip::Event";

std::string TaggedEvent::description =
    HipEventFields<decltype(TaggedEvent::bb), decltype(TaggedEvent::stamp)>();

std::string TaggedEvent::name = "hip::TaggedEvent";

std::string WaveState::description =
    HipEventFields<decltype(WaveState::stamp), decltype(WaveState::exec),
                   decltype(WaveState::hw_id), decltype(WaveState::bb)>();

std::string WaveState::name = "hip::WaveState";

std::string GlobalWaveState::description = HipEventFields<
    decltype(GlobalWaveState::stamp), decltype(GlobalWaveState::exec),
    decltype(GlobalWaveState::hw_id), decltype(GlobalWaveState::bb),
    decltype(GlobalWaveState::producer)>();

std::string GlobalWaveState::name = "hip::GlobalWaveState";

__global__ void
prefetchRegistries(CUChunkAllocator::CacheAlignedRegistry* regs) {
    auto cu_id = hip::gcnasm::get_cu_id();

    // if (threadIdx.x == 0) {
    //     printf("CU id : %d\n", cu_id);
    // }

    auto* ptr = &(regs[cu_id].reg);

    auto int_ptr = reinterpret_cast<uintptr_t>(ptr);

    uint32_t ptr_lo = static_cast<uint32_t>(int_ptr);
    uint32_t ptr_hi = static_cast<uint32_t>(int_ptr >> 32);

    asm volatile("s_mov_b32 s0, %0\n"
                 "s_mov_b32 s1, %1\n"
                 "s_load_dwordx2 s[2:3], s[0:1]\n"
                 "s_waitcnt lgkmcnt(0)\n" ::"r"(ptr_lo),
                 "r"(ptr_hi));
}

void dryRunRegistries(CUChunkAllocator& cu) {
    prefetchRegistries<<<1048576, 64>>>(cu.toDevice());
    hip::check(hipDeviceSynchronize());
}

} // namespace hip
