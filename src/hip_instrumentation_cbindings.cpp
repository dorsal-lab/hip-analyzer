/** \file hip_instrumentation_cbindings.hpp
 * \brief Kernel instrumentation embedded code C bindings for IR manipulation
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/hip_instrumentation_cbindings.hpp"
#include "hip_instrumentation/hip_instrumentation.hpp"
#include "hip_instrumentation/hip_trace_manager.hpp"
#include "hip_instrumentation/queue_info.hpp"
#include "hip_instrumentation/state_recoverer.hpp"

#include "hip_instrumentation/gpu_queue.hpp"

#include "hip/hip_runtime_api.h"

#include <atomic>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <type_traits>

#include "hip_analyzer_tracepoints.h"

extern "C" {

uint64_t rocmStamp() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(
               now.time_since_epoch())
        .count();
}

// ----- Instrumenter ----- //

hip::CounterInstrumenter* hipNewInstrumenter(const char* kernel_name,
                                             CounterType type) {
    dim3 blocks, threads;
    size_t shared_mem;
    hipStream_t stream;

    // Get the pushed call configuration
    if (__hipPopCallConfiguration(&blocks, &threads, &shared_mem, &stream) !=
        hipSuccess) {
        throw std::runtime_error(
            "hipNewInstrumenter() : Could not pop call configuration");
    }

    hip::CounterInstrumenter* instr;
    switch (type) {
    case CounterType::Thread:
        instr = new hip::ThreadCounterInstrumenter;
        break;
    case CounterType::Wave:
        instr = new hip::WaveCounterInstrumenter;
        break;
    }

    lttng_ust_tracepoint(hip_instrumentation, new_instrumenter, instr,
                         kernel_name);

    unsigned int bblocks = instr->loadDatabase(kernel_name).size();

    hip::KernelInfo ki{kernel_name, bblocks, blocks, threads};

    instr->setKernelInfo(ki);

    // Save stream for eventual re-push
    instr->shared_mem = shared_mem;
    instr->stream = stream;

    // Revert call configuration

    if (__hipPushCallConfiguration(blocks, threads, shared_mem, stream) !=
        hipSuccess) {
        throw std::runtime_error(
            "hipNewInstrumenter() : Could not push call configuration");
    }

    return instr;
}

hip::CounterInstrumenter* hipGetNextInstrumenter() {
    static std::optional<hip::HipTraceFile> trace_file;

    if (!trace_file.has_value()) {
        auto filename = std::getenv("HIPTRACE_INPUT");
        if (filename == nullptr) {
            throw std::runtime_error(
                "hipGetNextInstrumenter() : \"HIPTRACE_INPUT\" not defined for "
                "replay mode");
        }

        trace_file.emplace(filename);
    }

    if (trace_file->done()) {
        throw std::runtime_error(
            "hipGetNextInstrumenter() : Reached end of trace");
    }

    // Pop call configuration, verify constency
    dim3 blocks, threads;
    size_t shared_mem;
    hipStream_t stream;

    // Get the pushed call configuration
    if (__hipPopCallConfiguration(&blocks, &threads, &shared_mem, &stream) !=
        hipSuccess) {
        throw std::runtime_error(
            "hipGetNextInstrumenter() : Could not pop call configuration");
    }

    auto next = trace_file->getNext();

    hip::CounterInstrumenter* instr;

    std::visit(
        [&instr](auto&& event) {
            using T = std::decay_t<decltype(event)>;
            if constexpr (std::is_same_v<
                              T, hip::HipTraceManager::CountersQueuePayload<
                                     hip::HipTraceManager::ThreadCounters>>) {
                auto& [vec, ki, stamp, interval] = event;
                instr = new hip::ThreadCounterInstrumenter(std::move(vec), ki);
            } else if constexpr (std::is_same_v<
                                     T,
                                     hip::HipTraceManager::CountersQueuePayload<
                                         hip::HipTraceManager::WaveCounters>>) {
                auto& [vec, ki, stamp, interval] = event;
                instr = new hip::WaveCounterInstrumenter(std::move(vec), ki);
            } else {
                throw std::runtime_error(
                    "hipGetNextInstrumenter() : Unexpected event");
            }
        },
        next);

    lttng_ust_tracepoint(hip_instrumentation, new_replayer, instr,
                         instr->kernelInfo().name.c_str());

    // Save stream for eventual re-push
    instr->shared_mem = shared_mem;
    instr->stream = stream;

    // Revert call configuration

    if (__hipPushCallConfiguration(blocks, threads, shared_mem, stream) !=
        hipSuccess) {
        throw std::runtime_error(
            "hipGetNextInstrumenter() : Could not push call configuration");
    }

    return instr;
}

void* hipInstrumenterToDevice(hip::CounterInstrumenter* instr) {
    lttng_ust_tracepoint(hip_instrumentation, instr_to_device_begin, instr);
    auto* c = instr->toDevice();
    lttng_ust_tracepoint(hip_instrumentation, instr_to_device_end, instr, c);

    return c;
}

void hipInstrumenterFromDevice(hip::CounterInstrumenter* instr,
                               void* device_ptr) {
    lttng_ust_tracepoint(hip_instrumentation, instr_sync_begin, instr);
    if (hipDeviceSynchronize() != hipSuccess) {
        return;
    }
    lttng_ust_tracepoint(hip_instrumentation, instr_sync_end, instr);

    lttng_ust_tracepoint(hip_instrumentation, instr_from_device_begin, instr,
                         device_ptr);

    instr->fromDevice(device_ptr);
    hip::check(hipFree(device_ptr));

    lttng_ust_tracepoint(hip_instrumentation, instr_from_device_end, instr,
                         device_ptr);
}

void hipInstrumenterRecord(hip::CounterInstrumenter* instr) {
    lttng_ust_tracepoint(hip_instrumentation, instr_record_begin, instr);
    instr->record();
    lttng_ust_tracepoint(hip_instrumentation, instr_record_end, instr);
}

void freeHipInstrumenter(hip::CounterInstrumenter* instr) {
    delete instr;

    lttng_ust_tracepoint(hip_instrumentation, delete_instrumenter, instr);
}

// ----- State recoverer ----- //

hip::StateRecoverer* hipNewStateRecoverer() {
    auto* s = new hip::StateRecoverer;
    lttng_ust_tracepoint(hip_instrumentation, new_state_recoverer, s);
    return s;
}

void* hipStateRecovererRegisterPointer(hip::StateRecoverer* recoverer,
                                       void* potential_ptr) {
    auto copy_ptr = recoverer->registerCallArgs(potential_ptr);
    lttng_ust_tracepoint(hip_instrumentation, state_recoverer_register,
                         recoverer, potential_ptr, copy_ptr);
    return copy_ptr;
}

void hipStateRecovererRollback(hip::StateRecoverer* recoverer,
                               hip::CounterInstrumenter* instr) {
    // recoverer->boxed.rollback();

    // Revert call configuration
    auto& ki = instr->kernelInfo();
    if (__hipPushCallConfiguration(ki.blocks, ki.threads_per_blocks,
                                   instr->shared_mem,
                                   instr->stream) != hipSuccess) {
        throw std::runtime_error(
            "hipNewInstrumenter() : Could not push call configuration");
    }
}

void freeHipStateRecoverer(hip::StateRecoverer* recoverer) {
    lttng_ust_tracepoint(hip_instrumentation, state_recoverer_cleanup,
                         recoverer);
    delete recoverer;
}

// ----- Event queue ----- //

hip::QueueInfo*
newHipThreadQueueInfo(hip::ThreadCounterInstrumenter* thread_inst,
                      EventType event_type, QueueType queue_type) {

    constexpr auto extra_size = 2u; // Extra padding just in case
    switch (queue_type) {
    case QueueType::Thread:
        switch (event_type) {
        case EventType::Event:
            return new hip::QueueInfo{
                hip::QueueInfo::thread<hip::Event>(*thread_inst, extra_size)};
        case EventType::TaggedEvent:
            return new hip::QueueInfo{hip::QueueInfo::thread<hip::TaggedEvent>(
                *thread_inst, extra_size)};
        default:
            break;
        }
    case QueueType::Wave:
        switch (event_type) {
        case EventType::Event:
            return new hip::QueueInfo{
                hip::QueueInfo::wave<hip::Event>(*thread_inst, extra_size)};

        case EventType::TaggedEvent:
            return new hip::QueueInfo{hip::QueueInfo::wave<hip::TaggedEvent>(
                *thread_inst, extra_size)};

        case EventType::WaveState:
            return new hip::QueueInfo{
                hip::QueueInfo::wave<hip::WaveState>(*thread_inst, extra_size)};
        }
    }

    throw std::logic_error("newHipQueueInfo() : Unsupported queue type");
}

hip::QueueInfo* newHipWaveQueueInfo(hip::WaveCounterInstrumenter* wave_inst,
                                    EventType event_type,
                                    QueueType queue_type) {
    if (queue_type != QueueType::Wave) {
        throw std::logic_error(
            "newHipWaveQueueInfo() : Unsupported queue type");
    }

    switch (event_type) {
    case EventType::Event:
        return new hip::QueueInfo{hip::QueueInfo::wave<hip::Event>(*wave_inst)};

    case EventType::TaggedEvent:
        return new hip::QueueInfo{
            hip::QueueInfo::wave<hip::TaggedEvent>(*wave_inst)};

    case EventType::WaveState:
        return new hip::QueueInfo{
            hip::QueueInfo::wave<hip::WaveState>(*wave_inst)};
    }

    throw std::logic_error("newHipWaveQueueInfo() : Unsupported queue type");
}

hip::QueueInfo* newHipQueueInfo(hip::CounterInstrumenter* instr,
                                EventType event_type, QueueType queue_type) {
    switch (instr->getType()) {
    case hip::CounterInstrumenter::Type::Thread:
        return newHipThreadQueueInfo(
            reinterpret_cast<hip::ThreadCounterInstrumenter*>(instr),
            event_type, queue_type);
    case hip::CounterInstrumenter::Type::Wave:
        return newHipWaveQueueInfo(
            reinterpret_cast<hip::WaveCounterInstrumenter*>(instr), event_type,
            queue_type);
    case hip::CounterInstrumenter::Type::Default:
        throw std::logic_error("newHipQueueInfo() : unimplemented Queue info "
                               "from CounterInstrumenter!");
    }

    throw std::logic_error("newHipWaveQueueInfo() : Unsupported queue type");
}

void* hipQueueInfoAllocBuffer(hip::QueueInfo* queue_info) {
    auto ptr = queue_info->allocBuffer();
    // std::cerr << "Queue buffer " << ptr << ' ' << queue_info->totalSize()
    //           << '\n';
    return ptr;
}

void* hipQueueInfoAllocOffsets(hip::QueueInfo* queue_info) {
    void* o = queue_info->allocOffsets();

    return o;
}

void hipQueueInfoRecord(hip::QueueInfo* queue_info, void* events,
                        void* offsets) {
    lttng_ust_tracepoint(hip_instrumentation, instr_sync_begin, queue_info);
    if (hipDeviceSynchronize() != hipSuccess) {
        return;
    }
    lttng_ust_tracepoint(hip_instrumentation, instr_sync_end, queue_info);

    lttng_ust_tracepoint(hip_instrumentation, queue_record_begin, queue_info);
    queue_info->record(events);
    hip::check(hipFree(offsets));
    lttng_ust_tracepoint(hip_instrumentation, queue_record_end, queue_info);
}

void freeHipQueueInfo(hip::QueueInfo* q) { delete q; }

static std::optional<uint64_t> hiptrace_buffer_size =
    []() -> std::optional<uint64_t> {
    auto buffer_size = std::getenv("HIPTRACE_BUFFER_SIZE");
    if (buffer_size == nullptr) {
        return std::nullopt;
    }

    uint64_t val;
    std::string str{buffer_size};
    if (std::from_chars(str.data(), str.data() + str.length(), val).ec !=
        std::errc()) {
        std::cerr << "Could not parse u64 token : " << str
                  << ", using default size "
                  << hip::GlobalMemoryQueueInfo::DEFAULT_SIZE << '\n';
        return std::nullopt;
    }

    return val;
}();

hip::GlobalMemoryQueueInfo* newGlobalMemoryQueueInfo(size_t event_size) {
    hip::GlobalMemoryQueueInfo* ptr;
    if (hiptrace_buffer_size.has_value()) {
        ptr = new hip::GlobalMemoryQueueInfo(event_size, *hiptrace_buffer_size);
    } else {
        ptr = new hip::GlobalMemoryQueueInfo(event_size);
    }

    lttng_ust_tracepoint(hip_instrumentation, new_global_memory_queue, ptr);

    return ptr;
}

hip::GlobalMemoryQueueInfo::GlobalMemoryTrace*
hipGlobalMemQueueInfoToDevice(hip::GlobalMemoryQueueInfo* queue) {
    hip::GlobalMemoryQueueInfo::GlobalMemoryTrace* device_ptr;

    lttng_ust_tracepoint(hip_instrumentation, queue_compute_begin, queue,
                         queue);

    device_ptr = queue->toDevice();

    lttng_ust_tracepoint(hip_instrumentation, queue_compute_end, queue);

    return device_ptr;
}

void hipGlobalMemQueueInfoRecord(
    hip::GlobalMemoryQueueInfo* queue,
    hip::GlobalMemoryQueueInfo::GlobalMemoryTrace* device_ptr) {

    lttng_ust_tracepoint(hip_instrumentation, queue_record_begin, queue);
    queue->record(device_ptr);
    lttng_ust_tracepoint(hip_instrumentation, queue_record_end, queue);
}

void freeHipGlobalMemoryQueueInfo(hip::GlobalMemoryQueueInfo* queue) {
    delete queue;
}

hip::ChunkAllocator* newHipChunkAllocator(const char* kernel_name,
                                          size_t buffer_count,
                                          size_t buffer_size) {

    dim3 blocks, threads;
    size_t shared_mem;
    hipStream_t stream;

    // Get the pushed call configuration
    if (__hipPopCallConfiguration(&blocks, &threads, &shared_mem, &stream) !=
        hipSuccess) {
        throw std::runtime_error(
            "hipNewInstrumenter() : Could not pop call configuration");
    }

    hip::ChunkAllocator* alloc = hip::ChunkAllocator::getStreamAllocator(
        stream, buffer_count, buffer_size);

    if (__hipPushCallConfiguration(blocks, threads, shared_mem, stream) !=
        hipSuccess) {
        throw std::runtime_error(
            "hipNewInstrumenter() : Could not push call configuration");
    }

    lttng_ust_tracepoint(hip_instrumentation, new_chunk_allocator, alloc,
                         kernel_name, buffer_count, buffer_size,
                         alloc->toDevice());

    return alloc;
}

hip::ChunkAllocator::Registry*
hipChunkAllocatorToDevice(hip::ChunkAllocator* ca) {
    lttng_ust_tracepoint(hip_instrumentation, instr_to_device_begin, ca);

    auto* ptr = ca->toDevice();

    lttng_ust_tracepoint(hip_instrumentation, instr_to_device_end, ca, ptr);

    return ptr;
}

void hipChunkAllocatorRecord(hip::ChunkAllocator* ca, uint64_t stamp) {
    hip::check(hipDeviceSynchronize());
    lttng_ust_tracepoint(hip_instrumentation, queue_record_begin, ca);

    ca->record(stamp);

    lttng_ust_tracepoint(hip_instrumentation, queue_record_end, ca);
    lttng_ust_tracepoint(hip_instrumentation, delete_instrumenter, ca);
}

void freeChunkAllocator(hip::ChunkAllocator* ca) {
    lttng_ust_tracepoint(hip_instrumentation, delete_instrumenter, ca);
    delete ca;
}

hip::CUChunkAllocator* newHipCUChunkAllocator(const char* kernel_name,
                                              size_t buffer_count,
                                              size_t buffer_size) {

    dim3 blocks, threads;
    size_t shared_mem;
    hipStream_t stream;

    // Get the pushed call configuration
    if (__hipPopCallConfiguration(&blocks, &threads, &shared_mem, &stream) !=
        hipSuccess) {
        throw std::runtime_error(
            "hipNewInstrumenter() : Could not pop call configuration");
    }

    hip::CUChunkAllocator* alloc = hip::CUChunkAllocator::getStreamAllocator(
        stream, buffer_count / 64, buffer_size);

    if (__hipPushCallConfiguration(blocks, threads, shared_mem, stream) !=
        hipSuccess) {
        throw std::runtime_error(
            "hipNewInstrumenter() : Could not push call configuration");
    }

    lttng_ust_tracepoint(hip_instrumentation, new_chunk_allocator, alloc,
                         kernel_name, buffer_count, buffer_size,
                         alloc->toDevice());

    return alloc;
}

hip::CUChunkAllocator::CacheAlignedRegistry*
hipCUChunkAllocatorToDevice(hip::CUChunkAllocator* ca) {
    lttng_ust_tracepoint(hip_instrumentation, instr_to_device_begin, ca);

    auto* ptr = ca->toDevice();

    lttng_ust_tracepoint(hip_instrumentation, instr_to_device_end, ca, ptr);

    return ptr;
}

void hipCUChunkAllocatorRecord(hip::CUChunkAllocator* ca, uint64_t stamp) {
    hip::check(hipDeviceSynchronize());
    lttng_ust_tracepoint(hip_instrumentation, queue_record_begin, ca);

    ca->record(stamp);

    lttng_ust_tracepoint(hip_instrumentation, queue_record_end, ca);

    ca->sync();

    lttng_ust_tracepoint(hip_instrumentation, delete_instrumenter, ca);
}

void freeCUChunkAllocator(hip::CUChunkAllocator* ca) {
    lttng_ust_tracepoint(hip_instrumentation, delete_instrumenter, ca);
    delete ca;
}

// ----- Experimental - Kernel timer ----- //

namespace {

std::atomic<unsigned int> timer_kernel_id{0u};

}

unsigned int begin_kernel_timer(const char* kernel) {
    auto id = timer_kernel_id++;
    lttng_ust_tracepoint(hip_instrumentation, kernel_timer_begin, kernel, id);

    return id;
}

void end_kernel_timer(unsigned int id) {
    if (hipDeviceSynchronize() != hipSuccess) {
        return;
    }

    lttng_ust_tracepoint(hip_instrumentation, kernel_timer_end, id);
}
}
