/** \file hip_instrumentation_cbindings.hpp
 * \brief Kernel instrumentation embedded code C bindings for IR manipulation
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/hip_instrumentation_cbindings.hpp"
#include "hip_instrumentation/hip_instrumentation.hpp"
#include "hip_instrumentation/queue_info.hpp"
#include "hip_instrumentation/state_recoverer.hpp"

#include "hip_instrumentation/gpu_queue.hpp"

#include "hip/hip_runtime_api.h"

#include <atomic>
#include <chrono>
#include <fstream>
#include <stdexcept>

#include "hip_analyzer_tracepoints.h"

extern "C" {

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
