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

#include <chrono>
#include <fstream>

std::ofstream timer;

auto last_t = std::chrono::steady_clock::now();

extern "C" {

// ----- Instrumenter ----- //

hip::CounterInstrumenter* hipNewInstrumenter(const char* kernel_name,
                                             CounterType type) {
    if (!timer.is_open()) {

        std::string prefix = "";
        if (auto* benchmark = std::getenv("RODINIA_BENCHMARK")) {
            prefix = benchmark;
            prefix += '_';
        }

        timer.open(prefix + "timing.csv", std::ofstream::trunc);
        timer << "kernel,counters_prep,save_alloc,counters,counters_record,"
                 "queue_"
                 "prep,tracing,tracing_record\n";
    }
    timer << kernel_name << ',';
    last_t = std::chrono::steady_clock::now();

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
    auto* c = instr->toDevice();

    auto t = std::chrono::steady_clock::now();
    timer << std::chrono::duration_cast<std::chrono::nanoseconds>(t - last_t)
                 .count()
          << ',';
    last_t = std::chrono::steady_clock::now();

    return c;
}

void hipInstrumenterFromDevice(hip::CounterInstrumenter* instr,
                               void* device_ptr) {
    if (hipDeviceSynchronize() != hipSuccess) {
        return;
    }

    // Counters kernel timing
    auto t = std::chrono::steady_clock::now();
    timer << std::chrono::duration_cast<std::chrono::nanoseconds>(t - last_t)
                 .count()
          << ',';

    last_t = std::chrono::steady_clock::now();

    instr->fromDevice(device_ptr);
    hip::check(hipFree(device_ptr));

    t = std::chrono::steady_clock::now();
    timer << std::chrono::duration_cast<std::chrono::nanoseconds>(t - last_t)
                 .count()
          << ',';
}

void hipInstrumenterRecord(hip::CounterInstrumenter* instr) { instr->record(); }

void freeHipInstrumenter(hip::CounterInstrumenter* instr) {
    delete instr;
    timer << '\n';
    timer.flush();
}

// ----- State recoverer ----- //

hip::StateRecoverer* hipNewStateRecoverer() {
    auto* s = new hip::StateRecoverer;

    auto t = std::chrono::steady_clock::now();
    timer << std::chrono::duration_cast<std::chrono::nanoseconds>(t - last_t)
                 .count()
          << ',';

    last_t = std::chrono::steady_clock::now();

    return s;
}

void* hipStateRecovererRegisterPointer(hip::StateRecoverer* recoverer,
                                       void* potential_ptr) {
    return recoverer->registerCallArgs(potential_ptr);
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

void freeHipStateRecoverer(hip::StateRecoverer* recoverer) { delete recoverer; }

// ----- Event queue ----- //

hip::QueueInfo* newHipQueueInfo(hip::CounterInstrumenter* instr,
                                EventType event_type, QueueType queue_type) {
    last_t = std::chrono::steady_clock::now();

    auto& thread_inst = reinterpret_cast<hip::ThreadCounterInstrumenter&>(
        *instr); // DANGEROUS, TODO CHANGE

    constexpr auto extra_size = 20u; // Extra padding just in case
    switch (queue_type) {
    case QueueType::Thread:
        switch (event_type) {
        case EventType::Event:
            return new hip::QueueInfo{
                hip::QueueInfo::thread<hip::Event>(thread_inst, extra_size)};
        case EventType::TaggedEvent:
            return new hip::QueueInfo{hip::QueueInfo::thread<hip::TaggedEvent>(
                thread_inst, extra_size)};

        default:
            throw std::runtime_error(
                "newHipQueueInfo() : Unsupported queue type");
        }
    case QueueType::Wave:
        switch (event_type) {
        case EventType::Event:
            return new hip::QueueInfo{
                hip::QueueInfo::wave<hip::Event>(thread_inst, extra_size)};

        case EventType::TaggedEvent:
            return new hip::QueueInfo{hip::QueueInfo::wave<hip::TaggedEvent>(
                thread_inst, extra_size)};

        case EventType::WaveState:
            return new hip::QueueInfo{
                hip::QueueInfo::wave<hip::WaveState>(thread_inst, extra_size)};
        }
    }
}

void* hipQueueInfoAllocBuffer(hip::QueueInfo* queue_info) {
    return queue_info->allocBuffer();
}

void* hipQueueInfoAllocOffsets(hip::QueueInfo* queue_info) {
    void* o = queue_info->allocOffsets();

    auto t = std::chrono::steady_clock::now();
    timer << std::chrono::duration_cast<std::chrono::nanoseconds>(t - last_t)
                 .count()
          << ',';

    last_t = std::chrono::steady_clock::now();

    return o;
}

void hipQueueInfoRecord(hip::QueueInfo* queue_info, void* events,
                        void* offsets) {
    if (hipDeviceSynchronize() != hipSuccess) {
        return;
    }

    // Tracing kernel timing
    auto t = std::chrono::steady_clock::now();
    timer << std::chrono::duration_cast<std::chrono::nanoseconds>(t - last_t)
                 .count()
          << ',';

    last_t = std::chrono::steady_clock::now();

    queue_info->record(events);
    hip::check(hipFree(offsets));

    t = std::chrono::steady_clock::now();
    timer << std::chrono::duration_cast<std::chrono::nanoseconds>(t - last_t)
                 .count();
}

void freeHipQueueInfo(hip::QueueInfo* q) { delete q; }

// ----- Experimental - Kernel timer ----- //

std::ofstream kernel_timer_output;

auto kernel_timer = std::chrono::steady_clock::now();

void begin_kernel_timer(const char* kernel) {
    if (!kernel_timer_output.is_open()) {
        std::string prefix = "original";
        if (auto* benchmark = std::getenv("RODINIA_BENCHMARK")) {
            prefix += benchmark;
            prefix += '_';
        }

        kernel_timer_output.open(prefix + "timing.csv", std::ofstream::trunc);
        kernel_timer_output << "kernel,original\n";
    }
    kernel_timer_output << kernel << ',';
    kernel_timer = std::chrono::steady_clock::now();
}

void end_kernel_timer() {
    if (hipDeviceSynchronize() != hipSuccess) {
        return;
    }

    auto t1 = std::chrono::steady_clock::now();
    kernel_timer_output << std::chrono::duration_cast<std::chrono::nanoseconds>(
                               t1 - kernel_timer)
                               .count()
                        << '\n';
    kernel_timer_output.flush();
}
}
