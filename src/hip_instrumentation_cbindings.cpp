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

struct hipInstrumenter {
    hip::Instrumenter boxed;
    size_t shared_mem;
    hipStream_t stream;

    hipInstrumenter(hip::KernelInfo& ki) : boxed(ki) {}
    hipInstrumenter() = default;
};

struct hipStateRecoverer {
    hip::StateRecoverer boxed;
};

struct hipQueueInfo {
    hip::QueueInfo boxed;
    void* offsets;
    hipQueueInfo(hip::QueueInfo&& other) : boxed{other} {}
};

// ----- Instrumenter ----- //

hipInstrumenter* hipNewInstrumenter(const char* kernel_name) {
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

    auto* instr = new hipInstrumenter{};

    unsigned int bblocks = instr->boxed.loadDatabase(kernel_name).size();

    hip::KernelInfo ki{kernel_name, bblocks, blocks, threads};

    instr->boxed.setKernelInfo(ki);

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

counter_t* hipInstrumenterToDevice(hipInstrumenter* instr) {
    auto* c = instr->boxed.toDevice();

    auto t = std::chrono::steady_clock::now();
    timer << std::chrono::duration_cast<std::chrono::nanoseconds>(t - last_t)
                 .count()
          << ',';
    last_t = std::chrono::steady_clock::now();

    return c;
}

void hipInstrumenterFromDevice(hipInstrumenter* instr, void* device_ptr) {
    if (hipDeviceSynchronize() != hipSuccess) {
        return;
    }

    // Counters kernel timing
    auto t = std::chrono::steady_clock::now();
    timer << std::chrono::duration_cast<std::chrono::nanoseconds>(t - last_t)
                 .count()
          << ',';

    last_t = std::chrono::steady_clock::now();

    instr->boxed.fromDevice(device_ptr);

    t = std::chrono::steady_clock::now();
    timer << std::chrono::duration_cast<std::chrono::nanoseconds>(t - last_t)
                 .count()
          << ',';
}

void hipInstrumenterRecord(hipInstrumenter* instr) { instr->boxed.record(); }

void freeHipInstrumenter(hipInstrumenter* instr) {
    delete instr;
    timer << '\n';
    timer.flush();
}

// ----- State recoverer ----- //

hipStateRecoverer* hipNewStateRecoverer() {
    auto* s = new hipStateRecoverer;

    auto t = std::chrono::steady_clock::now();
    timer << std::chrono::duration_cast<std::chrono::nanoseconds>(t - last_t)
                 .count()
          << ',';

    last_t = std::chrono::steady_clock::now();

    return s;
}

void* hipStateRecovererRegisterPointer(hipStateRecoverer* recoverer,
                                       void* potential_ptr) {
    return recoverer->boxed.registerCallArgs(potential_ptr);
}

void hipStateRecovererRollback(hipStateRecoverer* recoverer,
                               hipInstrumenter* instr) {
    // recoverer->boxed.rollback();

    // Revert call configuration
    auto& ki = instr->boxed.kernelInfo();
    if (__hipPushCallConfiguration(ki.blocks, ki.threads_per_blocks,
                                   instr->shared_mem,
                                   instr->stream) != hipSuccess) {
        throw std::runtime_error(
            "hipNewInstrumenter() : Could not push call configuration");
    }
}

void freeHipStateRecoverer(hipStateRecoverer* recoverer) { delete recoverer; }

// ----- Event queue ----- //

hipQueueInfo* newHipQueueInfo(hipInstrumenter* instr, EventType event_type,
                              QueueType queue_type) {
    last_t = std::chrono::steady_clock::now();

    constexpr auto extra_size = 1u;
    switch (queue_type) {
    case QueueType::Thread:
        switch (event_type) {
        case EventType::Event:
            return new hipQueueInfo{
                hip::QueueInfo::thread<hip::Event>(instr->boxed, extra_size)};
        case EventType::TaggedEvent:
            return new hipQueueInfo{hip::QueueInfo::thread<hip::TaggedEvent>(
                instr->boxed, extra_size)};

        default:
            throw std::runtime_error(
                "newHipQueueInfo() : Unsupported queue type");
        }
    case QueueType::Wave:
        switch (event_type) {
        case EventType::Event:
            return new hipQueueInfo{
                hip::QueueInfo::wave<hip::Event>(instr->boxed, extra_size)};

        case EventType::TaggedEvent:
            return new hipQueueInfo{hip::QueueInfo::wave<hip::TaggedEvent>(
                instr->boxed, extra_size)};

        case EventType::WaveState:
            return new hipQueueInfo{
                hip::QueueInfo::wave<hip::WaveState>(instr->boxed, extra_size)};
        }
    }
}

void* hipQueueInfoAllocBuffer(hipQueueInfo* queue_info) {
    return queue_info->boxed.allocBuffer();
}

void* hipQueueInfoAllocOffsets(hipQueueInfo* queue_info) {
    void* o = queue_info->boxed.allocOffsets();
    queue_info->offsets = o;

    auto t = std::chrono::steady_clock::now();
    timer << std::chrono::duration_cast<std::chrono::nanoseconds>(t - last_t)
                 .count()
          << ',';

    last_t = std::chrono::steady_clock::now();

    return o;
}

void hipQueueInfoRecord(hipQueueInfo* queue_info, void* ptr) {
    if (hipDeviceSynchronize() != hipSuccess) {
        return;
    }

    // Tracing kernel timing
    auto t = std::chrono::steady_clock::now();
    timer << std::chrono::duration_cast<std::chrono::nanoseconds>(t - last_t)
                 .count()
          << ',';

    last_t = std::chrono::steady_clock::now();

    queue_info->boxed.record(ptr);
    hip::check(hipFree(queue_info->offsets));

    t = std::chrono::steady_clock::now();
    timer << std::chrono::duration_cast<std::chrono::nanoseconds>(t - last_t)
                 .count();
}

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
