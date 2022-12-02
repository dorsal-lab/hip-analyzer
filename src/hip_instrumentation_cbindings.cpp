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
    hipQueueInfo(hip::QueueInfo&& other) : boxed{other} {}
};

// ----- Instrumenter ----- //

hipInstrumenter* hipNewInstrumenter(const char* kernel_name) {
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
    return instr->boxed.toDevice();
}

void hipInstrumenterFromDevice(hipInstrumenter* instr, void* device_ptr) {
    if (hipDeviceSynchronize() != hipSuccess) {
        return;
    }
    instr->boxed.fromDevice(device_ptr);
}

void hipInstrumenterRecord(hipInstrumenter* instr) { instr->boxed.record(); }

void freeHipInstrumenter(hipInstrumenter* instr) { delete instr; }

// ----- State recoverer ----- //

hipStateRecoverer* hipNewStateRecoverer() { return new hipStateRecoverer; }

void hipStateRecovererRegisterPointer(hipStateRecoverer* recoverer,
                                      void* potential_ptr) {
    recoverer->boxed.registerCallArgs(potential_ptr);
}

void hipStateRecovererRollback(hipStateRecoverer* recoverer,
                               hipInstrumenter* instr) {
    recoverer->boxed.rollback();

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
    switch (queue_type) {
    case QueueType::Thread:
        switch (event_type) {
        case EventType::Event:
            return new hipQueueInfo{
                hip::QueueInfo::thread<hip::Event>(instr->boxed)};
        case EventType::TaggedEvent:
            return new hipQueueInfo{
                hip::QueueInfo::thread<hip::TaggedEvent>(instr->boxed)};

        default:
            throw std::runtime_error(
                "newHipQueueInfo() : Unsupported queue type");
        }
    case QueueType::Wave:
        switch (event_type) {
        case EventType::Event:
            return new hipQueueInfo{
                hip::QueueInfo::wave<hip::Event>(instr->boxed)};

        case EventType::TaggedEvent:
            return new hipQueueInfo{
                hip::QueueInfo::wave<hip::TaggedEvent>(instr->boxed)};

        case EventType::WaveState:
            return new hipQueueInfo{
                hip::QueueInfo::wave<hip::WaveState>(instr->boxed)};
        }
    }
}

void* hipQueueInfoAllocBuffer(hipQueueInfo* queue_info) {
    return queue_info->boxed.allocBuffer();
}

void* hipQueueInfoAllocOffsets(hipQueueInfo* queue_info) {
    return queue_info->boxed.allocOffsets();
}

void hipQueueInfoRecord(hipQueueInfo* queue_info) {
    queue_info->boxed.record();
}

void hipQueueInfoFromDevice(hipQueueInfo* queue_info, void* ptr) {
    if (hipDeviceSynchronize() != hipSuccess) {
        return;
    }
    queue_info->boxed.fromDevice(ptr);
}
}
