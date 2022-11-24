/** \file hip_instrumentation_cbindings.hpp
 * \brief Kernel instrumentation embedded code C bindings for IR manipulation
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/hip_instrumentation_cbindings.hpp"
#include "hip_instrumentation/hip_instrumentation.hpp"
#include "hip_instrumentation/state_recoverer.hpp"

#include "hip/hip_runtime_api.h"

extern "C" {

struct hipInstrumenter {
    hip::Instrumenter boxed;

    hipInstrumenter(hip::KernelInfo& ki) : boxed(ki) {}
    hipInstrumenter() = default;
};

struct hipStateRecoverer {
    hip::StateRecoverer boxed;
};

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

    unsigned int bblocks = instr->boxed.loadDatabase().size();

    hip::KernelInfo ki{kernel_name, bblocks, blocks, threads};

    instr->boxed.setKernelInfo(ki);

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

void hipStateRecovererRollback(hipStateRecoverer* recoverer) {
    recoverer->boxed.rollback();
}

void freeHipStateRecoverer(hipStateRecoverer* recoverer) { delete recoverer; }
}
