/** \file hip_instrumentation_cbindings.hpp
 * \brief Kernel instrumentation embedded code C bindings for IR manipulation
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation/hip_instrumentation_cbindings.hpp"
#include "hip_instrumentation/hip_instrumentation.hpp"

extern "C" {

struct hipKernelInfo {
    hip::KernelInfo boxed;
};

struct hipInstrumenter {
    hip::Instrumenter boxed;
    hipInstrumenter(hipKernelInfo* kinfo) : boxed(kinfo->boxed) {}
};

hipInstrumenter* hipNewInstrumenter(hipKernelInfo* kinfo) {
    return new hipInstrumenter(kinfo);
}

void hipInstrumenterLoadDb(hipInstrumenter* instr, const char* file_name) {
    instr->boxed.loadDatabase(file_name);
}

counter_t* hipInstrumenterToDevice(hipInstrumenter* instr) {
    return instr->boxed.toDevice();
}

void hipInstrumenterFromDevice(hipInstrumenter* instr, void* device_ptr) {
    instr->boxed.fromDevice(device_ptr);
}

void freeHipinstrumenter(hipInstrumenter* instr) { delete instr; }
}
