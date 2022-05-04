/** \file hip_instrumentation.cpp
 * \brief Kernel instrumentation embedded code
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "hip_instrumentation.hpp"

#include <iostream>

namespace hip {

inline void check(hipError_t err) {
    if (err != hipSuccess) {
        std::cerr << "error : " << hipGetErrorString(err) << " (" << err
                  << ")\n";
        throw std::runtime_error(std::string("Encountered hip error ") +
                                 hipGetErrorString(err));
    }
}

// GCN Assembly

/** \brief Saves the EXEC registers in two VGPRs (variables h & l)
 */
constexpr auto save_register =
    "asm volatile (\"s_mov_b32 s6, exec_lo\\n s_mov_b32 s7, exec_hi\\n "
    "v_mov_b32 %0, s6\\n v_mov_b32 %1, s7\":\"=v\" (l), \"=v\" (h):);";

// ---- Instrumentation ----- //

Instrumenter::Instrumenter(KernelInfo& ki)
    : kernel_info(ki), host_counters(ki.instr_size, 0u) {}

void* Instrumenter::toDevice() {
    void* data_device;
    hip::check(
        hipMalloc(&data_device, kernel_info.instr_size * sizeof(uint32_t)));

    hip::check(hipMemcpy(data_device, host_counters.data(),
                         kernel_info.instr_size * sizeof(uint32_t),
                         hipMemcpyHostToDevice));

    return data_device;
}

void Instrumenter::fromDevice(void* device_ptr) {
    hip::check(hipMemcpy(host_counters.data(), device_ptr,
                         kernel_info.instr_size, hipMemcpyDeviceToHost));
}

} // namespace hip
