/** \file hip_utils.hpp
 * \brief Common hip functions
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#ifndef HIP_HIP_UTILS_HPP_
#define HIP_HIP_UTILS_HPP_

#include "hip/hip_runtime.h"

// Std includes

#include <execinfo.h>
#include <iostream>
#include <unistd.h>

namespace hip {

inline void check(hipError_t err) {
    if (err != hipSuccess) {
        std::cerr << "error : " << hipGetErrorString(err) << " (" << err
                  << ")\nBacktrace :\n";

        void* fds[32];
        size_t size;

        size = backtrace(fds, 32);
        backtrace_symbols_fd(fds, size, STDERR_FILENO);

        throw std::runtime_error(std::string("Encountered hip error ") +
                                 hipGetErrorString(err));
    }
}

inline std::unique_ptr<hipDeviceProp_t> init(int device = 0) {
    auto properties = std::make_unique<hipDeviceProp_t>();
    check(hipSetDevice(device));
    check(hipGetDeviceProperties(properties.get(), device));
    return properties;
}

/** \fn getNumberCUs
 * \brief Returns the number of Physical Compute Units on the current
 * device
 */
inline int getNumberCUs() {
    int device_id;
    hip::check(hipGetDevice(&device_id));

    int val;
    hip::check(hipDeviceGetAttribute(
        &val, hipDeviceAttributePhysicalMultiProcessorCount, device_id));
    return val;
}

#ifdef __HIPCC__
template <class F, class... Args>
__device__ inline std::invoke_result_t<F, Args...> scalar(F&& f,
                                                          Args&&... args) {
    uint64_t exec = 0u;
    asm volatile("s_mov_b64 %0, exec" : "=s"(exec) :);
    uint64_t thread = __lastbit_u32_u64(exec);
    if (threadIdx.x % warpSize == thread) {
        std::invoke(f, args...);
    }
}
#endif

} // namespace hip

#endif
