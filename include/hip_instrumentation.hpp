/** \file hip_instrumentation.hpp
 * \brief Kernel instrumentation embedded code
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include "hip/hip_runtime.h"

#include <vector>

namespace hip {

struct KernelInfo {
    KernelInfo(unsigned int bblocks, dim3 blcks, dim3 t_p_blcks)
        : basic_blocks(bblocks), blocks(blcks), threads_per_blocks(t_p_blcks),
          total_blocks(blcks.x * blcks.y * blcks.z),
          total_threads_per_blocks(t_p_blcks.x * t_p_blcks.y * t_p_blcks.z),
          instr_size(basic_blocks * total_blocks * total_threads_per_blocks) {}

    const dim3 blocks, threads_per_blocks;
    const unsigned int basic_blocks;

    const uint32_t total_blocks;
    const uint32_t total_threads_per_blocks;
    const uint32_t instr_size;
};

class Instrumenter {
  public:
    Instrumenter(KernelInfo& kernel_info);

    void* allocDevice();

    void fromDevice(void* device_ptr);

  private:
    std::vector<uint32_t> host_counters;
    KernelInfo kernel_info;
};

} // namespace hip
