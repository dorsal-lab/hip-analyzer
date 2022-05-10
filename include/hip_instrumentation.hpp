/** \file hip_instrumentation.hpp
 * \brief Kernel instrumentation embedded code
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include "hip/hip_runtime.h"

#include <vector>

namespace hip {

/** \struct KernelInfo
 * \brief Holds data relative to the static and dynamic features of a kernel
 * launch (bblocks, kernel geometry)
 */
struct KernelInfo {
    KernelInfo(const std::string& _name, unsigned int bblocks, dim3 blcks,
               dim3 t_p_blcks)
        : name(_name), basic_blocks(bblocks), blocks(blcks),
          threads_per_blocks(t_p_blcks),
          total_blocks(blcks.x * blcks.y * blcks.z),
          total_threads_per_blocks(t_p_blcks.x * t_p_blcks.y * t_p_blcks.z),
          instr_size(basic_blocks * total_blocks * total_threads_per_blocks) {}

    const std::string name;
    const dim3 blocks, threads_per_blocks;
    const unsigned int basic_blocks;

    const uint32_t total_blocks;
    const uint32_t total_threads_per_blocks;
    const uint32_t instr_size;
};

class Instrumenter {
  public:
    /** \brief ctor
     */
    Instrumenter(KernelInfo& kernel_info);

    /** \fn toDevice
     * \brief Allocates data on both the host and the device, returns the device
     * pointer.
     */
    void* toDevice();

    /** \fn fromDevice
     * \brief Fetches data back from the device
     */
    void fromDevice(void* device_ptr);

    /** \fn dumpCsv
     * \brief Dump the data in a csv format. If no filename is given, it is
     * generated automatically from the kernel name and the timestamp
     */
    void dumpCsv(const std::string& filename = "");

  private:
    std::vector<uint32_t> host_counters;
    KernelInfo kernel_info;

    uint64_t stamp;
}; // namespace hip

} // namespace hip
