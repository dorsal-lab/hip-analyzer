/** \file hip_instrumentation.hpp
 * \brief Kernel instrumentation embedded code
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include "hip/hip_runtime.h"

#include <vector>

#include "basic_block.hpp"
#include "hip_utils.hpp"

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

    /** \fn dump
     * \brief Prints on the screen the data held by the struct
     */
    void dump() const;

    /** \fn json
     * \brief Returns the kernel info as a JSON database
     */
    std::string json();

    static KernelInfo fromJson(const std::string& filename);
};

/** \class Instrumenter
 * \brief Instrumentation instance, holding host-side counters. It can either be
 * used for instrumentation or post-mortem analysis ( see \ref loadCsv and \ref
 * loadBin)
 */
class Instrumenter {
    using counter_t = uint8_t;

  public:
    /** \brief ctor
     */
    Instrumenter(KernelInfo& kernel_info);

    // ----- Device data collection ----- //

    /** \fn dumpBin
     * \brief Load blocks from database
     */
    const std::vector<hip::BasicBlock>&
    loadDatabase(const std::string& filename = "");

    /** \fn toDevice
     * \brief Allocates data on both the host and the device, returns the device
     * pointer.
     */
    counter_t* toDevice();

    /** \fn fromDevice
     * \brief Fetches data back from the device
     */
    void fromDevice(void* device_ptr);

    // ----- Save & load data ----- //

    /** \fn data
     * \brief Const ref to the host counters
     */
    const std::vector<counter_t>& data() const { return host_counters; }

    /** \fn dumpCsv
     * \brief Dump the data in a csv format. If no filename is given, it is
     * generated automatically from the kernel name and the timestamp
     */
    void dumpCsv(const std::string& filename = "");

    /** \fn dumpBin
     * \brief Dump the data in a binary format (packed). A KernelInfo json dump
     * is then required to analyze the data
     */
    void dumpBin(const std::string& filename = "");

    /** \fn loadCsv
     * \brief Load data from a csv-formated file.
     */
    size_t loadCsv(const std::string& filename);

    /** \fn loadBin
     * \brief Load data from a packed binary format (see \ref dumpBin).
     */
    size_t loadBin(const std::string& filename);

    // ----- Post-instrumentation reduce ----- //

    /** \fn reduceFlops
     * \brief Compute the number of floating point operations in the
     * instrumented kernel execution
     *
     * \param device_ptr Pointer to the (device) instrumentation data
     * \param stream Synchronization stream. If nullptr, synchronizes the device
     */
    unsigned int reduceFlops(const counter_t* device_ptr,
                             hipStream_t stream = nullptr) const;

  private:
    /** \fn parseHeader
     * \brief Validate header from binary trace
     */
    bool parseHeader(const std::string& header);

    std::string autoFilenamePrefix() const;

    std::vector<counter_t> host_counters;
    KernelInfo kernel_info;

    std::vector<hip::BasicBlock> blocks;

    /** \brief std::chrono stamp for quick identification
     */
    uint64_t stamp;

    /** \brief Roctracer stamps for kernel launch identification
     */
    uint64_t stamp_begin;
    uint64_t stamp_end;
};

} // namespace hip
