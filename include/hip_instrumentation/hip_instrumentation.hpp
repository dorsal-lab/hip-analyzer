/** \file hip_instrumentation.hpp
 * \brief Kernel instrumentation embedded code
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

#include "hip/hip_runtime.h"

#include <optional>
#include <unordered_map>
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

    KernelInfo(const KernelInfo&) = default;

    const std::string name;
    const unsigned int basic_blocks;
    const dim3 blocks, threads_per_blocks;

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

/** \class CounterInstrumenter
 * \brief Instrumentation instance, holding host-side counters. It can either be
 * used for instrumentation or post-mortem analysis ( see \ref loadCsv and \ref
 * loadBin)
 */
class CounterInstrumenter {
  public:
    static constexpr auto HIP_ANALYZER_ENV = "HIP_ANALYZER_DATABASE";
    static constexpr auto HIP_ANALYZER_DEFAULT_FILE = "hip_analyzer.json";

    /** \brief ctor. Used when the kernel configuration is already known, and
     * the basic block count is stored as a constant
     */
    CounterInstrumenter(KernelInfo& kernel_info);

    /** \brief ctor. Default-initializes the kernel info.
     */
    CounterInstrumenter();

    virtual ~CounterInstrumenter() {}

    // ----- Device data collection ----- //

    /** \fn loadDatabase
     * \brief Load blocks from database, by looking in usual places to store
     * them
     *
     * \details Looks for the HIP_ANALYZER_DATABASE environnment variable,
     * hip_analyzer.json and <kernel_name>.json files.
     */
    const std::vector<hip::BasicBlock>& loadDatabase();

    /** \fn loadDatabase
     * \brief Load blocks from database
     */
    const std::vector<hip::BasicBlock>&
    loadDatabase(const std::string& kernel_name);

    /** \fn loadDatabase
     * \brief Load blocks from database
     */
    const std::vector<hip::BasicBlock>&
    loadDatabase(const std::string& filename_in,
                 const std::string& kernel_name);

    std::string getDatabaseName() const;

    /** \fn toDevice
     * \brief Allocates data on both the host and the device, returns the
     * device pointer.
     */
    virtual void* toDevice() = 0;

    /** \fn fromDevice
     * \brief Fetches data back from the device
     */
    virtual void fromDevice(void* device_ptr) = 0;

    // ----- Save & load data ----- //

    /** \fn data
     * \brief Const ptr to the host counters
     */
    virtual const void* data() const = 0;

    /** \fn dumpCsv
     * \brief Dump the data in a csv format. If no filename is given, it is
     * generated automatically from the kernel name and the timestamp
     */
    virtual void dumpCsv(const std::string& filename = "") = 0;

    /** \fn dumpBin
     * \brief Dump the data in a binary format (packed). A KernelInfo json dump
     * is then required to analyze the data
     */
    virtual void dumpBin(const std::string& filename = "") = 0;

    /** \fn record
     * \brief Delegates the counters to the trace manager, which will handle how
     * it will be saved in the filesystem
     */
    virtual void record() = 0;

    /** \fn kernelInfo
     * \brief Returns a ref to the kernel information
     */
    const KernelInfo& kernelInfo() const { return *kernel_info; }

    /** \fn setKernelInfo
     * \brief Sets the kernel info at runtime
     */
    virtual const KernelInfo& setKernelInfo(KernelInfo& ki) = 0;

    /** \fn getStamp
     * \brief Returns the Instrumenter timestamp (construction)
     */
    uint64_t getStamp() const { return stamp; }

    /** \fn getInterval
     * \brief Returns the kernel execution time
     */
    std::pair<uint64_t, uint64_t> getInterval() const {
        return std::make_pair(stamp_begin, stamp_end);
    }

    size_t shared_mem;
    hipStream_t stream;

    // Static factory methods

    /** \fn loadCsv
     * \brief Load data from a csv-formated file.
     */
    std::unique_ptr<CounterInstrumenter> fromCsv(const std::string& filename);

    /** \fn loadBin
     * \brief Load data from a packed binary format (see \ref dumpBin).
     */
    std::unique_ptr<CounterInstrumenter> fromBin(const std::string& filename);

  protected:
    std::string autoFilenamePrefix() const;

    std::optional<KernelInfo> kernel_info;

    std::vector<hip::BasicBlock>* blocks;

    /** \brief std::chrono stamp for quick identification
     */
    uint64_t stamp;

    /** \brief Roctracer stamps for kernel launch identification
     */
    uint64_t stamp_begin;
    uint64_t stamp_end;

    static std::unordered_map<std::string, std::vector<hip::BasicBlock>>
        known_blocks;

    static bool rocprofiler_initializer;
};

/** \class ThreadCounterInstrumenter
 * \brief Corresponds to a per-thread instrumented kernel.
 */
class ThreadCounterInstrumenter : public CounterInstrumenter {
  public:
    ThreadCounterInstrumenter(KernelInfo& ki) : CounterInstrumenter(ki) {
        host_counters.assign(ki.instr_size, 0u);
    }

    ThreadCounterInstrumenter() : CounterInstrumenter() {}

    /** \brief Counter underlying type
     */
    using counter_t = uint8_t;

    void* toDevice() override;

    void fromDevice(void* device_ptr) override;

    const void* data() const override { return host_counters.data(); }

    void dumpCsv(const std::string& filename = "") override;

    void dumpBin(const std::string& filename = "") override;

    void record() override;

    /** \fn setKernelInfo
     * \brief Sets the kernel info at runtime
     */
    const KernelInfo& setKernelInfo(KernelInfo& ki) override {
        kernel_info.emplace(ki);
        host_counters.reserve(ki.instr_size);
        host_counters.assign(ki.instr_size, 0u);

        return *kernel_info;
    }

    const std::vector<counter_t>& getVec() const { return host_counters; }

    size_t loadCsv(const std::string& filename);
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

    /** \fn parseHeader
     * \brief Validate header from binary trace
     */
    bool parseHeader(const std::string& header);

  private:
    std::vector<counter_t> host_counters;
};

/** \class WaveCounterInstrumenter
 * \brief Corresponds to a per-wavefront instrumented kernel.
 */
class WaveCounterInstrumenter : public CounterInstrumenter {
  public:
    /** \brief Counter underlying type
     */
    using counter_t = uint32_t;

  private:
    std::vector<counter_t> host_counters;
};

} // namespace hip
