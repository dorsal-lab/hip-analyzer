/** \file basic_block.hpp
 * \brief Kernel static informations
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#pragma once

// Std includes

#include <memory>
#include <string>
#include <vector>

namespace hip {

constexpr auto default_database = "hip_analyzer.json";
constexpr auto env_var_name = "HIP_ANALYZER_DB";

/** \struct BasicBlock
 * \brief Holds information about a basic block, both for front-end analysis and
 * runtime instrumentation
 */
struct BasicBlock {
    /** ctor
     */
    BasicBlock(uint32_t id, uint32_t clang_id, uint32_t flops,
               const std::string& begin, const std::string& end,
               uint32_t floating_ld = 0u, uint32_t floating_st = 0u);

    BasicBlock(const BasicBlock& other);

    BasicBlock& operator=(const BasicBlock& other);

    /** \fn json
     * \brief Dump block to JSON
     */
    std::string json() const;

    /** \fn jsonArray
     * \brief Dump blocks to JSON
     */
    static std::string jsonArray(const std::vector<BasicBlock>& blocks);

    /** \fn fromJson
     * \brief Load block from JSON format
     */
    static BasicBlock fromJson(const std::string& json);

    /** \fn fromJsonArray
     * \brief Load block from JSON array format
     */
    static std::vector<BasicBlock> fromJsonArray(const std::string& json);

    /** \fn getEnvDatabaseFile
     * \brief Returns a path to the instrumentation database, stored either as
     * an environment variable or using the default value
     */
    static std::string getEnvDatabaseFile(const std::string& kernel_name);

    /** \fn normalized
     * \brief Returns a (longer) vector in a "normalized" form : the id is its
     * index in the vector
     */
    static std::vector<BasicBlock>
    normalized(const std::vector<BasicBlock>& blocks);

    bool operator<(const BasicBlock& other) const { return id < other.id; }

    /** \brief Block id, its unique identifier wrt. the instrumentation runtime
     */
    uint32_t id;

    /** \brief Block id according to the clang front end. Used to relate
     * instrumentation information back to the original file
     */
    uint32_t clang_id;

    /** \brief Number of floating point operations in the basic block
     */
    uint32_t flops;

    /** \brief Bytes loaded from memory for floating point data
     */
    uint32_t floating_ld;

    /** \brief Bytes stored to memory for floating point data
     */
    uint32_t floating_st;

    // These are allocated as pointers as to reduce the memory footprint on the
    // device
    std::unique_ptr<std::string> begin_loc, end_loc;
};

} // namespace hip
