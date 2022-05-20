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

struct BasicBlock {
    /** ctor
     */
    BasicBlock(unsigned int id, unsigned int flops, const std::string& begin,
               const std::string& end);

    BasicBlock(const BasicBlock& other) : id(other.id), flops(other.flops) {}

    BasicBlock& operator=(const BasicBlock& other) {
        id = other.id;
        flops = other.flops;
        return *this;
    }

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

    unsigned int id;
    unsigned int flops;

    // These are allocated as pointers as to reduce the memory footprint on the
    // device
    std::unique_ptr<std::string> begin_loc, end_loc;
};

} // namespace hip
