/** \file basic_block.cpp
 * \brief Kernel static informations
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "basic_block.hpp"

// Std includes

#include <cstdlib>
#include <fstream>
#include <sstream>

// Jsoncpp (shipped with ubuntu & debian)

#include <json/json.h>

namespace hip {

BasicBlock::BasicBlock(unsigned int i, unsigned int f, const std::string& begin,
                       const std::string& end)
    : id(i), flops(f), begin_loc(std::make_unique<std::string>(begin)),
      end_loc(std::make_unique<std::string>(end)) {}

std::string hip::BasicBlock::json() const {
    std::stringstream ss;

    ss << "{\"id\": " << id << ", \"begin\": \"" << *begin_loc
       << "\", \"end\": \"" << *end_loc << "\", \"flops\": " << flops << "}";

    return ss.str();
}

std::string BasicBlock::jsonArray(const std::vector<BasicBlock>& blocks) {
    std::stringstream ss;

    ss << '[';

    for (auto& block : blocks) {
        ss << block.json() << ',';
    }

    // Remove the last comma and replace it with the closing bracket
    ss.seekp(-1, ss.cur);
    ss << ']';

    return ss.str();
}

BasicBlock BasicBlock::fromJson(const std::string& json) {
    Json::Value root;

    std::ifstream file_in(json);

    file_in >> root;

    unsigned int id = root.get("id", 0u).asUInt();
    unsigned int flops = root.get("flops", 0u).asUInt();
    std::string begin_loc = root.get("begin", "").asString();
    std::string end_loc = root.get("end", "").asString();

    return {id, flops, begin_loc, end_loc};
}

std::vector<BasicBlock> BasicBlock::fromJsonArray(const std::string& json) {
    std::vector<BasicBlock> blocks;
    Json::Value root;

    std::ifstream file_in(json);
    file_in >> root;

    for (auto value : root) {
        unsigned int id = value.get("id", 0u).asUInt();
        unsigned int flops = value.get("flops", 0u).asUInt();
        std::string begin_loc = value.get("begin", "").asString();
        std::string end_loc = value.get("end", "").asString();

        blocks.emplace_back(id, flops, begin_loc, end_loc);
    }

    return blocks;
}

std::string BasicBlock::getEnvDatabaseFile(const std::string& kernel_name) {
    // Unused arg kernel name
    if (const char* env = std::getenv(hip::env_var_name)) {
        return {env};
    } else {
        return {hip::default_database};
    }
}

std::vector<BasicBlock>
BasicBlock::normalized(const std::vector<BasicBlock>& blocks) {
    auto max_el = std::max_element(blocks.begin(), blocks.end())->id;

    std::vector<BasicBlock> b_norm(max_el + 1, {0, 0, "", ""});

    // No checking is done to verify if an id is unique, todo ?
    for (const auto& b : blocks) {
        b_norm[b.id] = b;
    }

    return b_norm;
}

} // namespace hip
