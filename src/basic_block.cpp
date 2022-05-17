/** \file basic_block.cpp
 * \brief Kernel static informations
 *
 * \author Sébastien Darche <sebastien.darche@polymtl.ca>
 */

#include "basic_block.hpp"

// Std includes

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
        unsigned int id = root.get("id", 0u).asUInt();
        unsigned int flops = root.get("flops", 0u).asUInt();
        std::string begin_loc = root.get("begin", "").asString();
        std::string end_loc = root.get("end", "").asString();

        blocks.emplace_back(id, flops, begin_loc, end_loc);
    }

    return blocks;
}

} // namespace hip
