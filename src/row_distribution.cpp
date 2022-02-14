#include "row_distribution.cuh"

#include <string>
#include <unordered_map>
#include <utility>

namespace cross {
static std::unordered_map<std::string, distribution> distributions {
    {"rectangle", distribution::rectangle},
    {"triangle", distribution::triangle}
};

distribution from_string(const std::string& val) {
    auto dist = distributions.find(val);
    if (dist == distributions.end()) {
        throw std::runtime_error("Unknown distribution "s + val);
    }
    return dist->second;
}

std::string to_string(distribution dist) {
    for (auto const& [key, val]: distributions) {
        if (val == dist) {
            return key;
        }
    }
    throw std::runtime_error("Unknown distribution"s + std::to_string(static_cast<std::underlying_type<distribution>::type>(dist)));
}

}
