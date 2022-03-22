#pragma once

#include <string>
#include <iostream>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace cross {

template<typename T>
inline T from_string(const std::string& in);


template<>
inline float from_string<float>(const std::string& in){
    return std::stof(in);
}

template<>
inline double from_string<double>(const std::string& in) {
    return std::stod(in);
}


template<typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec) {
    std::string sep = "";
    for(auto&& val: vec) {
        out << sep << val;
        sep = ", ";
    }
    return out;
}

template<typename VAL>
std::vector<VAL> get_sorted_values(const std::unordered_set<VAL>& set) {
    std::vector<std::string> values{set.size()};
    std::sort(values.begin(), values.end());
    return values;
}

template<typename KEY, typename VALUE>
std::vector<KEY> get_keys(const std::unordered_map<KEY, VALUE>& map) {
    std::vector<std::string> keys{map.size()};
    transform(map.begin(), map.end(), keys.begin(), [](auto pair){return pair.first;});
    return keys;
}

template<typename KEY, typename VALUE>
std::vector<KEY> get_sorted_keys(const std::unordered_map<KEY, VALUE>& map) {
    auto keys = get_keys(map);
    std::sort(keys.begin(), keys.end());
    return keys;
}

template<typename KEY, typename VALUE>
std::unordered_set<KEY> get_key_set(const std::unordered_map<KEY, VALUE>& map) {
    auto key_vector = get_keys(map);

    std::unordered_set<std::string> key_set{key_vector.begin(), key_vector.end()};

    return key_set;
}

}
