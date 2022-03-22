#pragma once

#include <iostream>
#include <vector>
#include <chrono>

#include "matrix.hpp"
#include "stopwatch.hpp"

namespace cross {

template<typename T>
void to_csv(std::ostream& out, T value) {
    out << value;
}

template<typename T>
void to_csv(std::ostream& out, const std::vector<T>& values) {
    auto sep = "";
    for (auto&& val: values) {
        out << sep << val;
        sep = ",";
    }
}

template<typename OUT_DURATION, typename REP, typename PERIOD>
void to_csv(std::ostream& out, const std::vector<std::chrono::duration<REP, PERIOD>>& durations) {
    auto sep = "";
    for (auto&& dur: durations) {
        out << sep << std::chrono::duration_cast<OUT_DURATION>(dur).count();
        sep = ",";
    }
}

template<typename OUT_DURATION, typename IN_DURATION>
void to_csv(std::ostream& out, const std::vector<measurement_result<IN_DURATION>>& results, bool with_iterations = true) {
    auto sep = "";
    for (auto&& result: results) {
        out << sep;
        out << std::chrono::duration_cast<OUT_DURATION>(result.get_iteration_time()).count();
        sep = ",";
        if (with_iterations) {
            out << sep;
            out << result.get_iterations();
        }
    }
}

template<typename T1, typename T2>
std::vector<T1> get_labels(const std::vector<std::pair<T1, T2>>& pairs) {
    std::vector<T1> vals{pairs.size()};
    std::transform(pairs.begin(), pairs.end(), vals.begin(),
        [](auto p){ return p.first; });
    return vals;
}

template<typename T1, typename T2>
std::vector<T2> get_values(const std::vector<std::pair<T1, T2>>& pairs) {
    std::vector<T2> vals{pairs.size()};
    std::transform(pairs.begin(), pairs.end(), vals.begin(),
        [](auto p){ return p.second; });
    return vals;
}

}
