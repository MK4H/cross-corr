#pragma once

#include <iostream>
#include <filesystem>
#include <chrono>
#include <unordered_set>

#include <boost/program_options.hpp>

namespace cross {

namespace po = boost::program_options;

enum class BenchmarkType {
    Unknown,
    None,
    Compute,
    // Measure steps common to all algorithms
    CommonSteps,
    // Algorithm specific steps
    Algorithm
};

std::istream& operator>>(std::istream& in, BenchmarkType& type);

std::ostream& operator<<(std::ostream& out, const BenchmarkType& type);

struct run_args {
    BenchmarkType benchmark_type;
    std::string data_type;
    std::string alg_name;
    std::optional<std::filesystem::path> alg_args_path;
    std::filesystem::path ref_path;
    std::filesystem::path target_path;
    std::optional<std::filesystem::path> out_path;
    std::filesystem::path measurements_path;
    po::variable_value validate;
    std::size_t outer_loops;
    bool normalize;
    bool append_measurements;
    bool print_progress;
    // Min time for adaptive benchmarking
    std::chrono::nanoseconds min_time;

    run_args(
        BenchmarkType benchmark_type,
        std::string data_type,
        std::string alg_name,
        std::optional<std::filesystem::path> alg_args_path,
        std::filesystem::path ref_path,
        std::filesystem::path target_path,
        std::optional<std::filesystem::path> out_path,
        std::filesystem::path measurements_path,
        po::variable_value validate,
        std::size_t outer_loops,
        bool normalize,
        bool append_measurements,
        bool print_progress,
        std::chrono::nanoseconds min_time
    );

    static std::optional<run_args> from_variables_map(
        const po::variables_map& vm,
        const std::unordered_set<std::string>& alg_names
    );

    [[nodiscard]] std::optional<std::filesystem::path> get_loop_out_path(std::size_t loop) const;
};

}
