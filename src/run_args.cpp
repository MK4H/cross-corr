#include "run_args.hpp"

#include <iostream>
#include <filesystem>
#include <chrono>
#include <unordered_set>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

#include "host_helpers.hpp"

namespace cross {

namespace po = boost::program_options;

std::istream& operator>>(std::istream& in, BenchmarkType& type) {
    std::string token;
    in >> token;
    if (boost::iequals(token, "Compute")) {
        type = BenchmarkType::Compute;
    } else if (boost::iequals(token, "CommonSteps")) {
        type = BenchmarkType::CommonSteps;
    } else if (boost::iequals(token, "Algorithm")) {
        type = BenchmarkType::Algorithm;
    } else if (boost::iequals(token, "None")) {
        type = BenchmarkType::None;
    } else {
        type = BenchmarkType::Unknown;
    }
    return in;
}

std::ostream& operator<<(std::ostream& out, const BenchmarkType& type) {
    switch (type) {
        case BenchmarkType::Unknown:
            out << "Unknown";
            break;
        case BenchmarkType::None:
            out << "None";
            break;
        case BenchmarkType::Compute:
            out << "Compute";
            break;
        case BenchmarkType::CommonSteps:
            out << "CommonSteps";
            break;
        case BenchmarkType::Algorithm:
            out << "Algorithm";
            break;
    }
    return out;
}

run_args::run_args(
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
) : benchmark_type(benchmark_type),
    data_type(std::move(data_type)),
    alg_name(std::move(alg_name)),
    alg_args_path(std::move(alg_args_path)),
    ref_path(std::move(ref_path)),
    target_path(std::move(target_path)),
    out_path(std::move(out_path)),
    measurements_path(std::move(measurements_path)),
    validate(std::move(validate)),
    outer_loops(outer_loops),
    normalize(normalize),
    append_measurements(append_measurements),
    print_progress(print_progress),
    min_time(min_time)
{
}

std::optional<run_args> run_args::from_variables_map(
    const po::variables_map &vm,
    const std::unordered_set<std::string> &alg_names
) {
    auto benchmark_type = vm["benchmark_type"].as<BenchmarkType>();
    auto data_type = vm["data_type"].as<std::string>();
    auto alg_name = vm["alg"].as<std::string>();

    auto args_path = vm.count("args_path") ? std::optional<std::filesystem::path>{vm["args_path"].as<std::filesystem::path>()} : std::nullopt;
    auto ref_path = vm["ref_path"].as<std::filesystem::path>();
    auto target_path = vm["target_path"].as<std::filesystem::path>();

    auto out_path_arg = vm["out"].as<std::filesystem::path>();
    auto out_path = !out_path_arg.empty() ? std::optional<std::filesystem::path>(out_path_arg) : std::nullopt;

    auto measurements_path = vm["times"].as<std::filesystem::path>();

    auto validate = vm["validate"];
    auto outer_loops = vm["outer_loops"].as<std::size_t>();
    auto normalize = vm["normalize"].as<bool>();
    auto append = vm["append"].as<bool>();
    auto progress = !vm["no_progress"].as<bool>();

    auto min_time_secs = vm["min_time"].as<double>();
    auto min_time = std::chrono::duration<double>(min_time_secs);

    // TODO: Change if there can be different algorithms for float and double
    if (alg_names.find(alg_name) == alg_names.end()) {
        std::cerr << "Unknown algorithm \"" << alg_name << "\", expected one of " << get_sorted_values(alg_names) << std::endl;
        return std::nullopt;
    }

    if (benchmark_type == BenchmarkType::Unknown) {
        std::cerr << "Unknown benchmark type" << std::endl;
        return std::nullopt;
    }

    return run_args(
        benchmark_type,
        std::move(data_type),
        std::move(alg_name),
        std::move(args_path),
        std::move(ref_path),
        std::move(target_path),
        std::move(out_path),
        std::move(measurements_path),
        std::move(validate),
        outer_loops,
        normalize,
        append,
        progress,
        std::chrono::duration_cast<std::chrono::nanoseconds>(min_time)
    );
}

[[nodiscard]] std::optional<std::filesystem::path> run_args::get_loop_out_path(std::size_t loop) const {
    using namespace std::literals;

    if (!out_path.has_value()) {
        return out_path;
    }
    auto loop_out_path = out_path.value();
    return loop_out_path.replace_filename(out_path->stem().string() + "_"s + std::to_string(loop) + out_path->extension().string());
}

}
