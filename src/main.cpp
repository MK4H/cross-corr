#include <iostream>
#include <filesystem>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/program_options.hpp>

#include "run_args.hpp"
#include "simple_logger.hpp"
#include "validate.hpp"
#include "matrix.hpp"
#include "cross_corr.hpp"
#include "allocator.cuh"
#include "csv.hpp"
#include "fft_helpers.hpp"
#include "one_to_one.hpp"
#include "one_to_many.hpp"
#include "n_to_mn.hpp"
#include "n_to_m.hpp"

#include "argument_error.hpp"
#include "host_helpers.hpp"

// Fix filesystem::path not working with program options when argument contains spaces
// https://stackoverflow.com/questions/68716288/q-boost-program-options-using-stdfilesystempath-as-option-fails-when-the-gi
namespace std::filesystem {
    template <class CharT>
    void validate(boost::any& v, std::vector<std::basic_string<CharT>> const& s,
                  [[maybe_unused]] std::filesystem::path* p, int)
    {
        assert(s.size() == 1);
        std::basic_stringstream<CharT> ss;

        for (auto& el : s)
            ss << std::quoted(el);

        path converted;
        ss >> std::noskipws >> converted;

        if (ss.peek(); !ss.eof())
            throw std::runtime_error("excess path characters");

        v = std::move(converted);
    }
}

using namespace cross;

namespace po = boost::program_options;

template<typename DATATYPE>
void validate(
    const std::filesystem::path& valid_path,
    const std::vector<std::filesystem::path>& target_paths,
    bool normalize,
    bool csv,
    bool print_header
){
    simple_logger logger{!csv, true};

    std::ifstream valid_file(valid_path);

    auto valid = data_array<DATATYPE>::template load_from_csv<no_padding>(valid_file);

    for (std::size_t i = 0; i < target_paths.size(); ++i) {
        logger.log("Validating "s + target_paths[i].string());
        std::ifstream target_file(target_paths[i]);
        auto target = data_array<DATATYPE>::template load_from_csv<no_padding>(target_file);
        if (normalize) {
            target = normalize_fft_results(target);
        }
        logger.result_stats(validate_result(target, valid), i == 0 && print_header);
    }
}

std::vector<std::string> with_iteration_labels(const std::vector<std::string>& labels) {
    std::vector<std::string> out{labels.size() * 2};
    for (std::size_t i = 0; i < labels.size(); ++i) {
        out[2*i] = labels[i];
        out[2*i + 1] = labels[i] + "_iterations";
    }
    return out;
}

template<typename DURATION>
void output_measurements(
    const std::filesystem::path& measurements_path,
    const std::vector<std::string>& labels,
    const std::vector<measurement_result<DURATION>>& measurements,
    const std::vector<std::pair<std::string, std::string>>& additional_properties,
    bool append
) {
    std::ofstream measurements_file;
    if (append) {
        measurements_file.open(measurements_path, std::ios_base::app);
    } else {
        measurements_file.open(measurements_path);
        to_csv(measurements_file, with_iteration_labels(labels));
        if (!additional_properties.empty()) {
            if (!labels.empty()) {
                measurements_file << ",";
            }
            to_csv(measurements_file, get_labels(additional_properties));
        }
        measurements_file << "\n";
    }

    to_csv<std::chrono::nanoseconds>(measurements_file, measurements);
    if (!additional_properties.empty()) {
        if (!measurements.empty()) {
            measurements_file << ",";
        }
        to_csv(measurements_file, get_values(additional_properties));
    }
    measurements_file << "\n";
}

template<typename ALG>
int run_computation_steps(
    ALG& alg,
    simple_logger& logger
) {
    try {
        logger.log("Allocating");
        alg.prepare();
    } catch (std::exception& e) {
        std::cerr << "Exception occured durign allocation step: " << e.what() << std::endl;
        return 3;
    }

    try {
        logger.log("Transfering data");
        alg.transfer();
    } catch (std::exception& e) {
        std::cerr << "Exception occured durign data transfer step: " << e.what() << std::endl;
        return 3;
    }

    try {
        logger.log("Running test alg");
        alg.run();
    } catch (std::exception& e) {
        std::cerr << "Exception occured durign computation step: " << e.what() << std::endl;
        return 3;
    }

    try {
        logger.log("Copying output data to host");
        alg.finalize();
    } catch (std::exception& e) {
        std::cerr << "Exception occured durign finalization step: " << e.what() << std::endl;
        return 3;
    }

    try {
        logger.log("Free resources");
        alg.free();
    } catch (std::exception& e) {
        std::cerr << "Exception occured durign free step: " << e.what() << std::endl;
        return 3;
    }

    return 0;
}

template<typename ALG>
void run_store_output(
    ALG& alg,
    simple_logger& logger,
    const std::optional<std::filesystem::path>& out_path,
    bool normalize
) {
    if (out_path.has_value()) {
        const auto& res = alg.results();
        std::ofstream out_file(out_path.value());
        if (alg.is_fft() && normalize) {
            logger.log("Normalizing and storing results");
            auto norm = normalize_fft_results(res);
            norm.store_to_csv(out_file);
        } else {
            logger.log("Storing results");
            res.store_to_csv(out_file);
        }
    }
}

template<typename ALG>
void run_validate(
    ALG& alg,
    simple_logger& logger,
    bool print_header,
    const po::variable_value& validate
) {
    if (validate.empty()) {
        logger.log("No validation");
    } else if (validate.as<std::filesystem::path>() != std::filesystem::path{}) {
        auto precomputed_data_path = validate.as<std::filesystem::path>();
        logger.log("Validating results against "s + precomputed_data_path.u8string());
        logger.result_stats(alg.validate(precomputed_data_path), print_header);
    } else {
        logger.log("Computing valid results and validating");
        logger.result_stats(alg.validate(), print_header);
    }
}

template<typename ALG>
int run_measurement(
    const run_args& run_args
) {
    simple_logger logger{run_args.print_progress && ALG::benchmarking_type != BenchmarkType::Compute, run_args.append_measurements};

    std::vector<std::string> compute_labels{"Computation"};
    stopwatch<std::chrono::high_resolution_clock> sw{compute_labels.size(), run_args.min_time};

    json alg_args;
    if (run_args.alg_args_path) {
        std::ifstream args_file{*run_args.alg_args_path};
        args_file >> alg_args;
    } else {
        alg_args = json::object();
    }

    ALG alg{alg_args, run_args.min_time};

    try {
        logger.log("Loading inputs");
        alg.load(run_args.ref_path, run_args.target_path);
    } catch (std::exception& e) {
        std::cerr << "Exception occured durign data loading step: " << e.what() << std::endl;
        return 3;
    }


    bool do_compute_measurement = ALG::benchmarking_type == BenchmarkType::Compute;
    for (std::size_t loop = 0; loop < run_args.outer_loops; ++loop) {
        CPU_ADAPTIVE_MEASURE(0, do_compute_measurement, sw, true,
            int ret = run_computation_steps(alg, logger);
            if (ret != 0) {
                return ret;
            }
        );

        // TODO: Maybe integrate into the algorithm itself, maybe free or finalize step
        try {
            CUCH(cudaDeviceSynchronize());
            CUCH(cudaGetLastError());
        } catch (std::exception& e) {
            std::cerr << "Exception somewhere in CUDA code: " << e.what() << std::endl;
            return 3;
        }


        if (ALG::benchmarking_type != BenchmarkType::None) {
            if (!do_compute_measurement) {
                try {
                    logger.log("Collect measurements");
                    alg.collect_measurements();
                } catch (std::exception &e) {
                    std::cerr << "Exception occured durign measurement collection step: " << e.what() << std::endl;
                    return 3;
                }
            }

            output_measurements(
                run_args.measurements_path,
                do_compute_measurement ? compute_labels : alg.measurement_labels(),
                do_compute_measurement ? sw.results() : alg.measurements(),
                alg.additional_properties(),
                run_args.append_measurements || loop > 0
            );

            if (!do_compute_measurement) {
                try {
                    logger.log("Reset measurement");
                    alg.reset_measurements();
                } catch (std::exception &e) {
                    std::cerr << "Exception occured durign measurement reset step: " << e.what() << std::endl;
                    return 3;
                }
            }
        }

        run_store_output(
            alg,
            logger,
            run_args.outer_loops > 1 ? run_args.get_loop_out_path(loop) : run_args.out_path,
            run_args.normalize
        );

        run_validate(
            alg,
            logger,
            !run_args.append_measurements && loop == 0,
            run_args.validate
        );

    }
    return 0;
}

int validate_input_size(
    const std::string& alg_type,
    dsize_t rows,
    dsize_t columns,
    dsize_t left_matrices,
    dsize_t right_matrices
) {
    static std::unordered_map<std::string, std::function<bool(
        dsize_t,
        dsize_t,
        dsize_t,
        dsize_t
    )>> input_size_validation{
        {"one_to_one", one_to_one_validate_input_size},
        {"one_to_many", one_to_many_validate_input_size},
        {"n_to_mn", n_to_mn_validate_input_size},
        {"n_to_m", n_to_m_validate_input_size}
    };

    auto validator = input_size_validation.find(alg_type);
    if (validator == input_size_validation.end()) {
        std::cerr << "Unknown algorithm type \"" << alg_type << "\", expected one of " << get_sorted_keys(input_size_validation) << std::endl;
        return 1;
    }

    if (validator->second(rows, columns, left_matrices, right_matrices)) {
        std::cout << "Valid\n";
    } else {
        std::cout << "Invalid\n";
    }
    return 0;
}

template<typename DATA_TYPE, BenchmarkType BENCH_TYPE>
std::unordered_map<std::string, std::function<int(
    const run_args&
)>> get_algorithms() {
    return std::unordered_map<std::string, std::function<int(
        const run_args&
    )>>{
        {"cpu_one_to_one", run_measurement<cpu_one_to_one<DATA_TYPE, BENCH_TYPE>>},
        {"cpu_one_to_many", run_measurement<cpu_one_to_many<DATA_TYPE, BENCH_TYPE>>},
        {"cpu_n_to_mn", run_measurement<cpu_n_to_mn<DATA_TYPE, BENCH_TYPE>>},
        {"cpu_n_to_m", run_measurement<cpu_n_to_m<DATA_TYPE, BENCH_TYPE>>},
        {"nai_orig_one_to_one", run_measurement<naive_original_alg_one_to_one<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>},
        {"nai_orig_one_to_many", run_measurement<naive_original_alg_one_to_many<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>},
        {"nai_orig_n_to_mn", run_measurement<naive_original_alg_n_to_mn<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>},
        {"nai_orig_n_to_m", run_measurement<naive_original_alg_n_to_m<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>},
        {"nai_warp_shuffle_one_to_one", run_measurement<naive_warp_shuffle_one_to_one<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>},
        {"nai_warp_shuffle_one_to_many", run_measurement<naive_warp_shuffle_one_to_many<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>},
        {"nai_warp_shuffle_work_distribution_one_to_one", run_measurement<naive_warp_shuffle_work_distribution_one_to_one<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>},
        {"nai_warp_shuffle_work_distribution_one_to_many", run_measurement<naive_warp_shuffle_work_distribution_one_to_many<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>},
        {"nai_warp_shuffle_work_distribution_n_to_m", run_measurement<naive_warp_shuffle_work_distribution_n_to_m<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>},
        {"nai_shuffle_specialized_n_to_m_orig", run_measurement<naive_shuffle_specialized_n_to_m_orig<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>},
        {"nai_shuffle_specialized_n_to_m", run_measurement<naive_shuffle_specialized_n_to_m<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>},
        {"nai_multirow_shuffle_one_to_one", run_measurement<naive_multirow_shuffle_one_to_one<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>},
        {"nai_multileft_shuffle_one_to_one", run_measurement<naive_multileft_shuffle_one_to_one<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>},
        {"nai_multileft_shuffle_one_to_one_orig", run_measurement<naive_multileft_shuffle_one_to_one_orig<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>},
        {"nai_multirow_multiright_shuffle_one_to_one", run_measurement<naive_multirow_multiright_shuffle_one_to_one<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>},
        {"nai_multirow_multiright_shuffle_one_to_many", run_measurement<naive_multirow_multiright_shuffle_one_to_many<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>},
        {"nai_shift_per_warp_one_to_one", run_measurement<naive_shift_per_warp_one_to_one<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>},
        {"nai_shift_per_warp_simple_indexing_one_to_one", run_measurement<naive_shift_per_warp_simple_indexing_one_to_one<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>},
        {"nai_shift_per_warp_work_distribution_one_to_one", run_measurement<naive_shift_per_warp_work_distribution_one_to_one<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>},
        {"nai_shift_per_warp_shared_mem_rows_one_to_one", run_measurement<naive_shift_per_warp_shared_mem_rows_one_to_one<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>},
        {"nai_shift_per_warp_shared_mem_rows_one_to_many", run_measurement<naive_shift_per_warp_shared_mem_rows_one_to_many<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>},
        {"nai_shift_per_block_one_to_one", run_measurement<naive_shift_per_block_one_to_one<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>},
        {"fft_orig_one_to_one", run_measurement<fft_original_alg_one_to_one<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>},
        {"fft_reduced_transfer_one_to_one", run_measurement<fft_reduced_transfer_one_to_one<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>},
        {"fft_orig_one_to_many", run_measurement<fft_original_alg_one_to_many<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>},
        {"fft_reduced_transfer_one_to_many", run_measurement<fft_reduced_transfer_one_to_many<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>},
        {"fft_orig_n_to_mn", run_measurement<fft_original_alg_n_to_mn<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>},
        {"fft_reduced_transfer_n_to_mn", run_measurement<fft_reduced_transfer_n_to_mn<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>},
        {"fft_better_n_to_m", run_measurement<fft_better_hadamard_alg_n_to_m<DATA_TYPE, BENCH_TYPE, pinned_allocator<DATA_TYPE>>>}
    };
}



template<typename DATA_TYPE, BenchmarkType BENCH_TYPE>
int run_alg_type_dispatch(
    const run_args& args
) {
    auto algs = get_algorithms<DATA_TYPE, BENCH_TYPE>();
    auto fnc = algs.find(args.alg_name);
    if (fnc == get_algorithms<DATA_TYPE, BENCH_TYPE>().end()) {
        throw std::runtime_error("Invalid algorithm specified \n"s + args.alg_name + "\"");
    }
    return fnc->second(args);
}

template<BenchmarkType BENCH_TYPE>
int run_data_type_dispatch(
    const run_args& args
) {
    if (args.data_type == "single") {
        return run_alg_type_dispatch<float, BENCH_TYPE>(args);
    } else if (args.data_type == "double") {
        return run_alg_type_dispatch<double, BENCH_TYPE> (args);
    } else {
        std::cerr << "Unknown data type " << args.data_type << "\n";
        return 1;
    }
}

int run_benchmark_type_dispatch(
    const run_args& args
) {
    switch (args.benchmark_type) {
        case BenchmarkType::None:
            return run_data_type_dispatch<BenchmarkType::None>(args);
        case BenchmarkType::Compute:
            return run_data_type_dispatch<BenchmarkType::Compute>(args);
        case BenchmarkType::CommonSteps:
            return run_data_type_dispatch<BenchmarkType::CommonSteps>(args);
        case BenchmarkType::Algorithm:
            return run_data_type_dispatch<BenchmarkType::Algorithm>(args);
        default:
            std::cerr << "Unknown benchmark type " << args.benchmark_type << "\n";
            return 1;
    }
}

int run(
    const run_args& args
) {
    return run_benchmark_type_dispatch(args);
}

void print_help(std::ostream& out, const std::string& name, const po::options_description& options) {
    out << "Usage: " << name << " [global options] command [command options]\n";
    out << "Commands: \n";
    out << "\t" << name << " [global options] list\n";
    out << "\t" << name << " [global options] run [run options] <alg> <ref_path> <target_path>\n";
    out << "\t" << name << " [global options] validate [validate options] <validate_data_path> <template_data_path>\n";
    out << "\t" << name << " [global options] input [input options] <alg_type> <rows> <columns> <left_matrices> <right_matrices>\n";
    out << options;
}

int main(int argc, char **argv) {
    try {
        // TODO: Add handling of -- to separate options from positional arguments as program options doesn't do this by itself
        po::options_description global_opts{"Global options"};
        global_opts.add_options()
            ("help,h", "display this help and exit")
            ("command", po::value<std::string>(), "command to execute")
            ("subargs", po::value<std::vector<std::string> >(), "Arguments for command")
            ;

        po::positional_options_description global_positional;
        global_positional.
            add("command", 1).
            add("subargs", -1);


        po::options_description val_opts{"Validate options"};
        val_opts.add_options()
            ("normalize,n", po::bool_switch()->default_value(false), "Normalize the data to be validated as they are denormalized fft output")
            ("csv,c", po::bool_switch()->default_value(false), "CSV output")
            ("print_header,p",po::bool_switch()->default_value(false), "Print header")
            ;

        po::options_description val_pos_opts;
        val_pos_opts.add_options()
            ("template_data_path", po::value<std::filesystem::path>()->required(), "path to the valid data")
            ("validate_data_path", po::value<std::vector<std::filesystem::path>>()->required()->multitoken(), "path to the data to be validated")
            ;

        po::positional_options_description val_positional;
        val_positional.add("template_data_path", 1);
        val_positional.add("validate_data_path", -1);

        po::options_description all_val_options;
        all_val_options.add(val_opts);
        all_val_options.add(val_pos_opts);

        po::options_description run_opts{"Run options"};
        run_opts.add_options()
            ("data_type,d", po::value<std::string>()->default_value("single"), "Data type to use for computation")
            ("out,o", po::value<std::filesystem::path>()->default_value(std::filesystem::path{}), "Path of the output file to be created")
            ("times,t", po::value<std::filesystem::path>()->default_value("measurements.csv"), "File to store the measured times in")
            ("validate,v", po::value<std::filesystem::path>()->implicit_value(""), "If validation of the results should be done and optionally path to a file containing the valid results")
            ("normalize,n", po::bool_switch()->default_value(false), "If algorithm is fft, normalize the results")
            ("append,a", po::bool_switch()->default_value(false), "Append time measurements without the header if the times file already exists instead of overwriting it")
            ("no_progress,p", po::bool_switch()->default_value(false), "Do not print human readable progress, instead any messages to stdout will be formated for machine processing")
            ("benchmark_type,b", po::value<BenchmarkType>()->default_value(BenchmarkType::Compute), "Which part should be measured, available parts are Compute, CommonSteps, Algorithm")
            ("outer_loops,l", po::value<std::size_t>()->default_value(1), "How many measurements of the computation loop should be done with the loaded data")
            ("min_time,m", po::value<double>()->default_value(1), "The minimum time to consider measurement statistically relevant, in seconds")
            ("args_path", po::value<std::filesystem::path>(), "Path to the JSON file containing arguments for the algorithm")
            ;

        po::options_description run_pos_opts;
        run_pos_opts.add_options()
            ("alg", po::value<std::string>()->required(), "Name of the algorithm to use")
            ("ref_path", po::value<std::filesystem::path>()->required(), "path to the reference data")
            ("target_path", po::value<std::filesystem::path>()->required(), "path to the target data")
            ;

        po::options_description all_run_options;
        all_run_options.add(run_opts);
        all_run_options.add(run_pos_opts);

        po::positional_options_description run_positional;
        run_positional.add("alg", 1);
        run_positional.add("ref_path", 1);
        run_positional.add("target_path", 1);

        po::options_description input_opts{"Input options"};
        input_opts.add_options()
           ;

        po::options_description input_pos_opts;
        input_pos_opts.add_options()
            ("alg_type", po::value<std::string>()->required(), "Type of the algorithm to validate thei input dimensions for")
            ("rows", po::value<dsize_t>()->required(), "Number of rows of each input matrix")
            ("columns", po::value<dsize_t>()->required(), "Number of columns of each input matrix")
            ("left_matrices", po::value<dsize_t>()->required(), "Number of left input matrices (for n_to_m, this would be the n)")
            ("right_matrices", po::value<dsize_t>()->required(), "Number of right input matrices (for n_to_m, this would be the m")
            ;

        po::options_description all_input_options;
        all_input_options.add(input_opts);
        all_input_options.add(input_pos_opts);

        po::positional_options_description input_positional;
        input_positional.add("alg_type", 1);
        input_positional.add("rows", 1);
        input_positional.add("columns", 1);
        input_positional.add("left_matrices", 1);
        input_positional.add("right_matrices", 1);


        po::options_description all_options;
        all_options.add(global_opts);
        all_options.add(all_val_options);
        all_options.add(all_run_options);
        all_options.add(all_input_options);

        po::parsed_options parsed = po::command_line_parser(argc, argv).
                options(global_opts).
                positional(global_positional).
                allow_unregistered().
                run();
        po::variables_map vm;
        po::store(parsed, vm);

        if (vm.count("help")) {
            print_help(std::cout, argv[0], all_options);
            return 0;
        }
        po::notify(vm);

        std::string cmd = vm["command"].as<std::string>();

        if (cmd == "run") {
            std::vector<std::string> opts = po::collect_unrecognized(parsed.options, po::include_positional);

            // Remove the command name
            opts.erase(opts.begin());

            po::store(
                po::command_line_parser(opts).
                    options(all_run_options).
                    positional(run_positional).
                    run(),
                vm
            );
            po::notify(vm);

            auto run_args = run_args::from_variables_map(vm, get_key_set(get_algorithms<float, BenchmarkType::Unknown>()));
            if (!run_args.has_value()) {
                print_help(std::cerr, argv[0], all_options);
                return 1;
            }
            auto ret = run(run_args.value());
            if (ret == 1) {
                print_help(std::cerr, argv[0], all_options);
            }
            return ret;

        } else if (cmd == "validate") {
            std::vector<std::string> opts = po::collect_unrecognized(parsed.options, po::include_positional);
            opts.erase(opts.begin());

            po::store(
                po::command_line_parser(opts).
                    options(all_val_options).
                    positional(val_positional).
                    run(),
                vm
            );
            po::notify(vm);

            auto normalize = vm["normalize"].as<bool>();
            auto csv = vm["csv"].as<bool>();
            auto print_header = vm["print_header"].as<bool>();
            auto template_data = vm["template_data_path"].as<std::filesystem::path>();
            auto validate_data = vm["validate_data_path"].as<std::vector<std::filesystem::path>>();

            validate<double>(template_data, validate_data, normalize, csv, print_header);
        } else if (cmd == "list") {
            auto algs = get_sorted_keys(get_algorithms<float, BenchmarkType::Unknown>());
            for (auto&& alg: algs) {
                std::cout << alg << "\n";
            }
        } else if (cmd == "input") {
            std::vector<std::string> opts = po::collect_unrecognized(parsed.options, po::include_positional);
            opts.erase(opts.begin());

            po::store(
                po::command_line_parser(opts).
                    options(all_input_options).
                    positional(input_positional).
                    run(),
                vm
            );
            po::notify(vm);

            auto alg_type = vm["alg_type"].as<std::string>();
            auto rows = vm["rows"].as<dsize_t>();
            auto columns = vm["columns"].as<dsize_t>();
            auto left_matrices = vm["left_matrices"].as<dsize_t>();
            auto right_matrices = vm["right_matrices"].as<dsize_t>();
            auto ret = validate_input_size(alg_type, rows, columns, left_matrices, right_matrices);
            if (ret != 0) {
                print_help(std::cerr, argv[0], all_options);
            }
            return ret;
        } else {
            std::cerr << "Unknown command " << cmd << "\n";
            print_help(std::cerr, argv[0], all_options);
            return 1;
        }
        return 0;
    }
    catch (po::error& e) {
        std::cerr << "Invalid commandline options: " << e.what() << std::endl;
        return 1;
    }
    catch (argument_error& e) {
        std::cerr << "Invalid algorithm argument value: " << e.what() << std::endl;
        return 1;
    }
    catch (std::exception& e) {
        std::cerr << "Exception occured: " << e.what() << std::endl;
        return 2;
    }
}
