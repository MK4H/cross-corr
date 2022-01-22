#include <iostream>
#include <fstream>
#include <filesystem>
#include <functional>
#include <unordered_map>
#include <algorithm>
#include <vector>

#include <boost/program_options.hpp>

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

// Fix filesystem::path not working with program options when argument contains spaces
// https://stackoverflow.com/questions/68716288/q-boost-program-options-using-stdfilesystempath-as-option-fails-when-the-gi
namespace std::filesystem {
    template <class CharT>
    void validate(boost::any& v, std::vector<std::basic_string<CharT>> const& s,
                  std::filesystem::path* p, int)
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
using data_type = float;

template<typename DATATYPE>
void validate(
    const std::filesystem::path& target_path,
    const std::filesystem::path& valid_path
){
    std::ifstream target_file(target_path);
    std::ifstream valid_file(valid_path);
    auto target = data_array<DATATYPE>::template load_from_csv<no_padding>(target_file);
    auto valid = data_array<DATATYPE>::template load_from_csv<no_padding>(valid_file);

    std::cout << validate_result(valid, target);
}

template<typename DURATION>
void output_measurements(
    const std::filesystem::path& measurements_path,
    const std::vector<std::string>& labels,
    const std::vector<DURATION>& measurements,
    bool append
) {
    std::ofstream measurements_file;
    if (append) {
        measurements_file.open(measurements_path, std::ios_base::app);
    } else {
        measurements_file.open(measurements_path);
        to_csv(measurements_file, labels);
    }

    to_csv<std::chrono::nanoseconds>(measurements_file, measurements);
}

template<typename ALG>
void run_measurement(
    const std::filesystem::path& ref_path,
    const std::filesystem::path& def_path,
    const std::filesystem::path& out_path,
    const std::filesystem::path& measurements_path,
    const po::variable_value& validate,
    bool normalize,
    bool append_measurements,
    bool print_progress
) {
    simple_logger logger{print_progress, append_measurements};
    ALG alg;
    logger.log("Loading inputs");
    alg.prepare(ref_path, def_path);

    logger.log("Running test alg");
    alg.run();

    logger.log("Copying output data to host");
    alg.finalize();

    auto res = alg.results();
    std::ofstream out_file(out_path);
    if (alg.is_fft() && normalize) {
        logger.log("Normalizing and storing results");
        auto norm = normalize_fft_results(res);
        norm.store_to_csv(out_file);
    } else {
        logger.log("Storing results");
        res.store_to_csv(out_file);
    }

    output_measurements(
        measurements_path,
        alg.measurement_labels(),
        alg.measurements(),
        append_measurements
    );


    if (validate.empty()) {
        logger.log("No validation");
    } else if (validate.as<std::filesystem::path>() != std::filesystem::path{}) {
        auto precomputed_data_path = validate.as<std::filesystem::path>();
        logger.log("Validating results agains "s + precomputed_data_path.u8string());
        logger.result_stats(alg.validate(precomputed_data_path));
    } else {
        logger.log("Computing valid results and validating");
        logger.result_stats(alg.validate());
    }
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
        {"one_to_one", one_to_one<data_type>::validate_input_size},
        {"one_to_many", one_to_many<data_type>::validate_input_size},
        {"n_to_mn", n_to_mn<data_type>::validate_input_size},
        {"n_to_m", n_to_m<data_type>::validate_input_size}
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





static std::unordered_map<std::string, std::function<void(
    const std::filesystem::path&,
    const std::filesystem::path&,
    const std::filesystem::path&,
    const std::filesystem::path&,
    const po::variable_value& validate,
    bool,
    bool,
    bool
)>> algorithms{
    {"cpu_one_to_one", run_measurement<cpu_one_to_one<data_type, false>>},
    {"cpu_one_to_many", run_measurement<cpu_one_to_many<data_type, false>>},
    {"cpu_n_to_mn", run_measurement<cpu_n_to_mn<data_type, false>>},
    {"cpu_n_to_m", run_measurement<cpu_n_to_m<data_type, false>>},
    {"nai_orig_one_to_one", run_measurement<naive_original_alg_one_to_one<data_type, false, pinned_allocator<data_type>>>},
    {"nai_orig_one_to_many", run_measurement<naive_original_alg_one_to_many<data_type, false, pinned_allocator<data_type>>>},
    {"nai_orig_n_to_mn", run_measurement<naive_original_alg_n_to_mn<data_type, false, pinned_allocator<data_type>>>},
    {"nai_rows_128", run_measurement<naive_ring_buffer_row_alg<data_type, 128, false, pinned_allocator<data_type>>>},
    {"nai_rows_256", run_measurement<naive_ring_buffer_row_alg<data_type, 256, false, pinned_allocator<data_type>>>},
    {"nai_def_block_128", run_measurement<naive_def_per_block<data_type, 128, false, pinned_allocator<data_type>>>},
    {"nai_def_block_256", run_measurement<naive_def_per_block<data_type, 256, false, pinned_allocator<data_type>>>},
    {"fft_orig_one_to_one", run_measurement<fft_original_alg_one_to_one<data_type, false, pinned_allocator<data_type>>>},
    {"fft_orig_one_to_many", run_measurement<fft_original_alg_one_to_many<data_type, false, pinned_allocator<data_type>>>},
    {"fft_orig_n_to_mn", run_measurement<fft_original_alg_n_to_mn<data_type, false, pinned_allocator<data_type>>>},
    {"fft_better_n_to_m", run_measurement<fft_better_hadamard_alg_n_to_m<data_type, false, pinned_allocator<data_type>>>}
};

void print_help(std::ostream& out, const std::string& name, const po::options_description& options) {
    out << "Usage: " << name << " [global options] command [run options] <alg> <ref_path> <target_path>\n";
    out << "Commands: \n";
    out << "\t" << name << " [global options] list\n";
    out << "\t" << name << " [global options] run [run options] <alg> <ref_path> <target_path>\n";
    out << "\t" << name << " [global options] validate [validate options] <validate_data_path> <template_data_path>\n";
    out << "\t" << name << " [global options] input [input options] <alg_type> <rows> <columns> <left_matrices> <right_matrices>\n";
    out << options;
}

int main(int argc, char **argv) {
    try {
        std::string alg_name;
        std::filesystem::path ref_path;
        std::filesystem::path target_path;
        std::filesystem::path out_path{"output.csv"};
        std::filesystem::path measurements_path{"measurements.csv"};

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

        po::options_description val_pos_opts;
        val_pos_opts.add_options()
            ("validate_data_path", po::value<std::filesystem::path>(&target_path)->required(), "path to the data to be validated")
            ("template_data_path", po::value<std::filesystem::path>(&ref_path)->required(), "path to the valid data")
            ;

        po::positional_options_description val_positional;
        val_positional.add("validate_data_path", 1);
        val_positional.add("template_data_path", 1);

        po::options_description all_val_options;
        all_val_options.add(val_opts);
        all_val_options.add(val_pos_opts);

        po::options_description run_opts{"Run options"};
        run_opts.add_options()
            ("out,o", po::value<std::filesystem::path>(&out_path)->default_value(out_path))
            ("times,t", po::value<std::filesystem::path>(&measurements_path)->default_value(measurements_path))
            ("validate,v", po::value<std::filesystem::path>()->implicit_value(""))
            ("normalize,n", po::bool_switch()->default_value(false), "If algorithm is fft, normalize the results")
            ("append,a", po::bool_switch()->default_value(false), "Append time measurements without the header if the times file already exists instead of overwriting it")
            ("no_progress,p", po::bool_switch()->default_value(false), "Do not print human readable progress, instead any messages to stdout will be formated for machine processing")
            ;

        po::options_description run_pos_opts;
        run_pos_opts.add_options()
            ("alg", po::value<std::string>(&alg_name)->required())
            ("ref_path", po::value<std::filesystem::path>(&ref_path)->required(), "path to the reference data")
            ("target_path", po::value<std::filesystem::path>(&target_path)->required(), "path to the target data")
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

            auto fnc = algorithms.find(alg_name);
            if (fnc == algorithms.end()) {
                std::cerr << "Unknown algorithm \"" << alg_name << "\", expected one of " << get_sorted_keys(algorithms) << std::endl;
                print_help(std::cerr, argv[0], all_options);
                return 1;
            }

            auto normalize = vm["normalize"].as<bool>();
            auto append = vm["append"].as<bool>();
            auto progress = !vm["no_progress"].as<bool>();
            auto validate = vm["validate"];
            fnc->second(ref_path, target_path, out_path, measurements_path, validate, normalize, append, progress);
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

            validate<double>(target_path, ref_path);
        } else if (cmd == "list") {
            auto algs = get_sorted_keys(algorithms);
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
    catch (std::exception& e) {
        std::cerr << "Exception occured: " << e.what() << std::endl;
        return 2;
    }
}