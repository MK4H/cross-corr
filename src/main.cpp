#include <iostream>
#include <fstream>
#include <filesystem>
#include <functional>
#include <unordered_map>
#include <algorithm>
#include <vector>

#include <boost/program_options.hpp>

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

template<typename ALG>
void run_measurement(
    const std::filesystem::path& ref_path,
    const std::filesystem::path& def_path,
    const std::filesystem::path& out_path,
    const std::filesystem::path& measurements_path,
    const po::variable_value& validate,
    bool normalize
) {
    ALG alg;
    std::cerr << "Loading inputs\n";
    alg.prepare(ref_path, def_path);

    std::cerr << "Running test alg\n";
    alg.run();

    std::cerr << "Copying output data to host\n";
    alg.finalize();

    auto res = alg.results();
    std::ofstream out_file(out_path);
    if (alg.is_fft() && normalize) {
        std::cerr << "Normalizing and storing results\n";
        auto norm = normalize_fft_results(res);
        norm.store_to_csv(out_file);
    } else {
        std::cerr << "Storing results\n";
        res.store_to_csv(out_file);
    }

    std::ofstream measurements_file(measurements_path);
    auto labels = alg.measurement_labels();
    auto measurements = alg.measurements();
    to_csv(measurements_file, labels);
    to_csv<std::chrono::nanoseconds>(measurements_file, measurements);


    if (validate.empty()) {
        std::cerr << "No validation\n";
    } else if (validate.as<std::filesystem::path>() != std::filesystem::path{}) {
        auto precomputed_data_path = validate.as<std::filesystem::path>();
        std::cerr << "Validating results agains " << precomputed_data_path << "\n";
        std::cout << alg.validate(precomputed_data_path);
    } else {
        std::cerr << "Computing valid results and validating\n";
        std::cout << alg.validate();
    }
}




static std::unordered_map<std::string, std::function<void(
    const std::filesystem::path&,
    const std::filesystem::path&,
    const std::filesystem::path&,
    const std::filesystem::path&,
    const po::variable_value& validate,
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
    out << "\t" << name << " [global options] run [run options] <alg> <ref_path> <target_path>\n";
    out << "\t" << name << " [global options] validate [validate options] <alg_type> <validate_data_path> <template_data_path>\n";
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
            ("normalize,n", "If algorithm is fft, normalize the results")
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

        po::options_description all_options;
        all_options.add(global_opts);
        all_options.add(all_val_options);
        all_options.add(all_run_options);

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

            auto validate = vm["validate"];
            fnc->second(ref_path, target_path, out_path, measurements_path, validate, vm.count("normalize") != 0);
            return 0;
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
            return 0;
        } else {
            std::cerr << "Unknown command " << cmd << "\n";
            print_help(std::cerr, argv[0], all_options);
            return 1;
        }
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