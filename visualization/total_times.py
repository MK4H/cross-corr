import argparse
import matplotlib.pyplot as plt

from pathlib import Path

from shared import InputSize, Benchmark, load_group_results

def per_matrix_size(args: argparse.Namespace):
    benchmarks = load_group_results(args.results_path)
    matrix_number_groups = set([(benchmark.input_size.left_matrices,benchmark.input_size.right_matrices) for benchmark in benchmarks])
    run_names = set(benchmark.name for benchmark in benchmarks)

    combinations = [(name, matrix_number[0], matrix_number[1]) for name in run_names for matrix_number in matrix_number_groups]

    for combination in combinations:
        matching = [
            benchmark for benchmark in benchmarks if benchmark.name == combination[0] and
                                                     benchmark.input_size.left_matrices == combination[1] and
                                                     benchmark.input_size.right_matrices == combination[2]
        ]
        matching.sort(key=lambda bench: bench.input_size.rows * bench.input_size.columns)
        plt.plot(
            [bench.input_size.rows * bench.input_size.columns for bench in matching],
            [bench.data["Total"].mean() for bench in matching],
            'o-',
            label=f"{combination[0]} {combination[1]} to {combination[2]}"
        )
    plt.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser("Plotting results measured by benchmarking scripts")
    parser.add_argument("results_path",
                        type=Path,
                        help="Path to results of a single group")

    args = parser.parse_args()
    per_matrix_size(args)

if __name__ == "__main__":
    main()
