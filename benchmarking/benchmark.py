import argparse
import re
import shutil
import sys

import input_generator

from typing import List, Tuple
from pathlib import Path

from external import input_size, validator, executable

DEFAULT_OUTPUT_PATH = Path.cwd() / "output"
DEFAULT_ITERATIONS = 100

alg_type_regex = re.compile(".*_([^_]+_to_[^_]+)$")
def get_algorithm_type(algorithm: str) -> str:
    match = alg_type_regex.fullmatch(algorithm)
    if match:
        return match.group(1)
    raise ValueError(f"Invalid algorithm name {algorithm}")


def generate_inputs(
        index: int,
        data_dir: Path,
        sizes: input_size.InputSize,
        keep_data: bool
) -> Tuple[Path, Path]:
    left_path = data_dir / (f"{index}_left_{sizes.rows}_{sizes.columns}_{sizes.left_matrices}.csv" if keep_data else "left.csv")
    right_path = data_dir / (f"{index}_right_{sizes.rows}_{sizes.columns}_{sizes.right_matrices}.csv" if keep_data else "right.csv")

    input_generator.generate_matrices(sizes.left_matrices, sizes.rows, sizes.columns, input_generator.OutputFormats.CSV, left_path)
    input_generator.generate_matrices(sizes.right_matrices, sizes.rows, sizes.columns, input_generator.OutputFormats.CSV, right_path)

    return left_path, right_path


def run_bechmarks(
        exec: executable.Executable,
        valid: validator.Validator,
        algs: List[str],
        sizes: List[input_size.InputSize],
        iter: int,
        validate: bool,
        keep_data: bool,
        prevent_overwrite: bool,
        out_dir: Path
):
    assert len(algs) != 0, "No algorithms given"
    assert len(sizes) != 0, "No input sizes given"
    assert iter > 0, f"Invalid number of iterations \"{iter}\" given"
    try:
        alg_type = get_algorithm_type(algs[0])
        for alg in algs:
            if get_algorithm_type(alg) != alg_type:
                print(
                    f"All algorithms have to be of the same type, got {get_algorithm_type(alg)} and {alg_type} types",
                    file=sys.stderr,
                )
                sys.exit(1)
    except ValueError as e:
        print(e)
        sys.exit(1)

    for in_size in sizes:
        if not exec.validate_input_size(alg_type, in_size):
            print(
                f"Input size {in_size} cannot be used with algorithms of type {alg_type}",
                file=sys.stderr,
            )
            sys.exit(1)

    try:
        out_dir.mkdir(exist_ok=not prevent_overwrite, parents=True)
    except FileExistsError:
        print(
            f"Output directory {out_dir} already exists and overwrite_check was enabled. Disable with --no_overwrite_check",
            file=sys.stderr,
        )
        sys.exit(3)

    data_dir = out_dir / "data"
    data_dir.mkdir(exist_ok=True, parents=True)

    print(f"All files will be stored under {str(out_dir.absolute())}")
    for idx, in_size in enumerate(sizes):
        print(f"Generating inputs of size {in_size}")
        left_path, right_path = generate_inputs(idx, data_dir, in_size, keep_data)

        if validate:
            print(f"Generating validation data")
            validation_data_path = data_dir / f"{idx}_valid_{in_size.rows}_{in_size.columns}_{in_size.left_matrices}_{in_size.right_matrices}.csv"
            valid.generate_validation_data(
                alg_type,
                left_path,
                right_path,
                validation_data_path
            )
        else:
            validation_data_path = None

        for alg in algs:
            msg = f"Benchmarking {alg} for {in_size}"
            print(msg)

            measurement_suffix = f"{alg}_{in_size}"
            measurement_data_dir = data_dir / f"output_{measurement_suffix}"
            measurement_data_dir.mkdir(exist_ok=True, parents=True)

            measurement_results_path = out_dir / f"{measurement_suffix}_time.csv"
            measurement_output_stats_path = out_dir / f"{measurement_suffix}_output_stats.csv"

            for iteration in range(iter):
                print(f"Iteration {iteration}", end="\r")
                out_data_path = measurement_data_dir / (f"{iteration}.csv" if keep_data else "out.csv")
                exec.run_benchmark(
                    alg,
                    left_path,
                    right_path,
                    out_data_path,
                    measurement_results_path,
                    measurement_output_stats_path,
                    iteration != 0,
                    validation_data_path,
                )

            print(f"Measured times: {str(measurement_results_path.absolute())}")
            if validate:
                print(f"Result data stats: {str(measurement_output_stats_path.absolute())}")
            print("-"*len(msg))

    if not keep_data:
        shutil.rmtree(data_dir)


def _run_benchmarks(args: argparse.Namespace):
    run_bechmarks(
        executable.Executable(args.executable_path),
        validator.Validator(args.validator_path),
        args.algorithms,
        args.sizes,
        args.iterations,
        args.validate,
        args.keep,
        not args.no_overwrite_check,
        args.output_path
    )


def list_algs(exec: executable.Executable):
    algs = exec.list_algorithms()
    print("\n".join(algs))


def _list_algs(args: argparse.Namespace):
    list_algs(executable.Executable(args.executable_path))


def benchmark_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("-o", "--output_path",
                        default=DEFAULT_OUTPUT_PATH,
                        type=Path,
                        help=f"Output directory path (defaults to {str(DEFAULT_OUTPUT_PATH)})")
    parser.add_argument("-a", "--algorithms",
                        nargs="+",
                        required=True,
                        type=str,
                        help="Algorithms to benchmark")
    parser.add_argument("-s", "--sizes",
                        nargs="+",
                        required=True,
                        type=input_size.InputSize.from_string,
                        help="Sizes of input to benchmark with in the format <rows>x<columns>x<left_matrices>x<right_matrices>")
    parser.add_argument("-i", "--iterations",
                        default=DEFAULT_ITERATIONS,
                        type=int,
                        help=f"Number of iterations for each configuration (defaults to {str(DEFAULT_ITERATIONS)})")
    parser.add_argument("-v", "--validate",
                        action="store_true",
                        help="Generate valid outputs using validator and validate that the benchmark outputs match")
    parser.add_argument("-k", "--keep",
                        action="store_true",
                        help="Keep inputs, outputs and possible validation outputs")
    parser.add_argument("--no_overwrite_check",
                        action="store_true",
                        help="Disable check which prevents overwrite of the output directory")
    parser.set_defaults(action=_run_benchmarks)


def list_algs_arguments(parser: argparse.ArgumentParser):
    parser.set_defaults(action=_list_algs)


def global_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("-e", "--executable_path",
                        default=executable.EXECUTABLE_PATH,
                        type=Path,
                        help=f"Path to the implementation executable (defaults to {str(executable.EXECUTABLE_PATH)})")
    parser.add_argument("-p", "--validator_path",
                        default=validator.VALIDATOR_PATH,
                        type=Path,
                        help=f"Path to the validator shell script (defaults to {str(validator.VALIDATOR_PATH)}")


def main():
    parser = argparse.ArgumentParser(description="Tool for simple benchmarking")
    global_arguments(parser)
    subparsers = parser.add_subparsers(required=True, dest="benchmarking",
                                             description="Generating and transforming input")
    benchmark_arguments(subparsers.add_parser("benchmark", help="Run benchmarks"))
    list_algs_arguments(subparsers.add_parser("list", help="List algorithms"))

    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
