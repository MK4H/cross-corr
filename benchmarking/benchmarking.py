import argparse

import validation
import benchmark
import input_generator

from pathlib import Path

from external import executable, validator


def main():
    parser = argparse.ArgumentParser(description="Tool for simple benchmarking")

    parser.add_argument("-e", "--executable_path",
                        default=executable.EXECUTABLE_PATH,
                        type=Path,
                        help=f"Path to the implementation executable (defaults to {str(executable.EXECUTABLE_PATH)})")
    parser.add_argument("-p", "--validator_path",
                        default=validator.VALIDATOR_PATH,
                        type=Path,
                        help=f"Path to the validator shell script (defaults to {str(validator.VALIDATOR_PATH)}")

    subparsers = parser.add_subparsers(required=True, dest="benchmarking",
                                             description="Tools for benchmarking")
    benchmark.benchmark_arguments(subparsers.add_parser("benchmark", help="Run benchmarks"))
    benchmark.list_algs_arguments(subparsers.add_parser("list", help="List algorithms"))
    input_generator.generator_arguments(subparsers.add_parser("generate", help="Generate input"))
    input_generator.transformer_arguments(subparsers.add_parser("transform", help="Transform input data arrays"))

    validation_parser = subparsers.add_parser("validation", help="Validate outputs and generate data for validation")
    validation.add_subparsers(validation_parser)

    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
