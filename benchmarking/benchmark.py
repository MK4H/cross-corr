import argparse
import re
import shutil
import sys
import json
from ruamel.yaml import YAML

import input_generator

from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

from external import input_size, validator, executable

DEFAULT_OUTPUT_PATH = Path.cwd() / "output"
DEFAULT_ITERATIONS = 100


class Run:
    def __init__(
            self,
            idx: int,
            name: str,
            algorithm: str,
            args: Dict[Any, Any],
    ):
        self.idx = idx
        self.name = name
        self.algorithm = algorithm
        self.algorithm_type = Run.get_algorithm_type(algorithm)
        self.args = args

    alg_type_regex = re.compile(".*_([^_]+_to_[^_]+)$")

    @classmethod
    def get_algorithm_type(cls, algorithm: str) -> str:
        match = cls.alg_type_regex.fullmatch(algorithm)
        if match:
            return match.group(1)
        raise ValueError(f"Invalid algorithm name {algorithm}")

    @classmethod
    def from_dict(cls, idx: int, data) -> "Run":
        return cls(
            idx,
            data["name"] if "name" in data else str(idx),
            data["algorithm"],
            data["args"] if "args" in data else {}
        )

    def create_args_file(self, path: Path):
        with path.open("w") as f:
            json.dump(self.args, f)

    def run(
            self,
            exe: executable.Executable,
            args_path: Path,
            left_input: Path,
            right_input: Path,
            iterations: int,
            result_times_path: Path,
            result_stats_path: Path,
            out_data_dir: Path,
            keep_outputs: bool,
            validation_data_path: Optional[Path]
    ):
        for iteration in range(iterations):
            print(f"Iteration {iteration}", end="\r")
            out_data_path = out_data_dir / (f"{iteration}.csv" if keep_outputs else "out.csv")
            exe.run_benchmark(
                self.algorithm,
                args_path,
                left_input,
                right_input,
                out_data_path,
                result_times_path,
                result_stats_path,
                iteration != 0,
                validation_data_path,
            )


class GlobalConfig:
    def __init__(
            self,
            output_path: Path,
            sizes: Optional[input_size.InputSize],
            iterations: Optional[int],
            validate: Optional[bool],
            keep: Optional[bool],
    ):
        self.output_path = output_path
        self.sizes = sizes
        self.iterations = iterations
        self.validate = validate
        self.keep = keep

    @property
    def data_path(self):
        return self.output_path / "data"

    @classmethod
    def from_dict(cls, data, output_path: Path) -> "GlobalConfig":
        if data is None:
            return cls(output_path, None, None, None, None)

        return cls(
            output_path,
            [input_size.InputSize.from_dict_or_string(in_size) for in_size in
             data["sizes"]] if "sizes" in data else None,
            int(data["iterations"]) if "iterations" in data else None,
            bool(data["validate"]) if "validate" in data else None,
            bool(data["keep"]) if "keep" in data else None
        )


class Group:
    def __init__(
            self,
            name: str,
            runs: List[Run],
            alg_type: str,
            sizes: List[input_size.InputSize],
            result_dir: Path,
            data_dir: Path,
            iterations: int,
            validate: bool,
            keep: bool
    ):
        self.name = name
        self.runs = runs
        self.alg_type = alg_type
        self.sizes = sizes
        self.result_dir = result_dir
        self.input_data_dir = data_dir / "inputs"
        self.output_data_dir = data_dir / "outputs"
        self.iterations = iterations
        self.validate = validate
        self.keep = keep

    @staticmethod
    def _config_from_dict(
            data,
            global_config: GlobalConfig
    ) -> Tuple[Optional[List[input_size.InputSize]], Optional[int], Optional[bool], Optional[bool]]:
        if data is None:
            return global_config.sizes, global_config.iterations, global_config.validate, global_config.keep
        sizes = [input_size.InputSize.from_dict_or_string(in_size) for in_size in data["sizes"]] if "sizes" in data else global_config.sizes
        iterations = int(data["iterations"]) if "iterations" in data else global_config.iterations
        validate = bool(data["validate"]) if "validate" in data else global_config.validate
        keep = bool(data["keep"]) if "keep" in data else global_config.keep

        return sizes, iterations, validate, keep

    @classmethod
    def from_dict(cls, data, global_config: GlobalConfig, index: int, exe: executable.Executable,):
        name = str(data['name']) if "name" in data else str(index)
        sizes, iterations, validate, keep = cls._config_from_dict(data.get("config", None), global_config)

        assert sizes is not None, "Missing list of sizes"
        assert iterations is not None, "Missing number of iterations"
        validate = validate if validate is not None else False
        keep = keep if keep is not None else False

        assert len(sizes) != 0, "No input sizes given"
        assert iterations > 0, f"Invalid number of iterations \"{iter}\" given"

        unique_name = f"{index}_{data['name']}" if "name" in data else str(index)
        group_dir = global_config.output_path / unique_name
        group_data_dir = global_config.data_path / unique_name

        try:
            runs = [Run.from_dict(run_idx, run) for run_idx, run in enumerate(data["runs"])]
        except ValueError as e:
            print(e)
            sys.exit(1)
        assert len(runs) != 0, "No runs given"

        alg_type = runs[0].algorithm_type
        for run in runs:
            if run.algorithm_type != alg_type:
                print(
                    f"All algorithms have to be of the same type, got {run.algorithm_type} and {alg_type} types",
                    file=sys.stderr,
                )
                sys.exit(1)

        for in_size in sizes:
            if not exe.validate_input_size(alg_type, in_size):
                print(
                    f"Input size {in_size} cannot be used with algorithms of type {alg_type}",
                    file=sys.stderr,
                )
                sys.exit(1)

        return cls(name, runs, alg_type, sizes, group_dir, group_data_dir, iterations, validate, keep)

    def generate_inputs(
            self,
            index: int,
            size: input_size.InputSize,
    ) -> Tuple[Path, Path]:
        left_path = self.input_data_dir / (
            f"{index}_left_{size.rows}_{size.columns}_{size.left_matrices}.csv" if self.keep else "left.csv")
        right_path = self.input_data_dir / (
            f"{index}_right_{size.rows}_{size.columns}_{size.right_matrices}.csv" if self.keep else "right.csv")

        input_generator.generate_matrices(size.left_matrices, size.rows, size.columns,
                                          input_generator.OutputFormats.CSV, left_path)
        input_generator.generate_matrices(size.right_matrices, size.rows, size.columns,
                                          input_generator.OutputFormats.CSV, right_path)

        return left_path, right_path

    def cleanup(self):
        shutil.rmtree(self.input_data_dir, ignore_errors=True)
        shutil.rmtree(self.output_data_dir, ignore_errors=True)

    def log_step(self, step: int, num_steps: int, message: str) -> int:
        print(f"[{self.name}][{step}/{num_steps}] {message}")
        return step + 1


    def run(
            self,
            exe: executable.Executable,
            valid: validator.Validator,
            prevent_override: bool
    ):
        self.result_dir.mkdir(exist_ok=not prevent_override, parents=True)
        self.cleanup()

        self.input_data_dir.mkdir(parents=True)
        self.output_data_dir.mkdir(parents=True)

        for run in self.runs:
            args_path = self.input_data_dir / f"{run.idx}_{run.name}_args.json"
            run.create_args_file(args_path)

        num_steps = len(self.sizes) * len(self.runs) + len(self.sizes) + (len(self.sizes) if self.validate else 0)
        step = 1

        for input_idx, in_size in enumerate(self.sizes):
            step = self.log_step(step, num_steps, f"Generating inputs of size {in_size}")
            left_path, right_path = self.generate_inputs(input_idx, in_size)

            if self.validate:
                step = self.log_step(step, num_steps, f"Generating validation data")
                validation_data_path = self.input_data_dir / f"{input_idx}_valid_{in_size.rows}_{in_size.columns}_{in_size.left_matrices}_{in_size.right_matrices}.csv"
                valid.generate_validation_data(
                    self.alg_type,
                    left_path,
                    right_path,
                    validation_data_path
                )
            else:
                validation_data_path = None

            for run in self.runs:
                step = self.log_step(step, num_steps, f"Benchmarking {run.name} for {in_size}")

                args_path = self.input_data_dir / f"{run.idx}_{run.name}_args.json"

                measurement_suffix = f"{input_idx}_{run.idx}_{run.name}_{in_size}"
                out_data_dir = self.output_data_dir / f"{measurement_suffix}"
                out_data_dir.mkdir(parents=True)

                measurement_results_path = self.result_dir / f"{measurement_suffix}_time.csv"
                measurement_output_stats_path = self.result_dir / f"{measurement_suffix}_output_stats.csv"

                run.run(
                    exe,
                    args_path,
                    left_path,
                    right_path,
                    self.iterations,
                    measurement_results_path,
                    measurement_output_stats_path,
                    out_data_dir,
                    self.keep,
                    validation_data_path
                )

                last_msg = f"Measured times: {str(measurement_results_path.absolute())}"
                print(last_msg)
                if self.validate:
                    last_msg = f"Result data stats: {str(measurement_output_stats_path.absolute())}"
                    print(last_msg)
                print("-" * len(last_msg))
        if not self.keep:
            self.cleanup()


def parse_benchmark_config(path: Path):
    yaml = YAML(typ='safe', pure=True)
    return yaml.load(path)


def run_bechmarks(
        exe: executable.Executable,
        valid: validator.Validator,
        benchmark_def_file: Path,
        prevent_overwrite: bool,
        out_dir_path: Optional[Path]
):

    definition = parse_benchmark_config(benchmark_def_file)
    benchmark = definition["benchmark"]
    name = benchmark["name"]
    if out_dir_path is None:
        out_dir_path = benchmark_def_file.parent / name
    elif not out_dir_path.is_absolute():
        out_dir_path = benchmark_def_file.parent / out_dir_path

    global_config = GlobalConfig.from_dict(benchmark.get("config", None), out_dir_path)
    groups = [Group.from_dict(group_data, global_config, group_idx, exe) for group_idx, group_data in enumerate(benchmark["groups"])]
    for group_idx, group in enumerate(groups):
        print(f"-- [{group_idx + 1}/{len(groups)}] Running group {group.name} --")
        group.run(
            exe,
            valid,
            prevent_overwrite
        )


def _run_benchmarks(args: argparse.Namespace):
    run_bechmarks(
        executable.Executable(args.executable_path),
        validator.Validator(args.validator_path),
        args.benchmark_definition_path,
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
                        type=Path,
                        help=f"Output directory path (defaults to the name of the benchmark)")
    parser.add_argument("--no_overwrite_check",
                        action="store_true",
                        help="Disable check which prevents overwrite of the output directory")
    parser.add_argument("benchmark_definition_path",
                        type=Path,
                        help="Path to the benchmark definition YAML file")
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
