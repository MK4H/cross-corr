import argparse
import itertools
import re
import shutil
import sys
import json


import input_generator

from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from abc import ABC, abstractmethod

from ruamel.yaml import YAML

from external import input_size, validator, executable, benchmark_script

DEFAULT_OUTPUT_PATH = Path.cwd() / "output"
DEFAULT_ITERATIONS = 100


class Run(ABC):
    def __init__(
        self,
        idx: int,
        name: str,
        algorithm_type: str
    ):
        self.idx = idx
        self.name = name
        self.algorithm_type = algorithm_type

    def prepare(self):
        """
        Prepare things which are independent of the input size
        :return:
        """
        pass



    @abstractmethod
    def run(self,
            left_input: Path,
            right_input: Path,
            data_type: str,
            iterations: int,
            result_times_path: Path,
            result_stats_path: Path,
            out_data_dir: Path,
            keep_outputs: bool,
            validation_data_path: Optional[Path],
            verbose: bool
            ):
        pass

    @classmethod
    def from_dict(cls, idx: int, exe: executable.Executable, base_dir_path: Path, input_data_dir: Path, data) -> List["Run"]:
        factory_methods = {
            "internal": InternalRun.from_dict,
            "external": ExternalRun.from_dict
        }
        if "type" not in data:
            return factory_methods["internal"](idx, exe, base_dir_path, input_data_dir, data)
        elif data["type"] in factory_methods:
            return factory_methods[data["type"]](idx, exe, base_dir_path, input_data_dir, data)
        else:
            raise ValueError(f"Unknown run type {data['type']}")


class InternalRun(Run):
    def __init__(
            self,
            idx: int,
            name: str,
            exe: executable.Executable,
            algorithm: str,
            args: Dict[Any, Any],
            args_file_path: Path,
    ):
        super().__init__(idx, name, InternalRun.get_algorithm_type(algorithm))
        self.exe = exe
        self.algorithm = algorithm
        self.args = args
        self.args_file_path = args_file_path

    alg_type_regex = re.compile(".*_([^_]+_to_[^_]+)$")

    @classmethod
    def get_algorithm_type(cls, algorithm: str) -> str:
        match = cls.alg_type_regex.fullmatch(algorithm)
        if match:
            return match.group(1)
        raise ValueError(f"Invalid algorithm name {algorithm}")

    @classmethod
    def from_dict(cls, idx: int, exe: executable.Executable, base_dir_path: Path, input_data_dir: Path, data) -> List["Run"]:
        algorithm = data["algorithm"]
        base_name = data["name"] if "name" in data else f"{idx}_{algorithm}"
        args = data["args"] if "args" in data else {}

        generate = {}
        singles = {}
        for key, value in args.items():
            if type(value) is list and len(value) > 1:
                generate[key] = value
            elif type(value) is list and len(value) == 1:
                singles[key] = value[0]
            else:
                singles[key] = value

        if len(generate) == 0:
            return [cls(
                idx,
                f"{base_name}____",
                exe,
                algorithm,
                singles,
                input_data_dir / f"{idx}-{base_name}-args.json"
            )]

        # Generate all combinations of values from each generate key
        keys, values = zip(*generate.items())
        combinations = itertools.product(*values)
        runs = []
        for combination in combinations:
            name_suffix = "_".join(str(val) for val in combination)
            run_args = {**singles, **dict(zip(keys, combination))}
            name = f"{base_name}__{name_suffix}__"
            runs.append(cls(
                idx,
                name,
                exe,
                algorithm,
                run_args,
                input_data_dir / f"{idx}-{name}-args.json",
            ))
        return runs

    def prepare(self):
        with self.args_file_path.open("w") as f:
            json.dump(self.args, f)

    def run(
            self,
            left_input: Path,
            right_input: Path,
            data_type: str,
            iterations: int,
            result_times_path: Path,
            result_stats_path: Path,
            out_data_dir: Path,
            keep_outputs: bool,
            validation_data_path: Optional[Path],
            verbose: bool
    ):
        for iteration in range(iterations):
            print(f"Iteration {iteration + 1}/{iterations}", end="\r")
            out_data_path = out_data_dir / (f"{iteration}.csv" if keep_outputs else "out.csv")
            self.exe.run_benchmark(
                self.algorithm,
                data_type,
                self.args_file_path,
                left_input,
                right_input,
                out_data_path,
                result_times_path,
                result_stats_path,
                iteration != 0,
                validation_data_path,
                verbose
            )


class ExternalRun(Run):
    def __init__(
            self,
            idx: int,
            name: str,
            alg_type: str,
            exe: executable.Executable,
            script_path: Path,
    ):
        super().__init__(idx, name, alg_type)
        self.exe = exe
        self.script = benchmark_script.BenchmarkScript(script_path)

    @classmethod
    def from_dict(cls, idx: int, exe: executable.Executable, base_dir_path: Path, input_data_dir: Path, data) -> List["ExternalRun"]:
        alg_type = data["alg_type"]
        name = data["name"] if "name" in data else f"{idx}_{alg_type}"
        script_path = Path(data["path"])

        script_path = script_path if script_path.is_absolute() else base_dir_path / script_path
        return [
            cls(idx,
                name,
                alg_type,
                exe,
                script_path,
            )
        ]

    def run(self,
            left_input: Path,
            right_input: Path,
            data_type: str,
            iterations: int,
            result_times_path: Path,
            result_stats_path: Path,
            out_data_dir: Path,
            keep_outputs: bool,
            validation_data_path: Optional[Path],
            verbose: bool
            ):
        self.script.run_benchmark(
            self.algorithm_type,
            data_type,
            iterations,
            left_input,
            right_input,
            out_data_dir,
            result_times_path,
            verbose
        )

        if validation_data_path is not None:
            output_paths = [file for file in out_data_dir.glob("*") if file.is_file()]
            validation_csv = self.exe.validate_data(validation_data_path, output_paths, csv=True, normalize=False)
            result_stats_path.write_text(validation_csv)


class GlobalConfig:
    def __init__(
            self,
            base_dir_path: Path,
            output_path: Path,
            sizes: Optional[input_size.InputSize],
            data_type: Optional[str],
            iterations: Optional[int],
            validate: Optional[bool],
            keep: Optional[bool],
    ):
        """

        :param base_dir_path: Path to the directory containing the benchmark definition
        :param output_path: Path to a writable directory to be created and used for benchmarks
        :param sizes: The default set of input sizes to be used
        :param data_type: Default data type to be used
        :param iterations: Default number of iterations of each benchmark
        :param validate: Default flag if the outputs should be validated
        :param keep: Default flag if outputs of each iterations should be kept
        """
        self.base_dir_path = base_dir_path
        self.output_path = output_path
        self.sizes = sizes
        self.data_type = data_type
        self.iterations = iterations
        self.validate = validate
        self.keep = keep

    @property
    def data_path(self):
        return self.output_path / "data"

    @classmethod
    def from_dict(cls, data, base_dir_path: Path, output_path: Path) -> "GlobalConfig":
        if data is None:
            return cls(base_dir_path, output_path, None, None, None, None, None)

        return cls(
            base_dir_path,
            output_path,
            [input_size.InputSize.from_dict_or_string(in_size) for in_size in
             data["sizes"]] if "sizes" in data else None,
            data["data_type"] if "data_type" in data else None,
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
            data_type: str,
            result_dir: Path,
            input_data_dir: Path,
            output_data_dir: Path,
            iterations: int,
            validate: bool,
            keep: bool
    ):
        self.name = name
        self.runs = runs
        self.alg_type = alg_type
        self.sizes = sizes
        self.data_type = data_type
        self.result_dir = result_dir
        self.input_data_dir = input_data_dir
        self.output_data_dir = output_data_dir
        self.iterations = iterations
        self.validate = validate
        self.keep = keep

    @staticmethod
    def _config_from_dict(
            data,
            global_config: GlobalConfig
    ) -> Tuple[Optional[List[input_size.InputSize]], Optional[str], Optional[int], Optional[bool], Optional[bool]]:
        if data is None:
            return global_config.sizes, global_config.data_type, global_config.iterations, global_config.validate, global_config.keep
        sizes = [input_size.InputSize.from_dict_or_string(in_size) for in_size in data["sizes"]] if "sizes" in data else global_config.sizes
        data_type = data["data_type"] if "data_type" in data else "single"
        iterations = int(data["iterations"]) if "iterations" in data else global_config.iterations
        validate = bool(data["validate"]) if "validate" in data else global_config.validate
        keep = bool(data["keep"]) if "keep" in data else global_config.keep

        return sizes, data_type, iterations, validate, keep

    @classmethod
    def from_dict(cls, data, global_config: GlobalConfig, index: int, exe: executable.Executable,):
        name = str(data['name']) if "name" in data else str(index)
        sizes, data_type, iterations, validate, keep = cls._config_from_dict(data.get("config", None), global_config)

        assert sizes is not None, "Missing list of sizes"
        assert data_type is not None, "Missing data type"
        assert iterations is not None, "Missing number of iterations"
        validate = validate if validate is not None else False
        keep = keep if keep is not None else False

        assert len(sizes) != 0, "No input sizes given"
        assert iterations > 0, f"Invalid number of iterations \"{iter}\" given"

        unique_name = f"{index}_{data['name']}" if "name" in data else str(index)
        group_dir = global_config.output_path / unique_name
        group_data_dir = global_config.data_path / unique_name

        input_data_dir = group_data_dir / "inputs"
        output_data_dir = group_data_dir / "outputs"
        try:
            # Load warps of runs and flatten them into a list of runs
            runs = list(itertools.chain.from_iterable([Run.from_dict(run_idx, exe, global_config.base_dir_path, input_data_dir, run) for run_idx, run in enumerate(data["runs"])]))
        except ValueError as e:
            print(f"Failed to load runs: {e}", file=sys.stderr)
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

        return cls(name, runs, alg_type, sizes, data_type, group_dir, input_data_dir, output_data_dir, iterations, validate, keep)

    def generate_inputs(
            self,
            index: int,
            size: input_size.InputSize,
    ) -> Tuple[Path, Path]:
        left_path = self.input_data_dir / (
            f"{index}-left-{size}.csv" if self.keep else "left.csv")
        right_path = self.input_data_dir / (
            f"{index}-right-{size}.csv" if self.keep else "right.csv")

        input_generator.generate_matrices(size.left_matrices, size.rows, size.columns,
                                          input_generator.OutputFormats.CSV, left_path)
        input_generator.generate_matrices(size.right_matrices, size.rows, size.columns,
                                          input_generator.OutputFormats.CSV, right_path)

        return left_path, right_path

    def cleanup(self):
        shutil.rmtree(self.input_data_dir, ignore_errors=True)
        shutil.rmtree(self.output_data_dir, ignore_errors=True)

    def log_step(self, step: int, num_steps: int, message: str) -> int:
        print(f"[{step}/{num_steps}] {message}")
        return step + 1

    def run(
            self,
            valid: validator.Validator,
            prevent_override: bool,
            verbose: bool
    ):
        self.result_dir.mkdir(exist_ok=not prevent_override, parents=True)
        self.cleanup()

        self.input_data_dir.mkdir(parents=True)
        self.output_data_dir.mkdir(parents=True)

        for run in self.runs:
            run.prepare()

        num_steps = len(self.sizes) * len(self.runs) + len(self.sizes) + (len(self.sizes) if self.validate else 0)
        step = 1

        for input_idx, in_size in enumerate(self.sizes):
            step = self.log_step(step, num_steps, f"Generating inputs of size {in_size}")
            left_path, right_path = self.generate_inputs(input_idx, in_size)

            if self.validate:
                step = self.log_step(step, num_steps, f"Generating validation data")
                validation_data_path = self.input_data_dir / f"{input_idx}-valid-{in_size}.csv"
                valid.generate_validation_data(
                    self.alg_type,
                    self.data_type,
                    left_path,
                    right_path,
                    validation_data_path
                )
            else:
                validation_data_path = None

            for run in self.runs:
                step = self.log_step(step, num_steps, f"Benchmarking {run.name} for {in_size}")

                measurement_suffix = f"{run.idx}-{input_idx}-{run.name}-{in_size}"
                out_data_dir = self.output_data_dir / f"{measurement_suffix}"
                out_data_dir.mkdir(parents=True)

                measurement_results_path = self.result_dir / f"{measurement_suffix}-time.csv"
                measurement_output_stats_path = self.result_dir / f"{measurement_suffix}-output_stats.csv"

                run.run(
                    left_path,
                    right_path,
                    self.data_type,
                    self.iterations,
                    measurement_results_path,
                    measurement_output_stats_path,
                    out_data_dir,
                    self.keep,
                    validation_data_path,
                    verbose
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
        group_filter: List[str],
        prevent_overwrite: bool,
        out_dir_path: Optional[Path],
        verbose: bool
):

    definition = parse_benchmark_config(benchmark_def_file)
    benchmark = definition["benchmark"]
    name = benchmark["name"]

    base_dir_path = benchmark_def_file.parent

    if out_dir_path is None:
        out_dir_path = base_dir_path / name
    elif not out_dir_path.is_absolute():
        out_dir_path = base_dir_path / out_dir_path

    global_config = GlobalConfig.from_dict(benchmark.get("config", None), base_dir_path, out_dir_path)
    groups = [Group.from_dict(group_data, global_config, group_idx, exe) for group_idx, group_data in enumerate(benchmark["groups"])]

    if len(group_filter) != 0:
        groups = [group for group in groups if group.name in group_filter]

    for group_idx, group in enumerate(groups):
        print(f"-- [{group_idx + 1}/{len(groups)}] Running group {group.name} --")
        group.run(
            valid,
            prevent_overwrite,
            verbose
        )


def _run_benchmarks(args: argparse.Namespace):
    run_bechmarks(
        executable.Executable(args.executable_path),
        validator.Validator(args.validator_path),
        args.benchmark_definition_path,
        args.groups,
        not args.no_overwrite_check,
        args.output_path,
        args.verbose
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
    parser.add_argument("-v", "--verbose",
                       action="store_true",
                       help="Increase verbosity of the commandline output")
    parser.add_argument("benchmark_definition_path",
                        type=Path,
                        help="Path to the benchmark definition YAML file")
    parser.add_argument("groups",
                        type=str,
                        nargs="*",
                        help="Groups to run, all by default"
    )
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
