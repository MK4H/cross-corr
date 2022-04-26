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
from enum import Enum

from ruamel.yaml import YAML

from external import input_size, validator, executable, benchmark_script, execution_error

DEFAULT_OUTPUT_PATH = Path.cwd() / "output"
DEFAULT_ITERATIONS = 100

yaml = YAML(typ='safe', pure=True)


class ExistingResultsPolicy(Enum):
    FAIL = 0
    DELETE = 1
    CONTINUE = 2


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
            benchmark_type: str,
            outer_iterations: int,
            inner_iterations: int,
            min_measure_seconds: float,
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

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass


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
        base_name = data.get("name", f"{idx}_{algorithm}")
        args = data.get("args", {})

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idx": self.idx,
            "name": self.name,
            "alg_type": self.algorithm_type,
            "executable": str(self.exe.executable_path),
            "algorithm": self.algorithm,
            "args": self.args
        }

    def prepare(self):
        with self.args_file_path.open("w") as f:
            json.dump(self.args, f)

    def run(
            self,
            left_input: Path,
            right_input: Path,
            data_type: str,
            benchmark_type: str,
            outer_iterations: int,
            inner_iterations: int,
            min_measure_seconds: float,
            result_times_path: Path,
            result_stats_path: Path,
            out_data_dir: Path,
            keep_output: bool,
            validation_data_path: Optional[Path],
            verbose: bool
    ):
        if keep_output:
            out_data_dir.mkdir(exist_ok=True,
                               parents=True)
        for iteration in range(outer_iterations):
            print(f"Iteration {iteration + 1}/{outer_iterations}", end="\r")
            out_data_path = (out_data_dir / f"{iteration}.csv") if keep_output else None
            self.exe.run_benchmark(
                self.algorithm,
                data_type,
                benchmark_type,
                inner_iterations,
                min_measure_seconds,
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
    def from_dict(
            cls,
            idx: int,
            exe: executable.Executable,
            base_dir_path: Path,
            input_data_dir: Path,
            data
    ) -> List["ExternalRun"]:
        alg_type = data["alg_type"]
        base_name = data.get("name", f"{idx}_{alg_type}")
        # Underscores to match the naming scheme of Internal runs
        #   and in the future to possibly hold used arguments as in Internal runs
        name = f"{base_name}____"
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idx": self.idx,
            "name": self.name,
            "alg_type": self.algorithm_type,
            "script": str(self.script.script_path),
        }

    def run(self,
            left_input: Path,
            right_input: Path,
            data_type: str,
            benchmark_type: str,
            outer_iterations: int,
            inner_iterations: int,
            min_measure_seconds: float,
            result_times_path: Path,
            result_stats_path: Path,
            out_data_dir: Path,
            keep_output: bool,
            validation_data_path: Optional[Path],
            verbose: bool
            ):

        write_output = keep_output or validation_data_path is not None

        if write_output:
            out_data_dir.mkdir(exist_ok=True,
                               parents=True)

        for iteration in range(outer_iterations):
            print(f"Iteration {iteration + 1}/{outer_iterations}", end="\r")
            out_data_path = (out_data_dir / f"{iteration}.csv") if keep_output else (out_data_dir / "out.csv")
            out_data_path = out_data_path if write_output else None
            self.script.run_benchmark(
                self.algorithm_type,
                data_type,
                inner_iterations,
                min_measure_seconds,
                left_input,
                right_input,
                out_data_path,
                result_times_path,
                verbose
            )

            if validation_data_path is not None:
                glob = f"{out_data_path.stem}*{out_data_path.suffix}"
                output_paths = [file for file in out_data_dir.glob(glob) if file.is_file()]
                validation_csv = self.exe.validate_data(
                    validation_data_path,
                    output_paths,
                    csv=True,
                    normalize=False,
                    print_header=iteration == 0
                )
                with result_stats_path.open("a") as f:
                    f.write(validation_csv)

        if write_output and not keep_output:
            shutil.rmtree(out_data_dir)


class GlobalConfig:
    def __init__(
            self,
            base_dir_path: Path,
            output_path: Path,
            sizes: Optional[input_size.InputSize],
            data_type: str,
            benchmark_type: str,
            outer_iterations: int,
            inner_iterations: int,
            min_measure_seconds: float,
            validate: bool,
            keep_output: bool,
    ):
        """

        :param base_dir_path: Path to the directory containing the benchmark definition
        :param output_path: Path to a writable directory to be created and used for benchmarks
        :param sizes: The default set of input sizes to be used
        :param data_type: Default data type to be used
        :param outer_iterations: Default number of outer iterations of each benchmark, where the program is started again for each iteration
        :param inner_iterations: Default number of inner iterations which are implemented by the program itself
        :param min_measure_seconds: Minimum number of seconds to measure before the result is taken as statistically relevant
        :param validate: Default flag if the outputs should be validated
        :param keep_output: Default flag if outputs of each iterations should be kept
        """
        self.base_dir_path = base_dir_path
        self.output_path = output_path
        self.sizes = sizes
        self.data_type = data_type
        self.benchmark_type = benchmark_type
        self.outer_iterations = outer_iterations
        self.inner_iterations = inner_iterations
        self.min_measure_seconds = min_measure_seconds
        self.validate = validate
        self.keep_output = keep_output

    @property
    def data_path(self):
        return self.output_path / "data"

    @classmethod
    def from_dict(cls, data, base_dir_path: Path, output_path: Path) -> "GlobalConfig":

        data = data if data is not None else {}

        return cls(
            base_dir_path,
            output_path,
            [input_size.InputSize.from_dict_or_string(in_size) for in_size in
             data["sizes"]] if "sizes" in data else None,
            data.get("data_type", "single"),
            data.get("benchmark_type", "Compute"),
            int(data.get("outer_iterations", 1)),
            int(data.get("inner_iterations", 1)),
            float(data.get("min_measure_seconds", 1.0)),
            bool(data.get("validate", False)),
            bool(data.get("keep_output", False))
        )


class Logger:
    def __init__(self, num_steps: int, verbose: bool, failure_log_path: Path):
        self.num_steps = num_steps
        self.step = 1
        self.verbose = verbose
        self.failure_log_path = failure_log_path
        self.last_msg = ""

    def __enter__(self):
        self.failure_log = self.failure_log_path.open("a")
        return self

    def __exit__(self, type, value, traceback):
        self.failure_log.close()

    def log_step(self, message: str, *, skip=False) -> None:
        skip_message = " SKIP:" if skip else ""
        print(f"[{self.step}/{self.num_steps}]{skip_message} {message}")
        self.step += 1

    def log_failure(
            self,
            run: Run,
            error: execution_error.ExecutionError,
            input_idx: int,
            in_size: input_size.InputSize,
            left_path: Path,
            right_path: Path
    ):
        # Append it as a single entry list, so that together all appends create one big top level list
        data = [{
            "run": run.to_dict(),
            "input": {
                "idx": input_idx,
                "size": in_size.to_dict(),
                "left": str(left_path),
                "right": str(right_path),
            },
            "error": error.to_dict(),
        }]
        yaml.dump(data, self.failure_log)

    def log(self, message):
        print(message)
        self.last_msg = message

    def underline_last_message(self):
        print("-"*len(self.last_msg))


class InputSizeSubgroup:
    def __init__(
            self,
            group: "Group",
            group_execution: "GroupExecution",
            logger: Logger,
            in_idx: int,
            in_size: input_size.InputSize,
            left_path: Path,
            right_path: Path,
            validation_data_path: Optional[Path],
    ):
        self.group = group
        self.group_execution = group_execution
        self.logger = logger
        self.index = in_idx
        self.input_size = in_size
        self.left_path = left_path
        self.right_path = right_path
        self.validation_data_path = validation_data_path

    @classmethod
    def generate_inputs(
            cls,
            group: "Group",
            group_execution: "GroupExecution",
            logger: Logger,
            in_idx: int,
            in_size: input_size.InputSize,
            valid: Optional[validator.Validator]
    ) -> "InputSizeSubgroup":
        left_path = group.input_data_dir / f"{in_idx}-left-{in_size}.csv"
        right_path = group.input_data_dir / f"{in_idx}-right-{in_size}.csv"
        validation_data_path = group.input_data_dir / f"{in_idx}-valid-{in_size}.csv"
        if valid is None:
            validation_data_path = None

        generated_inputs = False
        if (group_execution.existing_results == ExistingResultsPolicy.CONTINUE and
                left_path.exists() and
                right_path.exists()):
            logger.log_step(f"Generating inputs of size {in_size}", skip=True)
        else:
            logger.log_step(f"Generating inputs of size {in_size}")
            input_generator.generate_matrices(in_size.left_matrices, in_size.rows, in_size.columns,
                                              input_generator.OutputFormats.CSV, left_path)
            input_generator.generate_matrices(in_size.right_matrices, in_size.rows, in_size.columns,
                                              input_generator.OutputFormats.CSV, right_path)
            generated_inputs = True

        if valid is not None and (generated_inputs or not validation_data_path.exists()):
            logger.log_step(f"Generating validation data")
            tmp_validation_data_path = group.input_data_dir / f"{in_idx}-valid-{in_size}.tmp.csv"
            valid.generate_validation_data(
                group.alg_type,
                group.data_type,
                left_path,
                right_path,
                tmp_validation_data_path
            )
            # To make sure the validation data is not corrupted by cancellation or something
            # This should ensure that the whole uncorrupted validation data can only exist
            tmp_validation_data_path.rename(validation_data_path)
        else:
            logger.log_step(f"Generating validation data", skip=True)

        return cls(group, group_execution, logger, in_idx, in_size, left_path, right_path, validation_data_path)

    def get_run_result_paths(self, run: Run) -> Tuple[Path, Path, Path]:
        measurement_suffix = f"{run.idx}-{self.index}-{run.name}-{self.input_size}"
        out_data_dir = self.group.output_data_dir / f"{measurement_suffix}"

        measurement_results_path = self.group.result_dir / f"{measurement_suffix}-time.csv"
        measurement_output_stats_path = self.group.result_dir / f"{measurement_suffix}-output_stats.csv"

        return out_data_dir, measurement_results_path, measurement_output_stats_path

    def execute_runs(self):
        for run in self.group.runs:
            out_data_dir, measurement_results_path, measurement_output_stats_path = \
                self.get_run_result_paths(run)

            measure = self.group.benchmark_type.lower() != "none"
            validate = self.validation_data_path is not None

            skip = ((measurement_results_path.exists() or not measure) and
                    (measurement_output_stats_path.exists() or not validate))

            self.logger.log_step(f"Benchmarking {run.name} for {self.input_size}", skip=skip)

            try:
                if not skip:
                    run.run(
                        self.left_path,
                        self.right_path,
                        self.group.data_type,
                        self.group.benchmark_type,
                        self.group.outer_iterations,
                        self.group.inner_iterations,
                        self.group.min_measure_seconds,
                        self.group_execution.tmp_results_path,
                        self.group_execution.tmp_output_stats_path,
                        out_data_dir,
                        self.group.keep_output,
                        self.validation_data_path,
                        self.logger.verbose
                    )

                    if measure:
                        self.group_execution.tmp_results_path.rename(measurement_results_path)
                    if validate:
                        self.group_execution.tmp_output_stats_path.rename(measurement_output_stats_path)
                if measure:
                    self.logger.log(f"Measured times: {str(measurement_results_path.absolute())}")
                if validate:
                    self.logger.log(f"Result data stats: {str(measurement_output_stats_path.absolute())}")
            except execution_error.ExecutionError as e:
                self.logger.log_failure(run, e, self.index, self.input_size, self.left_path, self.right_path)
                self.logger.log(f"Run failed, see {str(self.logger.failure_log_path.absolute())}")
            self.logger.underline_last_message()


class GroupExecution:
    def __init__(self, group: "Group", existing_results: ExistingResultsPolicy):
        self.group = group
        self.step = 1
        self.tmp_results_path = group.result_dir / "time.tmp.csv"
        self.tmp_output_stats_path = group.result_dir / "output_stats.tmp.csv"
        self.failure_log_path = group.result_dir / "failed_runs.yml"
        self.existing_results = existing_results

    def clean_inputs(self):
        shutil.rmtree(self.group.input_data_dir, ignore_errors=True)

    def clean_outputs(self):
        shutil.rmtree(self.group.output_data_dir, ignore_errors=True)

    def prepare_directories(self):
        delete = self.existing_results == ExistingResultsPolicy.DELETE
        cont = self.existing_results == ExistingResultsPolicy.CONTINUE
        if delete:
            shutil.rmtree(self.group.result_dir, ignore_errors=True)
        self.group.result_dir.mkdir(exist_ok=cont, parents=True)

        # If FAIL, the results have been cleared so we can just delete data anyway
        # as they are useless without results
        if not cont:
            self.clean_inputs()
            self.clean_outputs()

        self.group.input_data_dir.mkdir(exist_ok=cont, parents=True)
        self.group.output_data_dir.mkdir(exist_ok=cont, parents=True)

        self.tmp_results_path.unlink(missing_ok=True)
        self.tmp_output_stats_path.unlink(missing_ok=True)

    def run(self, verbose: bool, valid: Optional[validator.Validator]):
        # + 2* once for generating inputs, once for generating validation data, even if skipped
        num_steps = len(self.group.sizes) * len(self.group.runs) + 2 * len(self.group.sizes)
        with Logger(num_steps, verbose, self.failure_log_path) as logger:
            for input_idx, in_size in enumerate(self.group.sizes):
                subgroup = InputSizeSubgroup.generate_inputs(
                    self.group,
                    self,
                    logger,
                    input_idx,
                    in_size,
                    valid
                )

                subgroup.execute_runs()


class Group:
    def __init__(
            self,
            name: str,
            runs: List[Run],
            alg_type: str,
            sizes: List[input_size.InputSize],
            data_type: str,
            benchmark_type: str,
            result_dir: Path,
            input_data_dir: Path,
            output_data_dir: Path,
            outer_iterations: int,
            inner_iterations: int,
            min_measure_seconds: float,
            validate: bool,
            keep_output: bool
    ):
        self.name = name
        self.runs = runs
        self.alg_type = alg_type
        self.sizes = sizes
        self.data_type = data_type
        self.benchmark_type = benchmark_type
        self.result_dir = result_dir
        self.input_data_dir = input_data_dir
        self.output_data_dir = output_data_dir
        self.outer_iterations = outer_iterations
        self.inner_iterations = inner_iterations
        self.min_measure_seconds = min_measure_seconds
        self.validate = validate
        self.keep_output = keep_output

    @staticmethod
    def _config_from_dict(
            data,
            global_config: GlobalConfig
    ) -> Tuple[Optional[List[input_size.InputSize]], str, str, int, int, float, bool, bool]:

        data = data if data is not None else {}

        sizes = [input_size.InputSize.from_dict_or_string(in_size) for in_size in data["sizes"]] if "sizes" in data else global_config.sizes
        data_type = data.get("data_type", global_config.data_type)
        benchmark_type = data.get("benchmark_type", global_config.benchmark_type)
        outer_iterations = int(data.get("outer_iterations", global_config.outer_iterations))
        inner_iterations = int(data.get("inner_iterations", global_config.inner_iterations))
        min_measure_seconds = float(data.get("min_measure_seconds", global_config.min_measure_seconds))
        validate = bool(data.get("validate", global_config.validate))
        keep = bool(data.get("keep_output", global_config.keep_output))

        return sizes, data_type, benchmark_type, outer_iterations, inner_iterations, min_measure_seconds, validate, keep

    @classmethod
    def from_dict(cls, data, global_config: GlobalConfig, index: int, exe: executable.Executable):
        name = str(data.get("name", index))
        sizes, data_type, benchmark_type, outer_iterations, inner_iterations, min_measure_seconds, validate, keep_output = cls._config_from_dict(
            data.get("config", None),
            global_config
        )

        assert sizes is not None, "Missing list of sizes"

        assert len(sizes) != 0, "No input sizes given"
        assert outer_iterations > 0, f"Invalid number of outer iterations \"{outer_iterations}\" given"
        assert inner_iterations > 0, f"Invalid number of inner iterations \"{inner_iterations}\" given"

        unique_name = f"{index}_{data['name']}" if "name" in data else str(index)
        group_dir = global_config.output_path / unique_name
        group_data_dir = global_config.data_path / unique_name

        input_data_dir = group_data_dir / "inputs"
        output_data_dir = group_data_dir / "outputs"
        try:
            # Load warps of runs and flatten them into a list of runs
            runs = list(itertools.chain.from_iterable(
                [Run.from_dict(run_idx, exe, global_config.base_dir_path, input_data_dir, run)
                 for run_idx, run in enumerate(data["runs"])])
            )
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

        return cls(
            name,
            runs,
            alg_type,
            sizes,
            data_type,
            benchmark_type,
            group_dir,
            input_data_dir,
            output_data_dir,
            outer_iterations,
            inner_iterations,
            min_measure_seconds,
            validate,
            keep_output,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "config": {
                "data_type": self.data_type,
                "benchmark_type": self.benchmark_type,
                "outer_iterations": self.outer_iterations,
                "inner_iterations": self.inner_iterations,
                "sizes": [size.to_dict() for size in self.sizes],
                "validate": self.validate,
                "keep_output": self.keep_output,
            },
            "runs": [run.to_dict() for run in self.runs]
        }

    def run(
            self,
            valid: validator.Validator,
            existing_results: ExistingResultsPolicy,
            verbose: bool
    ):
        execution = GroupExecution(self, existing_results)

        execution.prepare_directories()

        with (self.result_dir / "group_definition.yml").open("w") as definition:
            yaml.dump(self.to_dict(), definition)

        for run in self.runs:
            run.prepare()

        execution.run(verbose, valid if self.validate else None)
        if not self.keep_output:
            execution.clean_outputs()


def parse_benchmark_config(path: Path):
    return yaml.load(path)


def parse_benchmark(
        exe: executable.Executable,
        benchmark_def_file: Path,
        group_filter: List[str],
        out_dir_path: Optional[Path],
) -> Tuple[Path, List[Group]]:
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

    return out_dir_path, groups


def run_benchmarks(
        exe: executable.Executable,
        valid: validator.Validator,
        benchmark_def_file: Path,
        group_filter: List[str],
        existing_results: ExistingResultsPolicy,
        out_dir_path: Optional[Path],
        verbose: bool
):
    _, groups = parse_benchmark(
        exe,
        benchmark_def_file,
        group_filter,
        out_dir_path
    )

    for group_idx, group in enumerate(groups):
        print(f"-- [{group_idx + 1}/{len(groups)}] Running group {group.name} --")
        try:
            group.run(
                valid,
                existing_results,
                verbose
            )
        except FileExistsError:
            print("Output directory already exists. Run with -r|--rerun to overwrite existing results or -c|--cont to continue computation.", file=sys.stderr)
            raise


def clear_benchmark(
        exe: executable.Executable,
        benchmark_def_file: Path,
        group_filter: List[str],
        out_dir_path: Optional[Path],
):
    out_dir_path, groups = parse_benchmark(
        exe,
        benchmark_def_file,
        group_filter,
        out_dir_path
    )

    if len(group_filter) == 0:
        print(f"Delete {out_dir_path.absolute()}")
        shutil.rmtree(out_dir_path)
    else:
        for group in groups:
            print(f"Delete group directories: {group.result_dir.absolute()}, {group.input_data_dir.absolute()}, {group.output_data_dir.absolute()}")
            shutil.rmtree(group.result_dir)
            shutil.rmtree(group.input_data_dir)
            shutil.rmtree(group.output_data_dir)


def _run_benchmarks(args: argparse.Namespace):
    existing_results_policy = ExistingResultsPolicy.FAIL
    if args.rerun:
        existing_results_policy = existing_results_policy.DELETE
    elif args.cont:
        existing_results_policy = existing_results_policy.CONTINUE

    run_benchmarks(
        executable.Executable(args.executable_path),
        validator.Validator(args.validator_path),
        args.benchmark_definition_path,
        args.groups,
        existing_results_policy,
        args.output_path,
        args.verbose
    )


def _clear_benchmark(args: argparse.Namespace):
    clear_benchmark(
        executable.Executable(args.executable_path),
        args.benchmark_definition_path,
        args.groups,
        args.output_path,
    )


def list_algs(exec: executable.Executable):
    algs = exec.list_algorithms()
    print("\n".join(algs))


def _list_algs(args: argparse.Namespace):
    list_algs(executable.Executable(args.executable_path))


def benchmark_arguments(parser: argparse.ArgumentParser):
    existing_results_flags = parser.add_mutually_exclusive_group()

    existing_results_flags.add_argument("-r", "--rerun",
                        action="store_true",
                        help="Overwrite any previous runs of any of the benchmarked groups")
    existing_results_flags.add_argument("-c", "--cont",
                                        action="store_true",
                                        help="Continue benchmarking, skipping any existing results")

    parser.add_argument("-o", "--output_path",
                        type=Path,
                        help=f"Output directory path (defaults to the name of the benchmark)")

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


def clear_benchmark_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("-o", "--output_path",
                        type=Path,
                        help=f"Output directory path (defaults to the name of the benchmark)")
    parser.add_argument("benchmark_definition_path",
                        type=Path,
                        help="Path to the benchmark definition YAML file")
    parser.add_argument("groups",
                        type=str,
                        nargs="*",
                        help="Groups to run, all by default"
    )
    parser.set_defaults(action=_clear_benchmark)


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
    clear_benchmark_arguments(subparsers.add_parser("clear", help="Clear benchmark"))

    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
