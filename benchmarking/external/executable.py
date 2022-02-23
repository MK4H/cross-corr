import subprocess as sp
import sys

from external.input_size import InputSize
from external.execution_error import ExecutionError

from typing import Optional, List, Dict, Any
from pathlib import Path

EXECUTABLE_PATH = Path(__file__).parent.parent.parent / "build" / "cross"


class Executable:
    def __init__(self, executable_path: Path):
        self.executable_path = executable_path

    def validate_input_size(self, alg_type: str, size: InputSize) -> bool:
        res = sp.run(
            [
                str(self.executable_path.absolute()),
                "input",
                alg_type,
                str(size.rows),
                str(size.columns),
                str(size.left_matrices),
                str(size.right_matrices)
            ],
            capture_output=True,
            text=True
        )

        if res.returncode == 0:
            return res.stdout.startswith("Valid")
        else:
            raise ExecutionError(
                "Failed to run input size validation",
                self.executable_path,
                res.returncode,
                res.stdout,
                res.stderr
            )

    def list_algorithms(self) -> List[str]:
        res = sp.run(
            [
                str(self.executable_path.absolute()),
                "list",
            ],
            capture_output=True,
            text=True
        )

        if res.returncode == 0:
            return res.stdout.splitlines()
        else:
            raise ExecutionError(
                "Failed to list algorithms",
                self.executable_path,
                res.returncode,
                res.stdout,
                res.stderr
            )

    def validate_data(self, valid_data: Path, data_paths: List[Path], csv: bool = False, normalize: bool = False) -> str:
        command = [
            str(self.executable_path.absolute()),
            "validate",
        ]
        if csv:
            command.append("--csv")

        if normalize:
            command.append("--normalize")

        command.append(str(valid_data.absolute()))
        command.extend((str(path.absolute()) for path in data_paths))
        res = sp.run(
            command,
            capture_output=True,
            text=True
        )

        if res.returncode == 0:
            return res.stdout
        else:
            raise ExecutionError(
                "Failed to run output validation",
                self.executable_path,
                res.returncode,
                res.stdout,
                res.stderr
            )

    def run_benchmark(
        self,
        alg: str,
        data_type: str,
        args_path: Path,
        left_input_path: Path,
        right_input_path: Path,
        output_data_path: Path,
        timings_path: Path,
        output_stats_path: Path,
        append: bool,
        validation_data_path: Optional[Path],
        verbose: bool
    ):
        # Must be joined after the optional args
        # as boost program options does not handle optional values
        # with options well, so we have to end it in --no_progress
        default_options = [
            "--out", str(output_data_path.absolute()),
            "--times", str(timings_path.absolute()),
            "--no_progress",
            "--args_path", str(args_path.absolute()),
            "--data_type", str(data_type),
        ]

        positional_args = [
           alg,
           str(left_input_path.absolute()),
           str(right_input_path.absolute()),
        ]

        optional_options = []

        if validation_data_path is not None:
            optional_options.append("--validate")
            optional_options.append(str(validation_data_path.absolute()))

        if append:
            optional_options.append("--append")

        command = [str(self.executable_path.absolute()), "run"] + optional_options + default_options + positional_args
        res = sp.run(
            command,
            capture_output=True,
            text=True
        )
        if verbose:
            print(f"Command: {command}")

        if res.returncode != 0:
            raise ExecutionError(
                "Failed to run benchmark",
                self.executable_path,
                res.returncode,
                res.stdout,
                res.stderr
            )

        if validation_data_path is not None:
            with output_stats_path.open("a" if append else "w") as f:
                print(res.stdout, file=f, end="")
