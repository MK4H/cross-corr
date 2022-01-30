import subprocess as sp
import sys

from external.input_size import InputSize

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
            print("Failed to run input size validation", file=sys.stderr)
            print(f"Exit code: {res.returncode}", file=sys.stderr)
            print(f"Stdout: {res.stdout}", file=sys.stderr)
            print(f"Stderr: {res.stderr}", file=sys.stderr)
            sys.exit(2)

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
            print("Failed to list algorithms", file=sys.stderr)
            print(f"Exit code: {res.returncode}", file=sys.stderr)
            print(f"Stdout: {res.stdout}", file=sys.stderr)
            print(f"Stderr: {res.stderr}", file=sys.stderr)
            sys.exit(2)

    def validate_output(self, output: Path, valid_data: Path):
        res = sp.run(
            [
                str(self.executable_path.absolute()),
                "validate",
                str(output.absolute()),
                str(valid_data.absolute())
            ],
            capture_output=True,
            text=True
        )

        if res.returncode == 0:
            print(res.stdout)
        else:
            print("Failed to run output validation", file=sys.stderr)
            print(f"Exit code: {res.returncode}", file=sys.stderr)
            print(f"Stdout: {res.stdout}", file=sys.stderr)
            print(f"Stderr: {res.stderr}", file=sys.stderr)
            sys.exit(2)

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
            print("Failed to run benchmark", file=sys.stderr)
            print(f"Exit code: {res.returncode}", file=sys.stderr)
            print(f"Stdout: {res.stdout}", file=sys.stderr)
            print(f"Stderr: {res.stderr}", file=sys.stderr)
            sys.exit(2)

        if validation_data_path is not None:
            with output_stats_path.open("a" if append else "w") as f:
                print(res.stdout, file=f, end="")
