import sys
import subprocess as sp

from external.execution_error import ExecutionError

from pathlib import Path


class BenchmarkScript:
    def __init__(self, script_path: Path):
        self.script_path = script_path

    def run_benchmark(
            self,
            alg_type: str,
            data_type: str,
            iterations: int,
            left_input_path: Path,
            right_input_path: Path,
            output_data_dir: Path,
            timings_path: Path,
            verbose: bool
    ):
        command = [self.script_path, alg_type, data_type, str(iterations), left_input_path, right_input_path, output_data_dir,
            timings_path]
        res = sp.run(
            command,
            stderr=sp.PIPE,
            text=True
        )
        if verbose:
            print(f"Command: {command}")

        if res.returncode != 0:
            raise ExecutionError(
                "Failed to run external benchmark",
                self.script_path,
                res.returncode,
                "",
                res.stderr
            )
