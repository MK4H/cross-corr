import sys
import subprocess as sp

from external.execution_error import ExecutionError

from pathlib import Path

from typing import Optional


class BenchmarkScript:
    def __init__(self, script_path: Path):
        self.script_path = script_path

    def run_benchmark(
            self,
            alg_type: str,
            data_type: str,
            inner_iterations: int,
            min_measure_seconds: float,
            left_input_path: Path,
            right_input_path: Path,
            output_data_dir: Optional[Path],
            timings_path: Path,
            verbose: bool
    ):
        command = [self.script_path, alg_type, data_type, str(inner_iterations), str(min_measure_seconds), left_input_path, right_input_path,
            timings_path]

        if output_data_dir is not None:
            command.append(output_data_dir)
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
