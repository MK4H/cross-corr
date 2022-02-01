import sys
import subprocess as sp

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
            print("Failed to run external benchmark", file=sys.stderr)
            print(f"Exit code: {res.returncode}", file=sys.stderr)
            #print(f"Stdout: {res.stdout}", file=sys.stderr)
            print(f"Stderr: {res.stderr}", file=sys.stderr)
            sys.exit(2)
