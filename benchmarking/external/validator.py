import subprocess as sp
import sys

from pathlib import Path

VALIDATOR_PATH = Path(__file__).parent.parent.parent / "existing" / "matlab" / "compute_valid_results.sh"


class Validator:
    def __init__(self, validator_path: Path):
        self.validator_path = validator_path

    def generate_validation_data(
            self,
            alg_type: str,
            left_path: Path,
            right_path: Path,
            out_path: Path,
    ):
        res = sp.run(
            [
                str(self.validator_path.absolute()),
                alg_type,
                # TODO: Add as commandline argument
                "single",
                str(left_path.absolute()),
                str(right_path.absolute()),
                str(out_path.absolute())
            ],
            capture_output=True,
            text=True
        )

        if res.returncode != 0:
            print("Failed to generate validation data", file=sys.stderr)
            print(f"Exit code: {res.returncode}", file=sys.stderr)
            print(f"Stdout: {res.stdout}", file=sys.stderr)
            print(f"Stderr: {res.stderr}", file=sys.stderr)
            sys.exit(2)
