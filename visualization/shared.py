import re
import pandas as pd

from pathlib import Path
from typing import List

class InputSize:
    def __init__(self, rows: int, columns: int, left_matrices: int, right_matrices: int):
        self.rows = rows
        self.columns = columns
        self.left_matrices = left_matrices
        self.right_matrices = right_matrices

    @classmethod
    def from_string(cls, string: str) -> "InputSize":
        match = re.fullmatch("^([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)$", string)
        if match:
            return cls(
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
                int(match.group(4))
            )
        raise ValueError(f"Invalid input size {string}")

    def __repr__(self):
        return f"{self.rows}_{self.columns}_{self.left_matrices}_{self.right_matrices}"

    def __str__(self):
        return f"{self.rows}_{self.columns}_{self.left_matrices}_{self.right_matrices}"


class Benchmark:
    def __init__(self, name: str, input_size: InputSize, data: pd.DataFrame):
        self.name = name
        self.input_size = input_size
        self.data = data

    benchmark_result_filename_regex = re.compile("^[0-9]+_[0-9]+_(.*)_([0-9]+_[0-9]+_[0-9]+_[0-9]+)_time.csv$")

    @classmethod
    def load(cls, path: Path) -> "Benchmark":
        match = cls.benchmark_result_filename_regex.fullmatch(path.name)
        if match:
            return cls(
                match.group(1),
                InputSize.from_string(match.group(2)),
                pd.read_csv(
                    path
                )
            )
        else:
            raise ValueError(f"Invalid file path {str(path)}")

def load_group_results(group_dir_path: Path) -> List[Benchmark]:
    benchmark_files = group_dir_path.glob("*_time.csv")
    return [Benchmark.load(file) for file in benchmark_files]
