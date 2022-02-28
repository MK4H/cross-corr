import re
import pandas as pd

from pathlib import Path
from typing import List, Optional, Any, Dict


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

    def __eq__(self, other):
        return self.rows == other.rows and \
               self.columns == other.columns and \
               self.left_matrices == other.left_matrices and \
               self.right_matrices == other.right_matrices

    def matrix_area(self) -> int:
        return self.rows * self.columns

    def total_items(self) -> int:
        return self.matrix_area() * (self.left_matrices + self.right_matrices)


class Run:
    def __init__(self, name: str, args: Optional[str], input_size: InputSize, data: pd.DataFrame):
        self.name = name
        self.args = args
        self.input_size = input_size
        self.data = data

    run_result_filename_regex = re.compile("^[0-9]+-[0-9]+-(.*)__(.*)__-([0-9]+_[0-9]+_[0-9]+_[0-9]+)-time.csv$")

    def matrix_area(self) -> int:
        return self.input_size.matrix_area()

    @classmethod
    def load(cls, path: Path, group_name: str) -> "Run":
        match = cls.run_result_filename_regex.fullmatch(path.name)
        if match:
            name = match.group(1)
            args = match.group(2)
            input_size = InputSize.from_string(match.group(3))
            data = pd.read_csv(
                path
            )

            data["Name"] = name
            data["Args"] = args
            data["Input size"] = str(input_size)
            data["Input total items"] = input_size.total_items()
            data["Input matrix rows"] = input_size.rows
            data["Input matrix cols"] = input_size.columns
            data["Input matrix area"] = input_size.matrix_area()
            data["Input left matrices"] = input_size.left_matrices
            data["Input right matrices"] = input_size.right_matrices
            data["Input type"] = f"{input_size.left_matrices}x{input_size.right_matrices}"
            data["Group"] = group_name

            return cls(
                name,
                args,
                input_size,
                data
            )
        else:
            raise ValueError(f"Invalid file path {str(path)}")


class Group:
    def __init__(self, name: str, runs: List[Run]):
        self.name = name
        self.runs = runs

    group_dir_name_regex = re.compile("^[0-9]*_(.*)$")

    @classmethod
    def load(cls, group_dir_path: Path) -> "Group":
        match = cls.group_dir_name_regex.fullmatch(group_dir_path.name)
        if match:
            group_name = match.group(1)
            run_files = group_dir_path.glob("*-time.csv")
            runs = sorted([Run.load(file, group_name) for file in run_files], key=lambda run: run.input_size.total_items())
            return cls(
                group_name,
                runs
            )
        else:
            raise ValueError(f"Invalid group dir path {str(group_dir_path)}")


class Benchmark:
    def __init__(self, name: str, groups: Dict[str, Group]):
        self.name = name
        self.groups = groups

    @classmethod
    def load(cls, benchmark_dir_path: Path):
        # TODO: Load definition and pull the group names from there
        # TODO: For each group, provide the used configuration
        group_dirs = benchmark_dir_path.glob("*_*")
        groups = [Group.load(group_dir) for group_dir in group_dirs]
        return cls(
            benchmark_dir_path.name,
            {group.name: group for group in groups}
        )

#%%
