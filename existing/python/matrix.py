import re

import numpy as np

from pathlib import Path
from typing import Tuple, Any


class MatrixArray:
    def __init__(self, matrix_size: Tuple[int, int], num_matrices: int, data):
        self.matrix_size = matrix_size
        self.num_matrices = num_matrices
        self.data = data

    @classmethod
    def empty(cls, matrix_size: Tuple[int, int], num_matrices: int):
        data = np.empty((matrix_size[0]*num_matrices, matrix_size[1]))
        return cls(matrix_size, num_matrices, data)

    @classmethod
    def load_from_csv(cls, path: Path, data_type: Any) -> "MatrixArray":
        with path.open("r") as f:
            header = f.readline()
            match = re.fullmatch("# ([0-9]+),([0-9]+),([0-9]+)?", header.strip())
            if match:
                rows = int(match.group(1))
                cols = int(match.group(2))
                num_matrices = int(match.group(3)) if match.group(3) is not None else 1
            else:
                raise ValueError(f"Failed to read header from {str(path)}")

        data = np.loadtxt(path, dtype=data_type, delimiter=",")
        return cls((rows, cols), num_matrices, data)

    def save_to_csv(self, file):
        np.savetxt(
            file,
            self.data,
            delimiter=",",
            header=f"{self.matrix_size[0]},{self.matrix_size[1]},{self.num_matrices}"
        )

    def get_matrix(self, idx: int):
        if idx >= self.num_matrices:
            raise ValueError(f"Index out of bounds, requested matrix {idx} out of {self.num_matrices}")

        return self.data[idx*self.matrix_size[0]:(idx + 1)*self.matrix_size[0], :]

    def write_matrix(self, idx: int, data):
        self.data[idx*self.matrix_size[0]:(idx + 1)*self.matrix_size[0], :] = data

    def __iter__(self):
        return (self.get_matrix(i) for i in range(self.num_matrices))
