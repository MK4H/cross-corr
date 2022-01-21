import re


class InputSize:
    def __init__(self, rows: int, columns: int, left_matrices: int, right_matrices: int):
        self.rows = rows
        self.columns = columns
        self.left_matrices = left_matrices
        self.right_matrices = right_matrices

    def __repr__(self):
        return f"{self.rows}_{self.columns}_{self.left_matrices}_{self.right_matrices}"

    def __str__(self):
        return f"{self.rows}_{self.columns}_{self.left_matrices}_{self.right_matrices}"

    @classmethod
    def from_string(cls, string: str) -> "InputSize":
        match = re.fullmatch("^([0-9]+)x([0-9]+)x([0-9]+)x([0-9]+)$", string)
        if match:
            return cls(
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
                int(match.group(4))
            )
        raise ValueError(f"Invalid input size {string}")
