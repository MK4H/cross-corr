import argparse

from pathlib import Path
from typing import Tuple

import numpy as np


def partial_sums(
    left_path: Path,
    right_path: Path,
    left_position: Tuple[int, int],
    right_poasition: Tuple[int, int],
    size: Tuple[int, int],
    data_type=np.double
):
    left = np.loadtxt(left_path, dtype=data_type, delimiter=",")
    right = np.loadtxt(right_path, dtype=data_type, delimiter=",")

    # x in the input corresponds to columns, which is the second coordinate in numpy
    left_part = left[left_position[1]:left_position[1]+size[1], left_position[0]:left_position[0]+size[0]]
    right_part = right[right_poasition[1]:right_poasition[1] + size[1], right_poasition[0]:right_poasition[0] + size[0]]

    print(np.sum(np.multiply(left_part, right_part), axis=1))


def _partial_sums(args: argparse.Namespace):
    partial_sums(
        args.left,
        args.right,
        args.left_position,
        args.right_position,
        args.size
    )


def main():
    parser = argparse.ArgumentParser(description="COmpute partial sums of part of the input matrices")
    parser.add_argument("left",
                        type=Path,
                        help="Path to the left matrix csv file")
    parser.add_argument("right",
                        type=Path,
                        help="Path to the right matrix csv file"
    )
    parser.add_argument("left_position",
                        type=int,
                        nargs=2,
                        help="Position of the top left corner of the submatrix to multiply in [x,y], x for columns and y for rows")
    parser.add_argument("right_position",
                        type=int,
                        nargs=2,
                        help="Position of the top left corner of the submatrix to multiply in [x,y], x for columns and y for rows")
    parser.add_argument("size",
                        type=int,
                        nargs=2,
                        help="Size of the submatrices to multiply in [x,y], x for columns and y for rows")
    parser.set_defaults(action=_partial_sums)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
