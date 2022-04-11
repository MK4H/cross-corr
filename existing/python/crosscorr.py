import argparse
import itertools
import time
import numpy as np
import math

from typing import Tuple, Any, Optional

from scipy import signal
from pathlib import Path

from matrix import MatrixArray


class Timings:
    def __init__(self, adaptive_limit_sec: float):
        self.labels = ["Load", "Computation"]
        self.values = [0] * len(self.labels)
        self.iterations = [0] * len(self.labels)
        self.starts = [0] * len(self.labels)
        self.adaptive_limit_sec = adaptive_limit_sec

    def start(self, label: int):
        self.starts[label] = time.perf_counter_ns()

    def measure(self, label: int):
        now = time.perf_counter_ns()
        self.record(label, now - self.starts[label])

    def adaptive_measure(self, label: int, iterations: int) -> int:
        now = time.perf_counter_ns()
        total_time_ns = now - self.starts[label]
        self.record(label, total_time_ns, iterations)

        total_time_sec = total_time_ns / 1e9
        if total_time_sec >= self.adaptive_limit_sec:
            return 0

        ratio = self.adaptive_limit_sec / total_time_sec
        next_iterations = math.ceil(iterations * max(min(ratio * 1.5, 100), 1.5))
        return next_iterations

    def record(self, label: int, value: int, iterations: int = 1):
        self.values[label] = value / iterations
        self.iterations[label] = iterations

    def save_csv(self, path: Path):
        append = path.exists()
        # Zip and flatten
        values = [val for tup in zip(self.values, self.iterations) for val in tup]

        # Add iteration label for each label
        labels = [[label, f"{label}_iterations"] for label in self.labels]

        # Flatten
        labels = [label for pair in labels for label in pair]
        with path.open("a" if append else "w") as f:
            np.savetxt(f, [values], delimiter=",", fmt="%u", header="" if append else ",".join(labels), comments="")

    def reset(self):
        self.values = [0] * len(self.labels)
        self.iterations = [0] * len(self.labels)
        self.starts = [0] * len(self.labels)


def result_matrix_size(left_size: Tuple[int, int], right_size: Tuple[int, int]) -> Tuple[int, int]:
    return left_size[0] + right_size[0] - 1, left_size[1] + right_size[1] - 1


def one_to_one(
    left: MatrixArray,
    right: MatrixArray,
) -> MatrixArray:
    if left.num_matrices != 1 or right.num_matrices != 1:
        raise ValueError(f"Invalid number of input matrices: left={left.num_matrices}, right={right.num_matrices}")

    result = signal.correlate2d(
        left.get_matrix(0),
        right.get_matrix(0),
    )

    return MatrixArray(result_matrix_size(left.matrix_size, right.matrix_size), 1, result)


def one_to_many(
    left: MatrixArray,
    right: MatrixArray,
) -> MatrixArray:
    if left.num_matrices != 1 or right.num_matrices < 1:
        raise ValueError(f"Invalid number of input matrices: left={left.num_matrices}, right={right.num_matrices}")

    result = MatrixArray.empty(result_matrix_size(left.matrix_size, right.matrix_size), right.num_matrices)

    for idx, matrix in enumerate(right):
        result.write_matrix(
            idx,
            signal.correlate2d(
                left.get_matrix(0),
                matrix,
            ),
        )

    return result


def n_to_mn(
    left: MatrixArray,
    right: MatrixArray,
) -> MatrixArray:
    if left.num_matrices < 1 or right.num_matrices < 1 or right.num_matrices % left.num_matrices != 0:
        raise ValueError(f"Invalid number of input matrices: left={left.num_matrices}, right={right.num_matrices}")

    result = MatrixArray.empty(result_matrix_size(left.matrix_size, right.matrix_size), right.num_matrices)

    # The results should be ordered so we first have the cross-correlation of
    # the n matrices from the left matrix array with the corresponding matrix from the first
    # n matrices in the right matrix array. Then should follow the results of cross-correlation
    # of the n matrices from the left matrix array with the corresponding matrix from
    # the range [n,2n) of matrices from right matrix array etc. ending with
    # results of cross-correlating the n matrices from left matrix array
    # wit hthe matrices [(m-1)*n,m*n) from the right matrix array
    for right_idx, right_matrix in enumerate(right):
        result.write_matrix(
            right_idx,
            signal.correlate2d(
                left.get_matrix(right_idx % left.num_matrices),
                right_matrix,
            ),
        )

    return result


def n_to_m(
    left: MatrixArray,
    right: MatrixArray,
) -> MatrixArray:
    if left.num_matrices < 1 or right.num_matrices < 1:
        raise ValueError(f"Invalid number of input matrices: left={left.num_matrices}, right={right.num_matrices}")

    result = MatrixArray.empty(result_matrix_size(left.matrix_size, right.matrix_size), left.num_matrices * right.num_matrices)

    for left_idx, left_matrix in enumerate(left):
        for right_idx, right_matrix in enumerate(right):
            result.write_matrix(
                left_idx * right.num_matrices + right_idx,
                signal.correlate2d(
                    left_matrix,
                    right_matrix,
                ),
            )

    return result


def run_cross_corr(
        alg: str,
        data_type: Any,
        iterations: int,
        adaptive_limit: float,
        left_input: Path,
        right_input: Path,
        output: Optional[Path],
        timings_path: Optional[Path]
):
    algs = {
        "one_to_one": one_to_one,
        "one_to_many": one_to_many,
        "n_to_mn": n_to_mn,
        "n_to_m": n_to_m,
    }

    if alg not in algs:
        alg_names = ", ".join(algs.keys())
        raise ValueError(f"Invalid algorithm {alg}, expected one of {alg_names}")
    timings = Timings(adaptive_limit)

    timings.start(0)
    left = MatrixArray.load_from_csv(left_input, data_type)
    right = MatrixArray.load_from_csv(right_input, data_type)
    timings.measure(0)

    for iteration in range(iterations):
        measure_iters = 1
        while measure_iters > 0:
            timings.start(1)
            for i in range(measure_iters):
                result = algs[alg](left, right)
            measure_iters = timings.adaptive_measure(1, measure_iters)

        if output is not None:
            output = output.with_name(output.stem + f"_{iteration}" + output.suffix) if iterations > 1 else output
            with output.open("w") as f:
                result.save_to_csv(f)

        if timings_path is not None:
            timings.save_csv(timings_path)

        timings.reset()


def _run_cross_corr(args: argparse.Namespace):
    if args.data_type == "single":
        data_type = np.single
    else:
        data_type = np.double

    run_cross_corr(
        args.algorithm,
        data_type,
        args.iterations,
        args.adaptive_limit,
        args.left_input_path,
        args.right_input_path,
        args.output_path,
        args.timings_path
    )


def arguments(parser: argparse.ArgumentParser):
    parser.add_argument("-o", "--output_path",
                        type=Path,
                        help=f"Output directory path")
    parser.add_argument("-d", "--data_type",
                        default="single",
                        choices=["single", "double"],
                        help="Datatype to be used for computation"
                        )
    parser.add_argument("-i", "--iterations",
                        type=int,
                        default=1,
                        help="Number of iterations to compute with loaded data")
    parser.add_argument("-t", "--timings_path",
                        type=Path,
                        help=f"Path to store time measurements")
    parser.add_argument("-l", "--adaptive_limit",
                        type=float,
                        default=1,
                        help=f"Minimum time measured part of code must run to consider the measurement statistically relevant, in seconds")
    parser.add_argument("algorithm",
                        type=str,
                        help="Algorithms to run")
    parser.add_argument("left_input_path",
                        type=Path,
                        help="Path to the file containing left input data")
    parser.add_argument("right_input_path",
                        type=Path,
                        help="Path to the file containing right input data")
    parser.set_defaults(action=_run_cross_corr)


def main():
    parser = argparse.ArgumentParser(description="Scipy cross-correlation.")
    arguments(parser)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
