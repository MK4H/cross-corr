import argparse
from pathlib import Path
from enum import Enum

import numpy as np


class OutputFormats(Enum):
    CSV = "csv"


DEFAULT_OUTPUT_FORMAT = OutputFormats.CSV
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "data.csv"


def save(num_matrices: int, values: np.ndarray, out_format: OutputFormats, output_path: Path):
    if out_format == OutputFormats.CSV:
        print(f"Saving data to {str(output_path)}")
        with output_path.open(mode='w') as f:
            np.savetxt(
                f,
                values,
                delimiter=',',
                header=f'{int(values.shape[0] / num_matrices)},{values.shape[1]},{num_matrices}')


def generate_data(rows: int, columns: int):
    rng = np.random.default_rng()
    print("Generating random matrix")
    return rng.random(size=(rows, columns))


def generate_matrices(num_matrices: int, rows: int, cols: int, out_format: OutputFormats, out_path: Path):
    values = generate_data(num_matrices * rows, cols)
    save(num_matrices, values, out_format, out_path.absolute())


def _generate_matrix(args: argparse.Namespace):
    generate_matrices(args.num_matrices, args.rows, args.columns, args.format, args.output_path)


def generator_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("-o", "--output_path",
                                # TODO: Add different extensions based on output format
                                default=DEFAULT_OUTPUT_PATH,
                                type=Path,
                                help=f"Output file path (defaults to {str(DEFAULT_OUTPUT_PATH)})")
    parser.add_argument("-f", "--format",
                                type=OutputFormats,
                                choices=list(OutputFormats),
                                default=DEFAULT_OUTPUT_FORMAT,
                                help=f"Output file format (defaults to {DEFAULT_OUTPUT_FORMAT})")
    parser.add_argument("rows",
                                type=int,
                                help=f"Number of rows of the generated matrix")
    parser.add_argument("columns",
                                type=int,
                                help=f"Number of columns of the generated matrix")
    parser.add_argument("num_matrices",
                                nargs="?",
                                type=int,
                                default=1,
                                help=f"Number of matrices of given size to generate (default {1})")
    parser.set_defaults(action=_generate_matrix)


def main():
    parser = argparse.ArgumentParser(description="Tool for generating and transforming input data.")
    input_subparsers = parser.add_subparsers(required=True, dest="input",
                                             description="Generating and transforming input")

    input_generate = input_subparsers.add_parser("generate", help="Generate input matrix")
    generator_arguments(input_generate)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
