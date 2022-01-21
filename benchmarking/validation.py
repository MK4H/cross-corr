import argparse
import tempfile

from pathlib import Path

from external import validator, executable


def validate_from_inputs(
        val: validator.Validator,
        exe: executable.Executable,
        alg_type: str,
        left_input: Path,
        right_input: Path,
        data_path: Path,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "valid_data.csv"
        val.generate_validation_data(
            alg_type,
            left_input,
            right_input,
            tmp_path
        )

        exe.validate_output(data_path, tmp_path)


def generate_valid_output(
        val: validator.Validator,
        alg_type: str,
        left_input: Path,
        right_input: Path,
        output: Path
):
    val.generate_validation_data(
        alg_type,
        left_input,
        right_input,
        output
    )


def validate_from_pregenerated(
    exe: executable.Executable,
    data_path: Path,
    valid_data_path: Path,
):
    exe.validate_output(data_path, valid_data_path)


def _validate_from_inputs(args: argparse.Namespace):
    validate_from_inputs(
        validator.Validator(args.validator_path),
        executable.Executable(args.executable_path),
        args.alg_type,
        args.left_input_path,
        args.right_input_path,
        args.data_path,
    )


def _generate_valid_output(args: argparse.Namespace):
    generate_valid_output(
        validator.Validator(args.validator_path),
        args.alg_type,
        args.left_input_path,
        args.right_input_path,
        args.output_path,
    )


def _validate_from_pregenerated(args: argparse.Namespace):
    validate_from_pregenerated(
        executable.Executable(args.executable_path),
        args.data_path,
        args.valid_data_path,
    )


def validate_from_inputs_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("alg_type",
                        type=str,
                        help="Type of the algorithm to generate valid data for")
    parser.add_argument("left_input_path",
                        type=Path,
                        help="Path to the file containing left input data")
    parser.add_argument("right_input_path",
                        type=Path,
                        help="Path to the file containing right input data")
    parser.add_argument("data_path",
                        type=Path,
                        help="Path to the data to be validated")
    parser.set_defaults(action=_validate_from_inputs)

def generate_valid_output_arguments(parser: argparse.ArgumentParser):
    default_output_path = Path.cwd() / "valid_data.csv"
    parser.add_argument("-o", "--output_path",
                        default=default_output_path,
                        type=Path,
                        help=f"Output directory path (defaults to {str(default_output_path)})")
    parser.add_argument("alg_type",
                        type=str,
                        help="Type of the algorithm to generate valid data for")
    parser.add_argument("left_input_path",
                        type=Path,
                        help="Path to the file containing left input data")
    parser.add_argument("right_input_path",
                        type=Path,
                        help="Path to the file containing right input data")
    parser.set_defaults(action=_generate_valid_output)


def validate_from_pregenerated_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("data_path",
                        type=Path,
                        help="Path to the data to be validated")
    parser.add_argument("valid_data_path",
                        type=Path,
                        help="Path to the valid data to validate against")
    parser.set_defaults(action=_validate_from_pregenerated)


def global_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("-e", "--executable_path",
                        default=executable.EXECUTABLE_PATH,
                        type=Path,
                        help=f"Path to the implementation executable (defaults to {str(executable.EXECUTABLE_PATH)})")
    parser.add_argument("-p", "--validator_path",
                        default=validator.VALIDATOR_PATH,
                        type=Path,
                        help=f"Path to the validator shell script (defaults to {str(validator.VALIDATOR_PATH)}")


def add_subparsers(parser: argparse.ArgumentParser):
    subparsers = parser.add_subparsers(required=True,
                                       dest="validation",
                                       description="Generating valid outputs and checking validity of outputs"
                                       )
    validate_from_inputs_arguments(
        subparsers.add_parser(
            "from_inputs",
            help="Validate output against the expected output based on the given inputs"
        )
    )
    validate_from_pregenerated_arguments(
        subparsers.add_parser(
            "from_pregenerated",
            help="Validate output against pregenerated valid output")
    )
    generate_valid_output_arguments(
        subparsers.add_parser(
            "generate",
            help="Generate valid output from inputs"
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Generate valid outputs or validate outputs")
    global_arguments(parser)
    add_subparsers(parser)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
