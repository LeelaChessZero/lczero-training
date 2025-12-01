import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Leela Zero weights to JAX format."
    )
    parser.add_argument(
        "input", type=str, help="Path to the input Lc0 weights file."
    )
    parser.add_argument(
        "--output-model-config",
        type=str,
        help="Output path to the ModelConfig textproto.",
    )
    parser.add_argument(
        "--weights-dtype",
        default="F32",
        type=str,
        help="The data type of the weights.",
    )
    parser.add_argument(
        "--compute-dtype",
        default="F32",
        type=str,
        help="The data type for computation.",
    )
    parser.add_argument(
        "--print-model-config",
        action="store_true",
        help="Print the ModelConfig textproto to stdout.",
    )
    parser.add_argument(
        "--output-serialized-jax",
        type=str,
        help="Path to save the output JAX serialized state.",
    )
    parser.add_argument(
        "--output-leela-verification",
        type=str,
        help=(
            "Path to save the round-trip converted Leela network (.pb.gz) for "
            "verification."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Import on demand to avoid importing heavy deps on --help.
    from lczero_training.convert.leela_to_jax import (
        leela_to_jax_files,
    )

    leela_to_jax_files(
        input_path=args.input,
        weights_dtype=args.weights_dtype,
        compute_dtype=args.compute_dtype,
        output_modelconfig=args.output_model_config,
        output_serialized_jax=args.output_serialized_jax,
        output_leela_verification=args.output_leela_verification,
        print_modelconfig=args.print_model_config,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
