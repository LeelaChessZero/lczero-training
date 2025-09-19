import argparse

from .leela_to_jax import leela_to_jax_files


def configure_parser(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(
        dest="subcommand", help="Sub-command help"
    )
    subparsers.required = True

    leela2jax = subparsers.add_parser(
        "leela2jax", help="Convert Leela Zero weights to JAX format."
    )
    leela2jax.add_argument(
        "input", type=str, help="Path to the input Lc0 weights file."
    )
    leela2jax.add_argument(
        "--output-model-config",
        type=str,
        help="Output path to the ModelConfig textproto.",
    )
    leela2jax.add_argument(
        "--weights-dtype",
        default="F32",
        type=str,
        help="The data type of the weights.",
    )
    leela2jax.add_argument(
        "--compute-dtype",
        default="F32",
        type=str,
        help="The data type for computation.",
    )
    leela2jax.add_argument(
        "--print-model-config",
        action="store_true",
        help="Print the ModelConfig textproto to stdout.",
    )
    leela2jax.add_argument(
        "--output-serialized-jax",
        type=str,
        help="Path to save the output JAX serialized state.",
    )
    leela2jax.add_argument(
        "--output-orbax-checkpoint",
        type=str,
        help="Path to save the output Orbax checkpoint.",
    )
    leela2jax.add_argument(
        "--output-leela-verification",
        type=str,
        help="Path to save the round-trip converted Leela network (.pb.gz) for verification.",
    )

    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    if args.subcommand == "leela2jax":
        leela_to_jax_files(
            input_path=args.input,
            weights_dtype=args.weights_dtype,
            compute_dtype=args.compute_dtype,
            output_modelconfig=args.output_model_config,
            output_serialized_jax=args.output_serialized_jax,
            output_orbax_checkpoint=args.output_orbax_checkpoint,
            output_leela_verification=args.output_leela_verification,
            print_modelconfig=args.print_model_config,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Leela networks between various formats."
    )
    configure_parser(parser)
    args = parser.parse_args()
    args.func(args)
