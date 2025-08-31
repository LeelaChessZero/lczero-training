import argparse

from .leela_to_jax import leela_to_jax


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Leela networks between various formats."
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")
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
        "--output-serialized-jax",
        type=str,
        help="Path to save the output JAX serialized state.",
    )
    leela2jax.add_argument(
        "--output-orbax-checkpoint",
        type=str,
        help="Path to save the output Orbax checkpoint.",
    )

    args = parser.parse_args()

    if args.command == "leela2jax":
        leela_to_jax(
            input_path=args.input,
            weights_dtype=args.weights_dtype,
            compute_dtype=args.compute_dtype,
            output_modelconfig=args.output_model_config,
            output_serialized_jax=args.output_serialized_jax,
            output_orbax_checkpoint=args.output_orbax_checkpoint,
        )


if __name__ == "__main__":
    main()
