import argparse
import logging
import sys

from lczero_training.commands import configure_root_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Initialize a new training run from a config file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training config file.",
    )
    parser.add_argument(
        "--lczero_model",
        type=str,
        help="Path to an existing lczero model to start from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for initializing model parameters.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_root_logging(logging.INFO)

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Lazy import to keep --help responsive and avoid heavy deps unless needed.
    from lczero_training.training.init import init

    init(
        config_filename=args.config,
        lczero_model=args.lczero_model,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
