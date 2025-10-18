import argparse
import logging
import sys

from lczero_training.commands import configure_root_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Describe a trained model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training config file.",
    )
    parser.add_argument(
        "--shapes",
        action="store_true",
        help="Dump model shapes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_root_logging(logging.INFO)

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Import on demand to avoid importing heavy deps on --help.
    from lczero_training.training.describe import describe

    describe(
        config_filename=args.config,
        shapes=args.shapes,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
