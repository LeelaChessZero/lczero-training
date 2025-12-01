import argparse
import logging
import sys

from lczero_training.commands import configure_root_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an overfitting test on a single batch."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training config file.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        required=True,
        help="Number of training steps to run on the fixed batch.",
    )
    parser.add_argument(
        "--coin-flip",
        action="store_true",
        help=(
            "Train on two batches: first train batch A while evaluating batch B, then vice versa."
        ),
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        help="Optional path to write step-by-step overfit results.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_root_logging(logging.INFO)

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Import on demand to avoid importing heavy deps on --help.
    from lczero_training.training.overfit import overfit

    overfit(
        config_filename=args.config,
        num_steps=args.num_steps,
        coin_flip=args.coin_flip,
        csv_file=args.csv_file,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
