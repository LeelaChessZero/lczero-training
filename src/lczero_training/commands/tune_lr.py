import argparse
import logging
import sys

from lczero_training.commands import configure_root_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a learning rate tuning sweep."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training config file.",
    )
    parser.add_argument(
        "--start-lr",
        type=float,
        required=True,
        help="Starting learning rate for the sweep.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        required=True,
        help="Number of training steps to evaluate.",
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=1.01,
        help="Multiplier applied to the learning rate after each step.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help=(
            "Optional number of warmup steps to run at a fixed learning rate before "
            "the exponential sweep."
        ),
    )
    parser.add_argument(
        "--warmup-lr",
        type=float,
        help=(
            "Learning rate to use during warmup steps. Required when --warmup-steps > 0."
        ),
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        help=(
            "Optional path to write CSV results. Columns: lr, train_loss[, val_loss]."
        ),
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        help="Optional path to save a matplotlib plot of the sweep.",
    )
    parser.add_argument(
        "--num-test-batches",
        type=int,
        default=0,
        help=(
            "When > 0, also compute and report validation loss on this many fixed batches "
            "(averaged each step). Default 0 (training loss only)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_root_logging(logging.INFO)

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Lazy import to keep --help responsive and avoid heavy deps unless needed.
    from lczero_training.training.tune_lr import tune_lr

    tune_lr(
        config_filename=args.config,
        start_lr=args.start_lr,
        num_steps=args.num_steps,
        multiplier=args.multiplier,
        warmup_steps=args.warmup_steps,
        warmup_lr=args.warmup_lr,
        csv_output=args.csv_output,
        plot_output=args.plot_output,
        num_test_batches=args.num_test_batches,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
