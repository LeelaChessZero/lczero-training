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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip checkpoint creation.",
    )
    parser.add_argument(
        "--swa_initial_nets",
        type=int,
        default=0,
        help="Initial value for num_averages in SWA state.",
    )
    parser.add_argument(
        "--override_training_steps",
        type=int,
        help="Override training step number.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Allow overwriting existing checkpoint path.",
    )
    parser.add_argument(
        "--no-copy-swa",
        action="store_true",
        help="Don't copy model weights to SWA state.",
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
        dry_run=args.dry_run,
        swa_initial_nets=args.swa_initial_nets,
        override_training_steps=args.override_training_steps,
        override=args.override,
        no_copy_swa=args.no_copy_swa,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
