import argparse
import logging
import sys

from lczero_training.commands import configure_root_logging
from lczero_training.training.backfill_metrics import backfill_metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill metrics for existing checkpoints."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the RootConfig textproto config.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="Names of metrics to backfill (must be NPZ metrics).",
    )
    parser.add_argument(
        "--min-step",
        type=int,
        help="Minimum checkpoint step (inclusive) to process.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        help="Maximum checkpoint step (inclusive) to process.",
    )
    parser.add_argument(
        "--migration-config",
        help=(
            "Path to a CheckpointMigrationConfig textproto file. "
            "If provided, checkpoints will be migrated before evaluation."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_root_logging(logging.INFO)

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Lazy import to avoid heavy deps unless executing the command.

    backfill_metrics(
        config_path=args.config,
        metric_names=args.metrics,
        min_step=args.min_step,
        max_step=args.max_step,
        migration_config_path=args.migration_config,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
