import argparse
import logging
import sys

from lczero_training.commands import configure_root_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migrate a checkpoint to a new training state."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the RootConfig textproto config.",
    )
    parser.add_argument(
        "--new_checkpoint",
        help=(
            "Path to save the new checkpoint to. If not set, the tool only "
            "checks whether the migration rules fully cover the differences."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, allows overwriting existing checkpoint.",
    )
    parser.add_argument(
        "--rules_file",
        help=(
            "Path to a CheckpointMigrationConfig textproto file containing "
            "the migration rules."
        ),
    )
    parser.add_argument(
        "--serialized-model",
        action="store_true",
        default=False,
        help="Use serialized state for a model.",
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        help=(
            "If set, use this step when loading from old checkpoint instead "
            "of the latest."
        ),
    )
    parser.add_argument(
        "--new_checkpoint_step",
        type=int,
        help=(
            "If set, use this step when saving the new checkpoint instead of "
            "copying the old step."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_root_logging(logging.INFO)

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Lazy import to avoid heavy deps unless executing the command.
    from lczero_training.training.migrate_checkpoint import (
        migrate_checkpoint,
    )

    migrate_checkpoint(
        config=args.config,
        new_checkpoint=args.new_checkpoint,
        overwrite=args.overwrite,
        rules_file=args.rules_file,
        serialized_model=args.serialized_model,
        checkpoint_step=args.checkpoint_step,
        new_checkpoint_step=args.new_checkpoint_step,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
