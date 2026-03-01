import argparse
import logging
import sys

from lczero_training.commands import configure_root_logging
from lczero_training.daemon.daemon import TrainingDaemon


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the training daemon.")
    parser.add_argument("--memory-profile-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_root_logging(logging.INFO)

    parser = _build_parser()
    args = parser.parse_args(argv)

    daemon = TrainingDaemon(memory_profile_dir=args.memory_profile_dir)
    daemon.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
