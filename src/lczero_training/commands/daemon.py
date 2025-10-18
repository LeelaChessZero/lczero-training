import argparse
import logging
import sys

from lczero_training.commands import configure_root_logging
from lczero_training.daemon.daemon import TrainingDaemon


def _build_parser() -> argparse.ArgumentParser:
    # Placeholder for future flags; keeps parity with other commands.
    return argparse.ArgumentParser(description="Run the training daemon.")


def main(argv: list[str] | None = None) -> int:
    configure_root_logging(logging.INFO)

    parser = _build_parser()
    parser.parse_args(argv)

    daemon = TrainingDaemon()
    daemon.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
