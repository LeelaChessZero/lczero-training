import argparse
import logging
import sys

import anyio

from lczero_training.commands import configure_root_logging
from lczero_training.tui.app import TrainingTuiApp


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training TUI runner")
    TrainingTuiApp.add_arguments(parser)
    return parser


async def _amain(args: argparse.Namespace) -> None:
    app = TrainingTuiApp(args=args)
    await app.run_async()


def main(argv: list[str] | None = None) -> int:
    configure_root_logging(logging.INFO)
    parser = _build_parser()
    args = parser.parse_args(argv)
    anyio.run(_amain, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
