# ABOUTME: Main entry point for lczero_training package execution via -m flag.
# ABOUTME: Enables running as python -m lczero_training.
# ABOUTME: Launches the TUI application by default.

import argparse

import anyio

from .app import TrainingTuiApp


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Delegates argument configuration to the app class."""
    TrainingTuiApp.add_arguments(parser)
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Instantiates and runs the app, injecting the parsed args."""
    anyio.run(main, args)


async def main(args: argparse.Namespace) -> None:
    """Main entry point for the lczero_training package."""
    app = TrainingTuiApp(args=args)
    await app.run_async()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone TUI runner")
    configure_parser(parser)
    args = parser.parse_args()
    args.func(args)
