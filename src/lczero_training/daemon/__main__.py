# ABOUTME: Module entry point for daemon package execution via -m flag.
# ABOUTME: Enables running daemon as subprocess using python -m lczero_training.daemon.
# ABOUTME: Creates and starts daemon instance for IPC communication with parent TUI.

import argparse

from .daemon import TrainingDaemon


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    daemon = TrainingDaemon()
    daemon.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    configure_parser(parser)
    args = parser.parse_args()
    args.func(args)
