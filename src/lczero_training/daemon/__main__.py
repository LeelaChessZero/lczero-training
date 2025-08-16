# ABOUTME: Module entry point for daemon package execution via -m flag.
# ABOUTME: Enables running daemon as subprocess using python -m lczero_training.daemon.
# ABOUTME: Creates and starts daemon instance for IPC communication with parent TUI.

from .daemon import TrainingDaemon


def main() -> None:
    daemon = TrainingDaemon()
    daemon.run()


if __name__ == "__main__":
    main()
