# ABOUTME: Module entry point for TUI package execution via -m flag.
# ABOUTME: Enables running TUI as python -m lczero_training.tui.
# ABOUTME: Creates and starts TUI application instance.

from .app import TrainingTuiApp


def main():
    app = TrainingTuiApp()
    app.run()


if __name__ == "__main__":
    main()
