# ABOUTME: Main entry point for lczero_training package execution via -m flag.
# ABOUTME: Enables running as python -m lczero_training.
# ABOUTME: Launches the TUI application by default.

import anyio

from .app import TrainingTuiApp


async def main() -> None:
    """Main entry point for the lczero_training package."""
    app = TrainingTuiApp()
    await app.run_async()


if __name__ == "__main__":
    anyio.run(main)
