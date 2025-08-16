# ABOUTME: Main entry point for lczero_training package execution via -m flag.
# ABOUTME: Enables running as python -m lczero_training.
# ABOUTME: Launches the TUI application by default.

import argparse
import anyio
from .tui.app import TrainingTuiApp


async def main():
    """Main entry point for the lczero_training package."""
    parser = argparse.ArgumentParser(description="LCZero Training Dashboard")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the training configuration file",
    )
    args = parser.parse_args()

    app = await TrainingTuiApp.create(args.config)
    await app.run_async()


if __name__ == "__main__":
    anyio.run(main)
