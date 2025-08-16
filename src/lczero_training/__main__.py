# ABOUTME: Main entry point for lczero_training package execution via -m flag.
# ABOUTME: Enables running as python -m lczero_training.
# ABOUTME: Launches the TUI application by default.

import sys
import anyio
from .tui.app import TrainingTuiApp


async def main():
    """Main entry point for the lczero_training package."""
    if len(sys.argv) > 1:
        # If config is provided, could handle it here in the future
        print(f"Config file support not yet implemented: {sys.argv[1]}")
        return

    # Launch TUI in skeleton mode
    app = await TrainingTuiApp.create()
    await app.run_async()


if __name__ == "__main__":
    anyio.run(main)
