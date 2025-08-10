# ABOUTME: Main entry point for running the lczero-training application.
# ABOUTME: Provides command-line interface to launch the TUI dashboard.

import argparse
import sys

from .tui import TrainingTuiApp
from .config.root_config import RootConfig


def main() -> None:
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Leela Chess Zero training application"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="docs/example.yaml",
        help="Path to configuration YAML file (default: docs/example.yaml)",
    )

    args = parser.parse_args()

    try:
        config = RootConfig.from_yaml_file(args.config)
        app = TrainingTuiApp(config)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    app.run()


if __name__ == "__main__":
    main()
