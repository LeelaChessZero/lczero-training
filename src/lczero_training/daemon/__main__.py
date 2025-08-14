# ABOUTME: Module entry point for daemon package execution via -m flag.
# ABOUTME: Enables running daemon as subprocess using python -m lczero_training.daemon.

from .cli import main

if __name__ == "__main__":
    main()
