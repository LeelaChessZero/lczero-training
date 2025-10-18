"""Command entrypoint scaffolding and shared CLI helpers.

This package will host thin wrappers for individual tools (convert,
training, daemon, tui) as they are extracted from nested module
__main__ dispatchers in subsequent phases.

Phase 1 provides common helpers for consistent logging and argument
handling across commands without changing behaviour.
"""

from .common import (
    add_logging_arguments,
    configure_root_logging,
    parse_log_level,
)

__all__ = [
    "configure_root_logging",
    "add_logging_arguments",
    "parse_log_level",
]
