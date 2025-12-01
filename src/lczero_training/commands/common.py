import argparse
import logging
import os
import sys

_DEFAULT_FORMAT = (
    "%(levelname).1s%(asctime)s.%(msecs)03d %(name)s "
    "%(filename)s:%(lineno)d] %(message)s"
)
_DEFAULT_DATEFMT = "%m%d %H:%M:%S"


def configure_root_logging(level: int | str = logging.INFO) -> None:
    """Configure root logging with a consistent, terse format.

    - Matches existing project format used in module __main__ files.
    - Respects explicit level passed as int or name (e.g. "INFO").
    - Uses stderr by default.
    - Forces reconfiguration to avoid duplicate handlers during nested runs.
    """

    resolved_level = parse_log_level(level)
    logging.basicConfig(
        level=resolved_level,
        format=_DEFAULT_FORMAT,
        datefmt=_DEFAULT_DATEFMT,
        stream=sys.stderr,
        force=True,
    )


def parse_log_level(level: int | str) -> int:
    """Parse log level from int or string, with sane defaults.

    Accepts numeric levels or case-insensitive names like "DEBUG".
    Falls back to INFO on invalid input.
    """
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        name = level.strip().upper()
        return getattr(logging, name, logging.INFO)
    return logging.INFO


def add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common logging CLI arguments to a parser.

    Does not enable the flags by default; commands may opt-in and then call
    configure_root_logging(parse_log_level(args.log_level)) if present.
    """
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LCZERO_LOG_LEVEL", "INFO"),
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
