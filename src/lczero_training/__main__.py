import argparse
import logging
import sys

from .convert import __main__ as convert_main
from .daemon import __main__ as daemon_main
from .training import __main__ as training_main
from .tui import __main__ as tui_main

COMMANDS = [
    (
        "convert",
        "Convert Leela networks between various formats.",
        convert_main,
    ),
    ("daemon", "Run the training daemon for IPC communication.", daemon_main),
    ("training", "Training related commands.", training_main),
    ("tui", "Launch the TUI application.", tui_main),
]

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(levelname).1s%(asctime)s.%(msecs)03d %(name)s "
        "%(filename)s:%(lineno)d] %(message)s"
    ),
    datefmt="%m%d %H:%M:%S",
    stream=sys.stderr,
    force=True,
)

parser = argparse.ArgumentParser(description="Leela Chess Zero training tools")
subparsers = parser.add_subparsers(dest="command", required=True)

for name, help_text, module in COMMANDS:
    subp = subparsers.add_parser(name, help=help_text)
    module.configure_parser(subp)

args = parser.parse_args()
args.func(args)
