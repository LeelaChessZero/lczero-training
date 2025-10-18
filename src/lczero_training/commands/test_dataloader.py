import argparse
import logging
import sys

from lczero_training.commands import configure_root_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch batches from the data loader to measure latency and throughput."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training config file.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=10,
        help="Number of batches to fetch from the data loader.",
    )
    parser.add_argument(
        "--npz-output",
        type=str,
        help="Optional path to store fetched batches as an .npz archive.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_root_logging(logging.INFO)

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Import on demand to avoid importing heavy deps on --help.
    from lczero_training.training.dataloader_probe import probe_dataloader

    probe_dataloader(
        config_filename=args.config,
        num_batches=args.num_batches,
        npz_output=args.npz_output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
