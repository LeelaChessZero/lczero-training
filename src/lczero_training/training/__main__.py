import argparse

from .init import init
from .training import train


def configure_parser(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Init command
    init_parser = subparsers.add_parser(
        "init", help="Initialize a new training run from a config file."
    )
    init_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training config file.",
    )
    init_parser.add_argument(
        "--lczero_model",
        type=str,
        help="Path to an existing lczero model to start from.",
    )
    init_parser.set_defaults(func=run)

    # Train command
    train_parser = subparsers.add_parser("train", help="Start a training run.")
    train_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training config file.",
    )
    train_parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    if args.subcommand == "init":
        init(config_filename=args.config, lczero_model=args.lczero_model)
    elif args.subcommand == "train":
        train(config_filename=args.config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lczero Training CLI.")
    configure_parser(parser)
    args = parser.parse_args()
    args.func(args)
