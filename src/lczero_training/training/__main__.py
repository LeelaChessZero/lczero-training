import argparse

from .eval import eval
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

    # Eval command
    eval_parser = subparsers.add_parser(
        "eval", help="Evaluate a trained model."
    )
    eval_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training config file.",
    )
    eval_parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of samples to evaluate.",
    )
    eval_parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size from data loader config.",
    )
    eval_parser.add_argument(
        "--dump-stdout",
        action="store_true",
        help="Dump input/output tensors to stdout.",
    )
    eval_parser.add_argument(
        "--dump-file",
        type=str,
        help="Dump input/output tensors to specified file.",
    )
    eval_parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    if args.subcommand == "init":
        init(config_filename=args.config, lczero_model=args.lczero_model)
    elif args.subcommand == "train":
        train(config_filename=args.config)
    elif args.subcommand == "eval":
        eval(
            config_filename=args.config,
            num_samples=getattr(args, "num_samples", None),
            batch_size_override=getattr(args, "batch_size", None),
            dump_to_stdout=getattr(args, "dump_stdout", False),
            dump_to_file=getattr(args, "dump_file", None),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lczero Training CLI.")
    configure_parser(parser)
    args = parser.parse_args()
    args.func(args)
