import argparse

from .dataloader_probe import probe_dataloader
from .describe import describe
from .eval import eval
from .init import init
from .migrate_checkpoint import migrate_checkpoint
from .overfit import overfit
from .training import train
from .tune_lr import tune_lr


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
    init_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for initializing model parameters.",
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
    eval_parser.add_argument(
        "--dump-shelve",
        type=str,
        help="Dump input/output tensors to specified shelve database.",
    )
    eval_parser.add_argument(
        "--dump-json",
        type=str,
        help="Dump input/output tensors to specified JSON file.",
    )
    eval_parser.add_argument(
        "--onnx-model",
        type=str,
        help="Path to an ONNX model to compare against JAX outputs.",
    )
    eval_parser.add_argument(
        "--no-softmax-jax-wdl",
        action="store_true",
        help="Disable softmaxing the JAX WDL head before comparison.",
    )
    eval_parser.set_defaults(func=run)

    # Tune LR command
    tune_lr_parser = subparsers.add_parser(
        "tune_lr", help="Run a learning rate tuning sweep."
    )
    tune_lr_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training config file.",
    )
    tune_lr_parser.add_argument(
        "--start-lr",
        type=float,
        required=True,
        help="Starting learning rate for the sweep.",
    )
    tune_lr_parser.add_argument(
        "--num-steps",
        type=int,
        required=True,
        help="Number of training steps to evaluate.",
    )
    tune_lr_parser.add_argument(
        "--multiplier",
        type=float,
        default=1.01,
        help="Multiplier applied to the learning rate after each step.",
    )
    tune_lr_parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help=(
            "Optional number of warmup steps to run at a fixed learning "
            "rate before the exponential sweep."
        ),
    )
    tune_lr_parser.add_argument(
        "--warmup-lr",
        type=float,
        help=(
            "Learning rate to use during warmup steps. Required when "
            "--warmup-steps > 0."
        ),
    )
    tune_lr_parser.add_argument(
        "--csv-output",
        type=str,
        help="Optional path to write CSV results (lr, loss).",
    )
    tune_lr_parser.add_argument(
        "--plot-output",
        type=str,
        help="Optional path to save a matplotlib plot of the sweep.",
    )
    tune_lr_parser.add_argument(
        "--num-test-batches",
        type=int,
        default=1,
        help="Number of validation batches to use for computing the loss.",
    )
    tune_lr_parser.set_defaults(func=run)

    # Overfit command
    overfit_parser = subparsers.add_parser(
        "overfit", help="Run an overfitting test on a single batch."
    )
    overfit_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training config file.",
    )
    overfit_parser.add_argument(
        "--num-steps",
        type=int,
        required=True,
        help="Number of training steps to run on the fixed batch.",
    )
    overfit_parser.add_argument(
        "--coin-flip",
        action="store_true",
        help=(
            "Train on two batches: first train batch A while evaluating batch "
            "B, then vice versa."
        ),
    )
    overfit_parser.add_argument(
        "--csv-file",
        type=str,
        help="Optional path to write step-by-step overfit results.",
    )
    overfit_parser.set_defaults(func=run)

    # Describe command
    describe_parser = subparsers.add_parser(
        "describe", help="Describe a trained model."
    )
    describe_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training config file.",
    )
    describe_parser.add_argument(
        "--shapes",
        action="store_true",
        help="Dump model shapes.",
    )
    describe_parser.set_defaults(func=run)

    # Data loader test command
    dataloader_parser = subparsers.add_parser(
        "test-dataloader",
        help=(
            "Fetch batches from the data loader to measure latency and "
            "throughput."
        ),
    )
    dataloader_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training config file.",
    )
    dataloader_parser.add_argument(
        "--num-batches",
        type=int,
        default=10,
        help="Number of batches to fetch from the data loader.",
    )
    dataloader_parser.add_argument(
        "--npz-output",
        type=str,
        help="Optional path to store fetched batches as an .npz archive.",
    )
    dataloader_parser.set_defaults(func=run)

    # Migrate checkpoint command
    migrate_parser = subparsers.add_parser(
        "migrate-checkpoint",
        help="Migrate a checkpoint to a new training state.",
    )
    migrate_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the RootConfig textproto config.",
    )
    migrate_parser.add_argument(
        "--new_checkpoint",
        help="Path to save the new checkpoint to. If not set, the tool only "
        "checks whether the migration rules fully cover the differences.",
    )
    migrate_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, allows overwriting existing checkpoint.",
    )
    migrate_parser.add_argument(
        "--rules_file",
        help="Path to a CheckpointMigrationConfig textproto file containing "
        "the  migration rules.",
    )
    migrate_parser.add_argument(
        "--serialized-model",
        action="store_true",
        default=False,
        help="Use serialized state for a model.",
    )
    migrate_parser.add_argument(
        "--checkpoint_step",
        type=int,
        help="If set, use this step when loading from old checkpoint instead of "
        "the latest.",
    )
    migrate_parser.add_argument(
        "--new_checkpoint_step",
        type=int,
        help="If set, use this step when saving the new checkpoint instead of "
        "copying the old step.",
    )
    migrate_parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    if args.subcommand == "init":
        init(
            config_filename=args.config,
            lczero_model=args.lczero_model,
            seed=args.seed,
        )
    elif args.subcommand == "train":
        train(config_filename=args.config)
    elif args.subcommand == "eval":
        eval(
            config_filename=args.config,
            num_samples=getattr(args, "num_samples", None),
            batch_size_override=getattr(args, "batch_size", None),
            dump_to_stdout=getattr(args, "dump_stdout", False),
            dump_to_file=getattr(args, "dump_file", None),
            dump_to_shelve=getattr(args, "dump_shelve", None),
            dump_to_json=getattr(args, "dump_json", None),
            onnx_model=getattr(args, "onnx_model", None),
            softmax_jax_wdl=not getattr(args, "no_softmax_jax_wdl", False),
        )
    elif args.subcommand == "tune_lr":
        tune_lr(
            config_filename=args.config,
            start_lr=args.start_lr,
            num_steps=args.num_steps,
            multiplier=getattr(args, "multiplier", 1.01),
            warmup_steps=getattr(args, "warmup_steps", 0),
            warmup_lr=getattr(args, "warmup_lr", None),
            csv_output=getattr(args, "csv_output", None),
            plot_output=getattr(args, "plot_output", None),
            num_test_batches=getattr(args, "num_test_batches", 1),
        )
    elif args.subcommand == "overfit":
        overfit(
            config_filename=args.config,
            num_steps=args.num_steps,
            coin_flip=getattr(args, "coin_flip", False),
            csv_file=getattr(args, "csv_file", None),
        )
    elif args.subcommand == "describe":
        describe(
            config_filename=args.config,
            shapes=getattr(args, "shapes", False),
        )
    elif args.subcommand == "migrate-checkpoint":
        migrate_checkpoint(
            config=args.config,
            new_checkpoint=args.new_checkpoint,
            overwrite=args.overwrite,
            rules_file=args.rules_file,
            serialized_model=args.serialized_model,
            checkpoint_step=args.checkpoint_step,
            new_checkpoint_step=args.new_checkpoint_step,
        )
    elif args.subcommand == "test-dataloader":
        probe_dataloader(
            config_filename=args.config,
            num_batches=args.num_batches,
            npz_output=getattr(args, "npz_output", None),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lczero Training CLI.")
    configure_parser(parser)
    args = parser.parse_args()
    args.func(args)
