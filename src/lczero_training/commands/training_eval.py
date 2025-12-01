import argparse
import logging
import sys

from lczero_training.commands import configure_root_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training config file.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of samples to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size from data loader config.",
    )
    parser.add_argument(
        "--dump-stdout",
        action="store_true",
        help="Dump input/output tensors to stdout.",
    )
    parser.add_argument(
        "--dump-file",
        type=str,
        help="Dump input/output tensors to specified file.",
    )
    parser.add_argument(
        "--dump-shelve",
        type=str,
        help="Dump input/output tensors to specified shelve database.",
    )
    parser.add_argument(
        "--dump-json",
        type=str,
        help="Dump input/output tensors to specified JSON file.",
    )
    parser.add_argument(
        "--onnx-model",
        type=str,
        help="Path to an ONNX model to compare against JAX outputs.",
    )
    parser.add_argument(
        "--no-softmax-jax-wdl",
        action="store_true",
        help="Disable softmaxing the JAX WDL head before comparison.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_root_logging(logging.INFO)

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Lazy import to keep --help responsive and avoid heavy deps unless needed.
    from lczero_training.training.eval import eval as eval_fn

    eval_fn(
        config_filename=args.config,
        num_samples=args.num_samples,
        batch_size_override=args.batch_size,
        dump_to_stdout=args.dump_stdout,
        dump_to_file=args.dump_file,
        dump_to_shelve=args.dump_shelve,
        dump_to_json=args.dump_json,
        onnx_model=args.onnx_model,
        softmax_jax_wdl=not args.no_softmax_jax_wdl,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
