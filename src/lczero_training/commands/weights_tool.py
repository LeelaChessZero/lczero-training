"""CLI command for manipulating Lc0 neural network weights."""

import argparse
import sys

import numpy as np

from lczero_training.commands import configure_root_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manipulate Lc0 neural network weights."
    )
    parser.add_argument(
        "--expr",
        type=str,
        help=(
            "Python expression to execute "
            "(e.g., \"output = weights('A.pb') * 0.5\")"
        ),
    )
    parser.add_argument(
        "script",
        nargs="?",
        help="Path to Python script file (if --expr not provided)",
    )
    parser.add_argument(
        "--input",
        type=str,
        action="append",
        help="Pre-load input as NAME=PATH (e.g., --input A=net_A.pb.gz)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help='Default output path if "output" variable is set',
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="FLOAT16",
        choices=["LINEAR16", "FLOAT16", "BFLOAT16"],
        help="Output encoding format (default: FLOAT16)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_root_logging()

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Lazy import to avoid heavy dependencies on --help.
    from lczero_training.tools.weights_tool import load_weights, save_weights
    from proto import net_pb2

    # Build execution environment.
    env = {
        "np": np,
        "weights": load_weights,
        "save": save_weights,
        "lc0": net_pb2,
    }

    # Pre-load inputs.
    if args.input:
        for input_spec in args.input:
            if "=" not in input_spec:
                print(
                    f"Error: Invalid input spec '{input_spec}'. "
                    "Expected format: NAME=PATH",
                    file=sys.stderr,
                )
                return 1
            name, path = input_spec.split("=", 1)
            env[name] = load_weights(path)

    # Determine script source: --expr, file, or stdin.
    if args.expr:
        script = args.expr
    elif args.script:
        with open(args.script) as f:
            script = f.read()
    else:
        if sys.stdin.isatty():
            print(
                "Error: No script provided. Use --expr, provide script file, "
                "or pipe to stdin.",
                file=sys.stderr,
            )
            return 1
        script = sys.stdin.read()

    # Execute script.
    try:
        exec(script, env)
    except Exception as e:
        print(f"Error executing script: {e}", file=sys.stderr)
        return 1

    # Auto-save if 'output' variable is set.
    if "output" in env and args.output:
        from lczero_training.tools.weight_wrappers import NetWrapper

        output = env["output"]
        if isinstance(output, NetWrapper):
            save_weights(output, args.output, args.encoding)
            print(f"Saved result to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
