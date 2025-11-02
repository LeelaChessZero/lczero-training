import argparse
import sys

from google.protobuf import text_format
from graphviz import Digraph  # type: ignore

from lczero_training.commands import configure_root_logging
from proto.root_config_pb2 import RootConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize the data loader pipeline as a graph."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training config file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the visualization (.svg or .png).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_root_logging()

    parser = _build_parser()
    args = parser.parse_args(argv)

    config = RootConfig()
    with open(args.config, "r") as f:
        text_format.Parse(f.read(), config)

    dot = Digraph(comment="DataLoader Pipeline")
    dot.attr(rankdir="TB")
    dot.attr(
        "node",
        shape="box",
        fontname="monospace",
        fontsize="10",
        labeljust="l",
    )

    stage_names = set()
    for stage in config.data_loader.stage:
        stage_names.add(stage.name)
        stage_text = text_format.MessageToString(stage, as_one_line=False)
        br_tag = '<br align="left"/>'
        escaped_text = (
            stage_text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", br_tag)
        )
        label = f"<{escaped_text}>"
        dot.node(stage.name, label=label, shape="box")

        for input_spec in stage.input:
            source_stage = input_spec.split(".")[0]
            dot.edge(source_stage, stage.name)

    for output_spec in config.data_loader.output:
        parts = output_spec.split(":", 1)
        if len(parts) == 2:
            alias, source = parts
        else:
            alias = output_spec
            source = output_spec

        source_stage = source.split(".")[0]
        dot.node(
            f"output_{alias}",
            label=f"Output: {alias}",
            shape="ellipse",
            style="filled",
            fillcolor="lightblue",
        )
        dot.edge(source_stage, f"output_{alias}")

    output_format = args.output.rsplit(".", 1)[-1].lower()
    if output_format not in ("svg", "png"):
        output_format = "svg"

    dot.render(
        outfile=args.output, format=output_format, cleanup=True, engine="dot"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
