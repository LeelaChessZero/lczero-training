import argparse
import datetime
import gzip
import logging
import os
import sys

import orbax.checkpoint as ocp
from flax import nnx
from google.protobuf import text_format

from lczero_training.commands import configure_root_logging
from lczero_training.convert.jax_to_leela import (
    LeelaExportOptions,
    jax_to_leela,
)
from lczero_training.dataloader import make_dataloader
from lczero_training.model.loss_function import LczeroLoss
from lczero_training.model.model import LczeroModel
from lczero_training.training.lr_schedule import make_lr_schedule
from lczero_training.training.optimizer import make_gradient_transformation
from lczero_training.training.state import TrainingState
from lczero_training.training.training import Training, from_dataloader
from proto.root_config_pb2 import RootConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start a training run.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training config file.",
    )
    return parser


def train(config_filename: str) -> None:
    config = RootConfig()
    logging.info("Reading configuration from proto file")
    with open(config_filename, "r") as f:
        text_format.Parse(f.read(), config)

    if config.training.checkpoint.path is None:
        logging.error("Checkpoint path must be set in the configuration.")
        sys.exit(1)

    checkpoint_mgr = ocp.CheckpointManager(
        config.training.checkpoint.path,
        options=ocp.CheckpointManagerOptions(
            create=True,
        ),
    )

    logging.info("Creating state from configuration")
    empty_state = TrainingState.new_from_config(
        model_config=config.model,
        training_config=config.training,
    )
    logging.info("Restoring checkpoint")
    training_state = checkpoint_mgr.restore(
        None, args=ocp.args.PyTreeRestore(empty_state)
    )
    logging.info("Restored checkpoint")

    model, _ = nnx.split(
        LczeroModel(config=config.model, rngs=nnx.Rngs(params=42))
    )

    assert isinstance(training_state, TrainingState)

    jit_state = training_state.jit_state
    lr_sched = make_lr_schedule(config.training.lr_schedule)
    optimizer_tx = make_gradient_transformation(
        config.training.optimizer,
        max_grad_norm=getattr(config.training, "max_grad_norm", 0.0),
        lr_schedule=lr_sched,
    )
    training = Training(
        optimizer_tx=optimizer_tx,
        graphdef=model,
        loss_fn=LczeroLoss(config=config.training.losses),
        swa_config=(
            config.training.swa if config.training.HasField("swa") else None
        ),
    )
    new_state = training.run(
        jit_state,
        from_dataloader(make_dataloader(config.data_loader)),
        config.training.schedule.steps_per_network,
    )

    if config.export.destination_filename:
        date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        logging.info("Exporting network")

        options = LeelaExportOptions(
            min_version="0.28",
            num_heads=training_state.num_heads,
            license=None,
        )
        export_state = (
            new_state.swa_state
            if config.export.export_swa_model
            else new_state.model_state
        )
        assert isinstance(export_state, nnx.State)
        net = jax_to_leela(jax_weights=export_state, export_options=options)
        network_bytes = gzip.compress(net.SerializeToString())

        for destination_template in config.export.destination_filename:
            destination = destination_template.format(
                datetime=date_str, step=new_state.step
            )
            logging.info(f"Writing network to {destination}")
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            with open(destination, "wb") as f:
                f.write(network_bytes)
            logging.info(f"Finished writing network to {destination}")


def main(argv: list[str] | None = None) -> int:
    configure_root_logging(logging.INFO)

    parser = _build_parser()
    args = parser.parse_args(argv)

    train(config_filename=args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
