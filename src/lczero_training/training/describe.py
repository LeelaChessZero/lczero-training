import logging
import sys

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from google.protobuf import text_format

from lczero_training.training.state import TrainingState
from proto.root_config_pb2 import RootConfig

logger = logging.getLogger(__name__)


def describe(
    config_filename: str,
    shapes: bool = False,
    values: bool = False,
) -> None:
    config = RootConfig()
    logger.info("Reading configuration from proto file")
    with open(config_filename, "r") as f:
        text_format.Parse(f.read(), config)

    if config.training.checkpoint.path is None:
        logger.error("Checkpoint path must be set in the configuration.")
        sys.exit(1)

    checkpoint_mgr = ocp.CheckpointManager(
        config.training.checkpoint.path,
        options=ocp.CheckpointManagerOptions(
            create=True,
        ),
    )

    logger.info("Creating state from configuration")
    empty_state = TrainingState.new_from_config(
        model_config=config.model,
        training_config=config.training,
    )
    logger.info("Restoring checkpoint")
    training_state = checkpoint_mgr.restore(
        None, args=ocp.args.PyTreeRestore(empty_state)
    )
    logger.info("Restored checkpoint")

    assert isinstance(training_state, TrainingState)

    if values:
        logger.info("Dumping training state values")
        print("Training state:")
        print(training_state)

    if shapes:
        logger.info("Extracting training state shapes")
        shapes = jax.tree.map(jnp.shape, training_state)
        print("Training state shapes:")
        print(shapes)
