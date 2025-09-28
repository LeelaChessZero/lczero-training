import gzip
import logging
import os
import sys
from typing import Optional

import orbax.checkpoint as ocp
from flax import nnx
from google.protobuf import text_format

from lczero_training.convert.leela_to_jax import (
    LeelaImportOptions,
    fix_older_weights_file,
    leela_to_jax,
)
from lczero_training.convert.leela_to_modelconfig import leela_to_modelconfig
from lczero_training.model.model import LczeroModel
from lczero_training.training.state import TrainingState
from proto import hlo_pb2, net_pb2
from proto.root_config_pb2 import RootConfig

logger = logging.getLogger(__name__)


def init(config_filename: str, lczero_model: Optional[str]) -> None:
    """
    Initializes a new training run.
    """

    config = RootConfig()
    logger.info("Reading configuration from proto file")
    with open(config_filename, "r") as f:
        text_format.Parse(f.read(), config)

    if config.training.checkpoint.path is None:
        logger.error("Checkpoint path must be set in the configuration.")
        sys.exit(1)

    if os.path.exists(config.training.checkpoint.path):
        logger.error(
            f"Checkpoint path {config.training.checkpoint.path} already exists."
        )
        sys.exit(1)

    logger.info("Creating initial training state from configuration")
    training_state = TrainingState.new_from_config(
        model_config=config.model,
        training_config=config.training,
    )
    logger.info("Creating JAX FLAX/NNX model from configuration")
    model = LczeroModel(config=config.model, rngs=nnx.Rngs(params=42))
    model_state = nnx.state(model)

    if lczero_model is not None:
        logger.info(f"Using existing lczero model: {lczero_model}")
        lc0_weights = net_pb2.Net()
        with gzip.open(lczero_model, "rb") as f:
            contents = f.read()
            assert isinstance(contents, bytes)
            lc0_weights.ParseFromString(contents)
        fix_older_weights_file(lc0_weights)

        logger.info("Converting leela weights to model configuration")
        leela_config = leela_to_modelconfig(
            lc0_weights,
            hlo_pb2.XlaShapeProto.F32,
            config.model.defaults.compute_dtype,
        )
        if leela_config != config.model:
            logger.error(
                "The provided lczero model configuration "
                "differs from the one in the config file."
            )
            logger.error(f"Config file model config: {config.model}")
            logger.error(f"Leela model config: {leela_config}")
            sys.exit(1)

        logger.info("Loading leela weights into JAX model")
        import_options = LeelaImportOptions(
            weights_dtype=hlo_pb2.XlaShapeProto.F32,
            compute_dtype=config.model.defaults.compute_dtype,
        )
        model_state = leela_to_jax(lc0_weights, import_options)

        training_state = training_state.replace(
            jit_state=training_state.jit_state.replace(
                step=lc0_weights.training_params.training_steps,
                model_state=model_state,
            )
        )

    checkpoint_mgr = ocp.CheckpointManager(
        config.training.checkpoint.path,
        options=ocp.CheckpointManagerOptions(
            create=True,
        ),
    )

    logger.info(
        f"Saving initial checkpoint to {config.training.checkpoint.path}"
    )
    checkpoint_mgr.save(
        step=training_state.jit_state.step,
        args=ocp.args.PyTreeSave(training_state),
    )
    checkpoint_mgr.wait_until_finished()
    logger.info("Initialization complete.")
