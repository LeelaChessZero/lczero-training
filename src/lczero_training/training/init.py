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
from lczero_training.training.state import TrainingState
from proto import hlo_pb2, net_pb2
from proto.model_config_pb2 import ModelConfig
from proto.root_config_pb2 import RootConfig

logger = logging.getLogger(__name__)


def _load_lc0_model_state(
    path: str,
    expected_config: ModelConfig,
    compute_dtype: hlo_pb2.XlaShapeProto.Type,
) -> tuple[nnx.State, int]:
    """Load lc0 weights, validate config, return (model_state, training_steps)."""
    lc0_weights = net_pb2.Net()
    with gzip.open(path, "rb") as f:
        lc0_weights.ParseFromString(f.read())
    fix_older_weights_file(lc0_weights)

    leela_config = leela_to_modelconfig(
        lc0_weights, hlo_pb2.XlaShapeProto.F32, compute_dtype
    )
    if leela_config != expected_config:
        logger.error(
            "The provided lczero model configuration "
            "differs from the one in the config file."
        )
        logger.error(f"Config file model config: {expected_config}")
        logger.error(f"Leela model config: {leela_config}")
        sys.exit(1)

    import_options = LeelaImportOptions(
        weights_dtype=hlo_pb2.XlaShapeProto.F32, compute_dtype=compute_dtype
    )
    model_state = leela_to_jax(lc0_weights, import_options)
    return model_state, lc0_weights.training_params.training_steps


def init(
    config_filename: str,
    lczero_model: Optional[str],
    seed: int = 42,
    dry_run: bool = False,
    swa_initial_nets: int = 0,
    override_training_steps: Optional[int] = None,
    override: bool = False,
    from_checkpoint: Optional[str] = None,
    no_copy_swa: bool = False,
) -> None:
    """
    Initializes a new training run.
    """

    config = RootConfig()
    logger.info("Reading configuration from proto file")
    with open(config_filename, "r") as f:
        text_format.Parse(f.read(), config)

    if not dry_run:
        if config.training.checkpoint.path is None:
            logger.error("Checkpoint path must be set in the configuration.")
            sys.exit(1)

        if os.path.exists(config.training.checkpoint.path) and not override:
            logger.error(
                f"Checkpoint path {config.training.checkpoint.path} already exists."
            )
            sys.exit(1)

    logger.info("Creating initial training state from configuration")
    training_state = TrainingState.new_from_config(
        model_config=config.model,
        training_config=config.training,
    )

    if from_checkpoint is not None:
        logger.info(f"Loading from checkpoint: {from_checkpoint}")
        source_mgr = ocp.CheckpointManager(
            from_checkpoint,
            options=ocp.CheckpointManagerOptions(create=False),
        )
        training_state = source_mgr.restore(
            source_mgr.latest_step(),
            args=ocp.args.PyTreeRestore(training_state),
        )

    swa_enabled = config.training.HasField("swa")

    if lczero_model is None:
        if override_training_steps is not None:
            training_state = training_state.with_updated_step(
                override_training_steps
            )
        if swa_enabled and swa_initial_nets > 0:
            training_state = training_state.replace(
                jit_state=training_state.jit_state.replace(
                    num_averages=float(swa_initial_nets),
                )
            )
    else:
        logger.info(f"Loading lczero model: {lczero_model}")
        model_state, lc0_steps = _load_lc0_model_state(
            lczero_model, config.model, config.model.defaults.compute_dtype
        )
        step = override_training_steps or lc0_steps
        new_swa_state = (
            training_state.jit_state.swa_state
            if no_copy_swa
            else (model_state if swa_enabled else None)
        )
        training_state = training_state.replace(
            jit_state=training_state.jit_state.replace(
                model_state=model_state,
                swa_state=new_swa_state,
                num_averages=float(swa_initial_nets) if swa_enabled else 0.0,
            )
        ).with_updated_step(step)

    if dry_run:
        logger.info(
            f"Would save checkpoint to {config.training.checkpoint.path} "
            f"at step {training_state.jit_state.step}"
        )
    else:
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
