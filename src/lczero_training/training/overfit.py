"""Overfitting utility for quickly validating training setup."""

import logging
from contextlib import suppress

import jax
import jax.numpy as jnp
import jax.sharding as jshard
import numpy as np
from flax import nnx
from google.protobuf import text_format
from jax import tree_util
from jax.sharding import PartitionSpec as P

from lczero_training.dataloader import DataLoader, make_dataloader
from lczero_training.model.loss_function import LczeroLoss
from lczero_training.model.model import LczeroModel
from lczero_training.training.optimizer import make_gradient_transformation
from lczero_training.training.state import TrainingState
from lczero_training.training.training import Training
from proto.root_config_pb2 import RootConfig

logger = logging.getLogger(__name__)


def _stop_loader(loader: DataLoader) -> None:
    with suppress(Exception):
        loader.stop()


def _prepare_batch(batch: tuple) -> dict:
    inputs, policy, values, _, movesleft = batch
    return {
        "inputs": jnp.asarray(inputs),
        "value_targets": jnp.asarray(values),
        "policy_targets": jnp.asarray(policy),
        "movesleft_targets": jnp.asarray(movesleft),
    }


def overfit(*, config_filename: str, num_steps: int) -> None:
    """Runs an overfitting loop on a single batch to validate training."""

    if num_steps <= 0:
        raise ValueError("num_steps must be a positive integer")

    config = RootConfig()
    logger.info("Reading configuration from proto file")
    with open(config_filename, "r") as config_file:
        text_format.Parse(config_file.read(), config)

    logger.info("Creating data loader and fetching a single batch")
    loader = make_dataloader(config.data_loader)
    try:
        batch = loader.get_next()
    finally:
        _stop_loader(loader)

    prepared_batch = _prepare_batch(batch)

    logger.info("Creating training state from configuration")
    training_state = TrainingState.new_from_config(
        model_config=config.model,
        training_config=config.training,
    )

    graphdef, _ = nnx.split(
        LczeroModel(config=config.model, rngs=nnx.Rngs(params=42))
    )

    jit_state = training_state.jit_state
    if jax.device_count() > 1:
        mesh = jshard.Mesh(jax.devices(), axis_names=("batch",))
        replicated_sharding = jshard.NamedSharding(mesh, P())
        jit_state = jax.device_put(jit_state, replicated_sharding)

    optimizer_tx = make_gradient_transformation(
        config.training.optimizer,
        max_grad_norm=getattr(config.training, "max_grad_norm", 0.0),
    )

    training = Training(
        optimizer_tx=optimizer_tx,
        graphdef=graphdef,
        loss_fn=LczeroLoss(config=config.training.losses),
    )

    logger.info("Starting overfit loop for %d steps", num_steps)
    for _ in range(num_steps):
        jit_state, (loss, unweighted_losses) = training.train_step(
            optimizer_tx,
            jit_state,
            prepared_batch,
        )
        loss_value, unweighted_host = jax.device_get((loss, unweighted_losses))
        loss_value = float(np.asarray(loss_value))
        unweighted_host = tree_util.tree_map(
            lambda x: float(np.asarray(x)), unweighted_host
        )
        step_value = int(np.asarray(jax.device_get(jit_state.step)).flat[0])
        logger.info(
            "Step %d: loss=%f, unweighted_losses=%s",
            step_value,
            loss_value,
            unweighted_host,
        )
