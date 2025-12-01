"""Overfitting utility for quickly validating training setup."""

import csv
import logging
from contextlib import suppress
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from google.protobuf import text_format
from jax import tree_util

from lczero_training.dataloader import DataLoader, make_dataloader
from lczero_training.model.loss_function import LczeroLoss
from lczero_training.model.model import LczeroModel
from lczero_training.training.lr_schedule import make_lr_schedule
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


def _make_eval_step(graphdef: nnx.GraphDef, loss_fn: LczeroLoss) -> Any:
    @partial(nnx.jit, static_argnames=())
    def eval_step(model_state: nnx.State, batch: dict) -> tuple[jax.Array, Any]:
        model = nnx.merge(graphdef, model_state)

        def loss_for_batch(
            model_arg: LczeroModel, batch_arg: dict
        ) -> tuple[jax.Array, Any]:
            return loss_fn(
                model_arg,
                inputs=batch_arg["inputs"],
                value_targets=batch_arg["value_targets"],
                policy_targets=batch_arg["policy_targets"],
                movesleft_targets=batch_arg["movesleft_targets"],
            )

        loss_vfn = jax.vmap(loss_for_batch, in_axes=(None, 0), out_axes=0)
        per_sample_loss, unweighted_losses = loss_vfn(model, batch)
        mean_loss = jnp.mean(per_sample_loss)
        mean_unweighted = tree_util.tree_map(jnp.mean, unweighted_losses)
        return mean_loss, mean_unweighted

    return eval_step


def overfit(
    *,
    config_filename: str,
    num_steps: int,
    coin_flip: bool = False,
    csv_file: str | None = None,
) -> None:
    """Runs an overfitting loop to validate training."""

    if num_steps <= 0:
        raise ValueError("num_steps must be a positive integer")

    if jax.device_count() > 1:
        raise ValueError(
            f"Overfit utility does not support multi-GPU training. "
            f"Detected {jax.device_count()} devices. "
            f"Please set CUDA_VISIBLE_DEVICES to use only one GPU."
        )

    config = RootConfig()
    logger.info("Reading configuration from proto file")
    with open(config_filename, "r") as config_file:
        text_format.Parse(config_file.read(), config)

    logger.info("Creating data loader and fetching batches")
    loader = make_dataloader(config.data_loader)
    try:
        batch_a = loader.get_next()
        batch_b = loader.get_next() if coin_flip else None
    finally:
        _stop_loader(loader)

    prepared_batch_a = _prepare_batch(batch_a)
    prepared_batch_b = _prepare_batch(batch_b) if batch_b is not None else None

    logger.info("Creating training state from configuration")
    training_state = TrainingState.new_from_config(
        model_config=config.model,
        training_config=config.training,
    )

    graphdef, _ = nnx.split(
        LczeroModel(config=config.model, rngs=nnx.Rngs(params=42))
    )

    jit_state = training_state.jit_state
    lr_sched = make_lr_schedule(config.training.lr_schedule)
    optimizer_tx = make_gradient_transformation(
        config.training.optimizer,
        max_grad_norm=getattr(config.training, "max_grad_norm", 0.0),
        lr_schedule=lr_sched,
    )

    loss_fn = LczeroLoss(config=config.training.losses)
    training = Training(
        optimizer_tx=optimizer_tx,
        graphdef=graphdef,
        loss_fn=loss_fn,
    )
    eval_step = _make_eval_step(graphdef, loss_fn)

    csv_handle = None
    csv_writer: Any | None = None
    if csv_file is not None:
        logger.info("Writing overfit results to %s", csv_file)
        csv_handle = open(csv_file, "w", newline="")
        csv_writer = csv.writer(csv_handle)
        csv_writer.writerow(
            [
                "step",
                "train_batch",
                "train_loss",
                "train_unweighted",
                "eval_batch",
                "eval_loss",
                "eval_unweighted",
            ]
        )
        csv_handle.flush()

    def log_step(
        *,
        step_value: int,
        train_batch_name: str,
        train_loss: float,
        train_unweighted: Any,
        eval_batch_name: str | None,
        eval_loss: float | None,
        eval_unweighted: Any | None,
    ) -> None:
        if eval_batch_name is None or eval_loss is None:
            logger.info(
                "Step %d: batch=%s train_loss=%f, unweighted_losses=%s",
                step_value,
                train_batch_name,
                train_loss,
                train_unweighted,
            )
        else:
            logger.info(
                (
                    "Step %d: trained %s train_loss=%f, unweighted_losses=%s; "
                    "evaluated %s eval_loss=%f, eval_unweighted=%s"
                ),
                step_value,
                train_batch_name,
                train_loss,
                train_unweighted,
                eval_batch_name,
                eval_loss,
                eval_unweighted,
            )

        if csv_writer is not None and csv_handle is not None:
            csv_writer.writerow(
                [
                    step_value,
                    train_batch_name,
                    train_loss,
                    repr(train_unweighted),
                    eval_batch_name or "",
                    "" if eval_loss is None else eval_loss,
                    "" if eval_unweighted is None else repr(eval_unweighted),
                ]
            )
            csv_handle.flush()

    try:
        if coin_flip:
            if prepared_batch_b is None:
                raise RuntimeError(
                    "Coin flip mode requires two batches but only one was fetched"
                )

            logger.info(
                "Starting coin-flip overfit: %d steps on batch A then %d on batch B",
                num_steps,
                num_steps,
            )

            def run_phase(
                train_batch: dict,
                train_name: str,
                eval_batch: dict,
                eval_name: str,
            ) -> None:
                nonlocal jit_state
                for _ in range(num_steps):
                    jit_state, metrics = training.train_step(
                        optimizer_tx,
                        jit_state,
                        train_batch,
                    )
                    loss = metrics["loss"]
                    unweighted_losses = metrics["unweighted_losses"]
                    loss_value, unweighted_host = jax.device_get(
                        (loss, unweighted_losses)
                    )
                    loss_value = float(np.asarray(loss_value))
                    unweighted_host = tree_util.tree_map(
                        lambda x: float(np.asarray(x)), unweighted_host
                    )

                    eval_loss, eval_unweighted = eval_step(
                        jit_state.model_state, eval_batch
                    )
                    eval_loss, eval_unweighted = jax.device_get(
                        (eval_loss, eval_unweighted)
                    )
                    eval_loss_value = float(np.asarray(eval_loss))
                    eval_unweighted_host = tree_util.tree_map(
                        lambda x: float(np.asarray(x)), eval_unweighted
                    )

                    step_value = int(
                        np.asarray(jax.device_get(jit_state.step)).flat[0]
                    )
                    log_step(
                        step_value=step_value,
                        train_batch_name=train_name,
                        train_loss=loss_value,
                        train_unweighted=unweighted_host,
                        eval_batch_name=eval_name,
                        eval_loss=eval_loss_value,
                        eval_unweighted=eval_unweighted_host,
                    )

            run_phase(prepared_batch_a, "A", prepared_batch_b, "B")
            run_phase(prepared_batch_b, "B", prepared_batch_a, "A")
        else:
            logger.info("Starting overfit loop for %d steps", num_steps)
            for _ in range(num_steps):
                jit_state, metrics = training.train_step(
                    optimizer_tx,
                    jit_state,
                    prepared_batch_a,
                )
                loss = metrics["loss"]
                unweighted_losses = metrics["unweighted_losses"]
                loss_value, unweighted_host = jax.device_get(
                    (loss, unweighted_losses)
                )
                loss_value = float(np.asarray(loss_value))
                unweighted_host = tree_util.tree_map(
                    lambda x: float(np.asarray(x)), unweighted_host
                )
                step_value = int(
                    np.asarray(jax.device_get(jit_state.step)).flat[0]
                )
                log_step(
                    step_value=step_value,
                    train_batch_name="single",
                    train_loss=loss_value,
                    train_unweighted=unweighted_host,
                    eval_batch_name=None,
                    eval_loss=None,
                    eval_unweighted=None,
                )
    finally:
        if csv_handle is not None:
            csv_handle.close()
