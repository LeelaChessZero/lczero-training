import csv
import logging
import sys
from functools import partial
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import orbax.checkpoint as ocp
from flax import nnx
from google.protobuf import text_format
from jax import tree_util

from lczero_training.dataloader import make_dataloader
from lczero_training.model.loss_function import LczeroLoss
from lczero_training.model.model import LczeroModel
from lczero_training.training.state import TrainingState
from proto.root_config_pb2 import RootConfig

from .training import Training, from_dataloader

logger = logging.getLogger(__name__)


def _prepare_batch(batch: Tuple) -> Dict[str, jax.Array]:
    inputs, policy, values, _, movesleft = batch
    return {
        "inputs": inputs,
        "value_targets": values,
        "policy_targets": policy,
        "movesleft_targets": movesleft,
    }


def _make_geometric_schedule(
    start_lr: float, multiplier: float
) -> optax.Schedule:
    start = jnp.asarray(start_lr, dtype=jnp.float32)
    mult = jnp.asarray(multiplier, dtype=jnp.float32)

    def schedule(count: jax.Array) -> jax.Array:
        step = jnp.asarray(count, dtype=jnp.float32)
        return start * jnp.power(mult, step)

    return schedule


def _make_optimizer_with_schedule(
    training_state: TrainingState,
    config: RootConfig,
    schedule: optax.Schedule,
) -> optax.GradientTransformation:
    max_grad_norm = getattr(config.training, "max_grad_norm", 0.0)
    opt_config = config.training.optimizer

    if opt_config.HasField("nadamw"):
        conf = opt_config.nadamw
        tx: optax.GradientTransformation = optax.nadamw(
            schedule,
            b1=conf.beta_1,
            b2=conf.beta_2,
            eps=conf.epsilon,
            weight_decay=conf.weight_decay,
        )
    else:
        raise ValueError(
            "Unsupported optimizer type: {}".format(
                opt_config.WhichOneof("optimizer_type")
            )
        )

    if max_grad_norm is not None and max_grad_norm > 0:
        tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), tx)

    # The restored training state already contains optimizer state; ensure it matches.
    if training_state.jit_state.opt_state is None:
        raise ValueError("Optimizer state must be available in the checkpoint.")

    return tx


def _make_eval_step(graphdef: nnx.GraphDef, loss_fn: LczeroLoss) -> Any:
    @partial(nnx.jit, static_argnames=())
    def eval_step(
        model_state: nnx.State, batch: Dict[str, jax.Array]
    ) -> jax.Array:
        model = nnx.merge(graphdef, model_state)

        def loss_for_grad(
            model_arg: LczeroModel, batch_arg: Dict[str, jax.Array]
        ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
            return loss_fn(
                model_arg,
                inputs=batch_arg["inputs"],
                value_targets=batch_arg["value_targets"],
                policy_targets=batch_arg["policy_targets"],
                movesleft_targets=batch_arg["movesleft_targets"],
            )

        loss_vfn = jax.vmap(loss_for_grad, in_axes=(None, 0), out_axes=0)
        per_sample_data_loss, _ = loss_vfn(model, batch)
        return jnp.mean(per_sample_data_loss)

    return eval_step


def tune_lr(
    *,
    config_filename: str,
    start_lr: float,
    num_steps: int,
    multiplier: float = 1.01,
    csv_output: str | None = None,
    plot_output: str | None = None,
    num_test_batches: int = 1,
) -> None:
    if num_steps <= 0:
        logger.error("num_steps must be a positive integer")
        sys.exit(1)

    if start_lr <= 0:
        logger.error("start_lr must be positive")
        sys.exit(1)

    if multiplier <= 0:
        logger.error("multiplier must be positive")
        sys.exit(1)

    config = RootConfig()
    logger.info("Reading configuration from proto file")
    with open(config_filename, "r") as f:
        text_format.Parse(f.read(), config)

    if config.training.checkpoint.path is None:
        logger.error("Checkpoint path must be set in the configuration.")
        sys.exit(1)

    checkpoint_mgr = ocp.CheckpointManager(
        config.training.checkpoint.path,
        options=ocp.CheckpointManagerOptions(create=False),
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
    if training_state is None:
        logger.error(
            "No checkpoint found at %s", config.training.checkpoint.path
        )
        sys.exit(1)
    logger.info("Restored checkpoint")

    assert isinstance(training_state, TrainingState)

    model, _ = nnx.split(
        LczeroModel(config=config.model, rngs=nnx.Rngs(params=42))
    )

    datagen = from_dataloader(make_dataloader(config.data_loader))
    logger.info("Fetching %d validation batches", num_test_batches)
    validation_batches = []
    for _ in range(num_test_batches):
        batch = _prepare_batch(next(datagen))
        batch = tree_util.tree_map(lambda x: jnp.asarray(x), batch)
        validation_batches.append(batch)

    schedule = _make_geometric_schedule(start_lr, multiplier)

    # The restored optimizer state has a step count from previous training.
    # To start the geometric LR schedule from the beginning without resetting the
    # whole optimizer state, we offset the step count passed to the schedule.
    # The restored training state has a step count from previous training.
    # To start the geometric LR schedule from the beginning without resetting the
    # whole optimizer state, we offset the step count passed to the schedule.
    initial_step = training_state.jit_state.step

    def offset_schedule(count: jax.Array) -> jax.Array:
        return schedule(count - initial_step)

    optimizer_tx = _make_optimizer_with_schedule(
        training_state, config, offset_schedule
    )

    loss_fn = LczeroLoss(config=config.training.losses)
    training = Training(
        optimizer_tx=optimizer_tx,
        graphdef=model,
        loss_fn=loss_fn,
    )
    eval_step = _make_eval_step(model, loss_fn)

    results: List[Tuple[float, float]] = []

    csvfile = None
    csv_writer: Any | None = None
    if csv_output:
        logger.info("Writing learning-rate sweep results to %s", csv_output)
        csvfile = open(csv_output, "w", newline="")
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["lr", "loss"])
        csvfile.flush()

    try:
        for step_idx in range(num_steps):
            current_lr = start_lr * (multiplier**step_idx)
            logger.info(
                "Running step %d/%d with learning rate %.8f",
                step_idx + 1,
                num_steps,
                current_lr,
            )

            train_batch = _prepare_batch(next(datagen))
            train_batch = tree_util.tree_map(
                lambda x: jnp.asarray(x), train_batch
            )
            new_jit_state, _ = training.train_step(
                optimizer_tx,
                training_state.jit_state,
                train_batch,
            )
            training_state = training_state.replace(jit_state=new_jit_state)

            total_val_loss = 0.0
            for val_batch in validation_batches:
                total_val_loss += eval_step(
                    training_state.jit_state.model_state, val_batch
                )
            val_loss = total_val_loss / num_test_batches
            results.append((current_lr, float(val_loss)))
            if csv_writer is not None and csvfile is not None:
                csv_writer.writerow([current_lr, float(val_loss)])
                csvfile.flush()
            logger.info(
                "Validation loss at lr %.8f: %.6f", current_lr, float(val_loss)
            )
    finally:
        if csvfile is not None:
            csvfile.close()

    if plot_output:
        logger.info("Saving plot to %s", plot_output)
        lrs, losses = zip(*results)
        plt.figure()
        plt.plot(lrs, losses, marker="o")
        plt.xlabel("Learning rate")
        plt.ylabel("Validation loss")
        plt.title("Learning rate tuning")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_output, dpi=150)
        plt.close()
