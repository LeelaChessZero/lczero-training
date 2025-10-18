import csv
import logging
import sys
from contextlib import nullcontext
from functools import partial
from typing import Callable, Dict, List, Tuple, cast

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
            f"Unsupported optimizer type: {opt_config.WhichOneof('optimizer_type')}"
        )

    if max_grad_norm > 0:
        tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), tx)

    if training_state.jit_state.opt_state is None:
        raise ValueError("Optimizer state must be available in the checkpoint.")

    return tx


def _make_eval_step(
    graphdef: nnx.GraphDef, loss_fn: LczeroLoss
) -> Callable[[nnx.State, Dict[str, jax.Array]], jax.Array]:
    @partial(nnx.jit, static_argnames=())
    def eval_step(
        model_state: nnx.State, batch: Dict[str, jax.Array]
    ) -> jax.Array:
        model = nnx.merge(graphdef, model_state)

        def calculate_loss(
            model_arg: LczeroModel, batch_arg: Dict[str, jax.Array]
        ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
            return loss_fn(model_arg, **batch_arg)

        loss_vfn = jax.vmap(calculate_loss, in_axes=(None, 0), out_axes=0)
        per_sample_data_loss, _ = loss_vfn(model, batch)
        return jnp.mean(per_sample_data_loss)

    return cast(
        Callable[[nnx.State, Dict[str, jax.Array]], jax.Array], eval_step
    )


def _plot_results(results: List[Tuple[float, float]], plot_output: str) -> None:
    logger.info("Saving plot to %s", plot_output)
    lrs, losses = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses, marker="o")
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")
    plt.grid(True, which="both", linestyle="--")
    plt.tight_layout()
    plt.savefig(plot_output, dpi=150)
    plt.close()


def tune_lr(
    *,
    config_filename: str,
    start_lr: float,
    num_steps: int,
    multiplier: float = 1.01,
    warmup_steps: int = 0,
    warmup_lr: float | None = None,
    csv_output: str | None = None,
    plot_output: str | None = None,
    num_test_batches: int = 1,
    fixed_validation_batch: bool = False,
) -> None:
    if num_steps <= 0 or start_lr <= 0 or multiplier <= 0 or warmup_steps < 0:
        logger.error(
            "num_steps, start_lr, and multiplier must be positive, "
            "and warmup_steps non-negative."
        )
        sys.exit(1)
    if warmup_steps > 0 and (warmup_lr is None or warmup_lr <= 0):
        logger.error("warmup_lr must be a positive value when warmup_steps > 0")
        sys.exit(1)

    config = RootConfig()
    logger.info("Reading configuration from %s", config_filename)
    with open(config_filename, "r") as f:
        text_format.Parse(f.read(), config)

    if not config.training.checkpoint.path:
        logger.error("Checkpoint path must be set in the configuration.")
        sys.exit(1)

    checkpoint_mgr = ocp.CheckpointManager(
        config.training.checkpoint.path,
        options=ocp.CheckpointManagerOptions(create=False),
    )

    logger.info("Creating state from configuration")
    empty_state = TrainingState.new_from_config(
        model_config=config.model, training_config=config.training
    )

    logger.info("Restoring checkpoint from %s", config.training.checkpoint.path)
    training_state = checkpoint_mgr.restore(
        checkpoint_mgr.latest_step(), args=ocp.args.PyTreeRestore(empty_state)
    )
    if training_state is None:
        logger.error("No checkpoint found.")
        sys.exit(1)
    logger.info("Restored checkpoint at step %d", training_state.jit_state.step)

    assert isinstance(training_state, TrainingState)

    model, _ = nnx.split(
        LczeroModel(config=config.model, rngs=nnx.Rngs(params=42))
    )

    datagen = from_dataloader(make_dataloader(config.data_loader))

    # Prepare fixed validation batches only if requested.
    if fixed_validation_batch:
        logger.info("Fetching %d validation batches", num_test_batches)
        validation_batches = [
            tree_util.tree_map(jnp.asarray, _prepare_batch(next(datagen)))
            for _ in range(num_test_batches)
        ]
    else:
        validation_batches = []

    loss_fn = LczeroLoss(config=config.training.losses)
    eval_step = _make_eval_step(model, loss_fn)

    def avg_val_loss() -> float:
        assert fixed_validation_batch
        total_loss = 0.0
        for vb in validation_batches:
            total_loss += float(
                eval_step(training_state.jit_state.model_state, vb)
            )
        return total_loss / float(num_test_batches)

    def train_one_step(
        training: Training, tx: optax.GradientTransformation
    ) -> float:
        nonlocal training_state
        batch = tree_util.tree_map(jnp.asarray, _prepare_batch(next(datagen)))
        new_jit_state, metrics = training.train_step(
            tx, training_state.jit_state, batch
        )
        training_state = training_state.replace(jit_state=new_jit_state)
        return float(metrics["loss"])  # training batch loss

    def run_phase(
        *,
        steps: int,
        schedule: optax.Schedule,
        lr_at: Callable[[int], float],
        label: str,
        on_result: Callable[[float, float, float | None], None],
    ) -> None:
        start_step = training_state.jit_state.step

        def offset_schedule(count: jax.Array) -> jax.Array:
            return schedule(count - start_step)

        tx = _make_optimizer_with_schedule(
            training_state, config, offset_schedule
        )
        training = Training(optimizer_tx=tx, graphdef=model, loss_fn=loss_fn)
        for i in range(steps):
            current_lr = lr_at(i)
            logger.info(
                "%s step %d/%d at lr %.8f", label, i + 1, steps, current_lr
            )
            train_loss = train_one_step(training, tx)
            val_loss = avg_val_loss() if fixed_validation_batch else None
            on_result(current_lr, train_loss, val_loss)
            if fixed_validation_batch:
                logger.info(
                    "%s at lr %.8f: train=%.6f, val=%.6f",
                    label,
                    current_lr,
                    train_loss,
                    cast(float, val_loss),
                )
            else:
                logger.info(
                    "%s train loss at lr %.8f: %.6f",
                    label,
                    current_lr,
                    train_loss,
                )

    results: List[Tuple[float, float]] = []
    with (
        open(csv_output, "w", newline="") if csv_output else nullcontext()
    ) as csv_file:
        writer = csv.writer(csv_file) if csv_file else None
        if writer:
            if fixed_validation_batch:
                writer.writerow(["lr", "train_loss", "val_loss"])
            else:
                writer.writerow(["lr", "train_loss"])

        def on_result(
            lr: float, train_loss: float, val_loss: float | None
        ) -> None:
            results.append((lr, train_loss))
            if writer and csv_file:
                if fixed_validation_batch:
                    writer.writerow([lr, train_loss, val_loss])
                else:
                    writer.writerow([lr, train_loss])
                csv_file.flush()

        phases = []
        if warmup_steps > 0 and warmup_lr is not None:
            phases.append(
                {
                    "label": "Warmup",
                    "steps": warmup_steps,
                    "schedule": optax.constant_schedule(warmup_lr),
                    "lr_at": lambda _: float(warmup_lr),
                }
            )
        phases.append(
            {
                "label": "Sweep",
                "steps": num_steps,
                "schedule": optax.exponential_decay(
                    start_lr, transition_steps=1, decay_rate=multiplier
                ),
                "lr_at": lambda i: start_lr * (multiplier**i),
            }
        )

        for phase_params in phases:
            run_phase(**phase_params, on_result=on_result)

    if plot_output:
        _plot_results(results, plot_output)
