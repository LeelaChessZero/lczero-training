import logging
import sys
from functools import partial
from typing import Callable, Dict, Generator, Tuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from google.protobuf import text_format
from jax import tree_util

from lczero_training.dataloader import DataLoader, make_dataloader
from lczero_training.model.loss_function import LczeroLoss
from lczero_training.model.model import LczeroModel
from lczero_training.training.optimizer import make_gradient_transformation
from lczero_training.training.state import TrainingState
from proto.root_config_pb2 import RootConfig

logger = logging.getLogger(__name__)


def from_dataloader(
    loader: DataLoader,
) -> Generator[Tuple[np.ndarray, ...], None, None]:
    while True:
        yield loader.get_next()


class Training:
    optimizer_tx: optax.GradientTransformation
    train_step: Callable[
        [optax.GradientTransformation, TrainingState, dict],
        Tuple[TrainingState, Tuple[jax.Array, Dict[str, jax.Array]]],
    ]

    def __init__(
        self,
        optimizer_tx: optax.GradientTransformation,
        graphdef: nnx.GraphDef,
        loss_fn: LczeroLoss,
    ):
        self.optimizer_tx = optimizer_tx

        @partial(nnx.jit, static_argnames=("optimizer_tx"))
        def _step(
            optimizer_tx: optax.GradientTransformation,
            state: TrainingState,
            batch: dict,
        ) -> Tuple[TrainingState, Tuple[jax.Array, Dict[str, jax.Array]]]:
            model = nnx.merge(graphdef, state.model_state)

            def loss_for_grad(
                model_arg: LczeroModel, batch_arg: dict
            ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
                return loss_fn(
                    model_arg,
                    inputs=batch_arg["inputs"],
                    value_targets=batch_arg["value_targets"],
                    policy_targets=batch_arg["policy_targets"],
                    movesleft_targets=batch_arg["movesleft_targets"],
                )

            loss_vfn = jax.vmap(
                loss_for_grad,
                in_axes=(None, 0),  # (model_arg, batch_arg)
                out_axes=0,
            )

            def mean_loss_for_grad(
                model_arg: LczeroModel, batch_arg: dict
            ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
                per_sample_data_loss, unweighted_losses = loss_vfn(
                    model_arg, batch_arg
                )
                mean_data_loss = jnp.mean(per_sample_data_loss)

                mean_unweighted = tree_util.tree_map(
                    jnp.mean, unweighted_losses
                )
                total_l2_loss = mean_unweighted["l2"]

                batch_size = batch_arg["inputs"].shape[0]
                mean_l2_loss = total_l2_loss / batch_size

                mean_loss = mean_data_loss + loss_fn.l2_weight * mean_l2_loss

                return mean_loss, unweighted_losses

            grad_fn = nnx.value_and_grad(mean_loss_for_grad, has_aux=True)
            (mean_loss, unweighted_losses), mean_grads = grad_fn(model, batch)

            assert state.opt_state is not None
            updates, new_opt_state = optimizer_tx.update(
                mean_grads, state.opt_state, state.model_state
            )
            new_model_state = optax.apply_updates(state.model_state, updates)

            new_train_state = state.replace(
                step=state.step + 1,
                model_state=new_model_state,
                opt_state=new_opt_state,
            )

            mean_unweighted = tree_util.tree_map(jnp.mean, unweighted_losses)
            return new_train_state, (mean_loss, mean_unweighted)

        self.train_step = cast(
            Callable[
                [optax.GradientTransformation, TrainingState, dict],
                Tuple[TrainingState, Tuple[jax.Array, Dict[str, jax.Array]]],
            ],
            _step,
        )

    def run(
        self,
        training_state: TrainingState,
        datagen: Generator[Tuple[np.ndarray, ...], None, None],
        num_steps: int,
    ) -> TrainingState:
        assert training_state.opt_state is not None
        for _ in range(num_steps):
            logger.info(f"Starting step {training_state.step}")
            batch = next(datagen)
            b_inputs, b_policy, b_values, _, b_movesleft = batch
            logger.info("Fetched batch from dataloader")
            training_state, (loss, unweighted_losses) = self.train_step(
                self.optimizer_tx,
                training_state,
                {
                    "inputs": b_inputs,
                    "value_targets": b_values,
                    "policy_targets": b_policy,
                    "movesleft_targets": b_movesleft,
                },
            )
            logger.info(
                f"Step {training_state.step}, Loss: {loss}, Unweighted losses:"
                f" {unweighted_losses}"
            )
        return training_state


def train(config_filename: str) -> None:
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
        0, args=ocp.args.StandardRestore(empty_state)
    )
    logger.info("Restored checkpoint")

    model, _ = nnx.split(
        LczeroModel(config=config.model, rngs=nnx.Rngs(params=42))
    )

    assert isinstance(training_state, TrainingState)
    optimizer_tx = make_gradient_transformation(config.training.optimizer)
    training = Training(
        optimizer_tx=optimizer_tx,
        graphdef=model,
        loss_fn=LczeroLoss(config=config.training.losses),
    )
    training.run(
        training_state, from_dataloader(make_dataloader(config.data_loader)), 30
    )
