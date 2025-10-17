import datetime
import gzip
import logging
import os
import sys
from functools import partial
from typing import Any, Callable, Dict, Generator, Optional, Tuple, cast

import jax
import jax.numpy as jnp
import jax.sharding as jshard
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from google.protobuf import text_format
from jax import tree_util
from jax.sharding import PartitionSpec as P

from lczero_training.convert.jax_to_leela import (
    LeelaExportOptions,
    jax_to_leela,
)
from lczero_training.dataloader import DataLoader, make_dataloader
from lczero_training.model.loss_function import LczeroLoss
from lczero_training.model.model import LczeroModel
from lczero_training.training.lr_schedule import make_lr_schedule
from lczero_training.training.optimizer import make_gradient_transformation
from lczero_training.training.state import JitTrainingState, TrainingState
from proto import training_config_pb2 as training_config_pb2
from proto.root_config_pb2 import RootConfig

MetricsDict = Dict[str, Any]
MetricsHook = Callable[[int, MetricsDict], None]

logger = logging.getLogger(__name__)


def from_dataloader(
    loader: DataLoader,
) -> Generator[Tuple[np.ndarray, ...], None, None]:
    while True:
        yield loader.get_next()


class Training:
    optimizer_tx: optax.GradientTransformation
    train_step: Callable[
        [optax.GradientTransformation, JitTrainingState, dict],
        Tuple[JitTrainingState, MetricsDict],
    ]
    _swa_config: Optional[training_config_pb2.SWAConfig]

    def __init__(
        self,
        optimizer_tx: optax.GradientTransformation,
        graphdef: nnx.GraphDef,
        loss_fn: LczeroLoss,
        swa_config: Optional[training_config_pb2.SWAConfig] = None,
    ):
        self.optimizer_tx = optimizer_tx
        self._swa_config = swa_config

        jit_kwargs: Dict[str, object] = {"static_argnames": ("optimizer_tx",)}
        if jax.device_count() > 1:
            mesh = jshard.Mesh(jax.devices(), axis_names=("batch",))
            replicated = jshard.NamedSharding(mesh, P())
            dp_sharding = jshard.NamedSharding(mesh, P("batch"))

            batch_sharding = {
                "inputs": dp_sharding,
                "value_targets": dp_sharding,
                "policy_targets": dp_sharding,
                "movesleft_targets": dp_sharding,
            }
            in_shardings = (replicated, batch_sharding)
            out_shardings = replicated

            jit_kwargs["in_shardings"] = in_shardings
            jit_kwargs["out_shardings"] = out_shardings

        @partial(nnx.jit, **jit_kwargs)
        def _step(
            optimizer_tx: optax.GradientTransformation,
            jit_state: JitTrainingState,
            batch: dict,
        ) -> Tuple[JitTrainingState, MetricsDict]:
            model = nnx.merge(graphdef, jit_state.model_state)

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
                mean_loss = jnp.mean(per_sample_data_loss)
                return mean_loss, unweighted_losses

            grad_fn = nnx.value_and_grad(mean_loss_for_grad, has_aux=True)
            (mean_loss, unweighted_losses), mean_grads = grad_fn(model, batch)
            grad_norm = optax.global_norm(mean_grads)

            assert jit_state.opt_state is not None
            updates, new_opt_state = optimizer_tx.update(
                mean_grads, jit_state.opt_state, jit_state.model_state
            )
            new_model_state = optax.apply_updates(
                jit_state.model_state, updates
            )

            new_jit_state = jit_state.replace(
                step=jit_state.step + 1,
                model_state=new_model_state,
                opt_state=new_opt_state,
            )

            mean_unweighted = tree_util.tree_map(jnp.mean, unweighted_losses)
            metrics: MetricsDict = {
                "loss": mean_loss,
                "unweighted_losses": mean_unweighted,
                "grad_norm": grad_norm,
            }
            return new_jit_state, metrics

        self.train_step = cast(
            Callable[
                [optax.GradientTransformation, JitTrainingState, dict],
                Tuple[JitTrainingState, MetricsDict],
            ],
            _step,
        )

    def update_swa(
        self, jit_state: JitTrainingState, weight: float
    ) -> JitTrainingState:
        """Update SWA using the provided weight for the current model.

        Assumes `jit_state.swa_state` is initialized and `_swa_config` present.
        """
        logger.info(
            "Updating SWA model, weight=%f, num_averages=%f",
            weight,
            jit_state.num_averages,
        )
        assert self._swa_config is not None
        assert jit_state.swa_state is not None
        assert weight > 0.0
        max_num_averages = self._swa_config.num_averages
        denom = jit_state.num_averages + weight
        alpha = jit_state.num_averages / denom
        beta = weight / denom
        new_swa_state = tree_util.tree_map(
            lambda a, b: alpha * a + beta * b,
            jit_state.swa_state,
            jit_state.model_state,
        )
        new_num_averages = min(
            max_num_averages, jit_state.num_averages + weight
        )
        return jit_state.replace(
            swa_state=new_swa_state, num_averages=new_num_averages
        )

    def maybe_update_swa(
        self,
        jit_state: JitTrainingState,
        steps_completed: int,
        total_steps: int,
    ) -> JitTrainingState:
        """Optionally update SWA based on configured schedule and epoch progress.

        Returns the original jit_state when no update is scheduled.
        """
        assert self._swa_config is not None
        period_steps = self._swa_config.period_steps
        assert period_steps > 0
        if steps_completed % period_steps == 0:
            return self.update_swa(jit_state, 1.0)
        if steps_completed == total_steps:
            remainder = total_steps % period_steps
            return self.update_swa(jit_state, remainder / period_steps)
        return jit_state

    def run(
        self,
        jit_state: JitTrainingState,
        datagen: Generator[Tuple[np.ndarray, ...], None, None],
        num_steps: int,
        metrics_hook: Optional[MetricsHook] = None,
    ) -> JitTrainingState:
        assert jit_state.opt_state is not None
        for local_step in range(num_steps):
            logger.info(f"Starting step {jit_state.step}")
            batch = next(datagen)
            b_inputs, b_policy, b_values, _, b_movesleft = batch
            logger.info("Fetched batch from dataloader")
            jit_state, metrics = self.train_step(
                self.optimizer_tx,
                jit_state,
                {
                    "inputs": b_inputs,
                    "value_targets": b_values,
                    "policy_targets": b_policy,
                    "movesleft_targets": b_movesleft,
                },
            )
            step_value = int(
                np.asarray(jax.device_get(jit_state.step)).reshape(())
            )
            jit_state = self.maybe_update_swa(
                jit_state, local_step + 1, num_steps
            )
            if metrics_hook is not None:
                metrics_hook(step_value, metrics)
            loss = metrics["loss"]
            unweighted_losses = metrics["unweighted_losses"]
            grad_norm = metrics["grad_norm"]
            logger.info(
                f"Step {step_value} ({local_step}/{num_steps}), Loss: {loss}, "
                f"Unweighted losses: {unweighted_losses}, Grad norm: {grad_norm}"
                f" {unweighted_losses}, Grad norm: {grad_norm}"
            )
        return jit_state


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
        None, args=ocp.args.PyTreeRestore(empty_state)
    )
    logger.info("Restored checkpoint")

    model, _ = nnx.split(
        LczeroModel(config=config.model, rngs=nnx.Rngs(params=42))
    )

    assert isinstance(training_state, TrainingState)

    jit_state = training_state.jit_state
    if jax.device_count() > 1:
        mesh = jshard.Mesh(jax.devices(), axis_names=("batch",))
        replicated_sharding = jshard.NamedSharding(mesh, P())
        jit_state = jax.device_put(jit_state, replicated_sharding)

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

    if config.export.HasField("path"):
        date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        export_filename = os.path.join(
            config.export.path,
            f"lc0-{date_str}-{new_state.step:08d}.pb.gz",
        )

        logging.info(f"Exporting model to {export_filename}")

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
        logging.info(f"Writing model to {export_filename}")
        os.makedirs(config.export.path, exist_ok=True)
        with gzip.open(export_filename, "wb") as f:
            f.write(net.SerializeToString())
        logging.info(f"Finished writing model to {export_filename}")
