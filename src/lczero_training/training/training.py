import dataclasses
import logging
from functools import partial
from typing import Any, Callable, Dict, Generator, Optional, Tuple, cast

import jax
import jax.numpy as jnp
import jax.sharding as jshard
import numpy as np
import optax
from flax import nnx
from jax import tree_util
from jax.sharding import PartitionSpec as P

from lczero_training.dataloader import DataLoader
from lczero_training.model.loss_function import LczeroLoss
from lczero_training.model.model import LczeroModel
from lczero_training.training.state import (
    JitTrainingState,
    TrainingBatch,
    TrainingSample,
)
from proto import training_config_pb2 as training_config_pb2

MetricsDict = Dict[str, Any]


@dataclasses.dataclass
class StepHookData:
    """Data passed to the step hook callback during training."""

    global_step: int
    local_step: int
    steps_per_epoch: int
    metrics: MetricsDict
    jit_state: JitTrainingState


StepHook = Callable[[StepHookData], None]

logger = logging.getLogger(__name__)


def from_dataloader(
    loader: DataLoader,
) -> Generator[tuple[np.ndarray, ...], None, None]:
    while True:
        yield loader.get_next()


class Training:
    optimizer_tx: optax.GradientTransformation
    train_step: Callable[
        [optax.GradientTransformation, JitTrainingState, TrainingBatch],
        Tuple[JitTrainingState, MetricsDict],
    ]
    _swa_config: Optional[training_config_pb2.SWAConfig]
    _dp_sharding: Optional[jshard.NamedSharding]

    def __init__(
        self,
        optimizer_tx: optax.GradientTransformation,
        graphdef: nnx.GraphDef,
        loss_fn: LczeroLoss,
        swa_config: Optional[training_config_pb2.SWAConfig] = None,
    ):
        self.optimizer_tx = optimizer_tx
        self._swa_config = swa_config
        self._dp_sharding = None

        jit_kwargs: Dict[str, Any] = {"static_argnames": ("optimizer_tx",)}
        if jax.device_count() > 1:
            num_devices = jax.device_count()
            logger.info(
                f"Multi-GPU training enabled: {num_devices} devices detected"
            )
            mesh = jshard.Mesh(jax.devices(), axis_names=("batch",))
            replicated = jshard.NamedSharding(mesh, P())
            dp_sharding = jshard.NamedSharding(mesh, P("batch"))
            self._dp_sharding = dp_sharding

            batch_sharding = TrainingBatch(
                inputs=dp_sharding,
                probabilities=dp_sharding,
                values=dp_sharding,
            )
            in_shardings = (replicated, batch_sharding)
            out_shardings = replicated

            jit_kwargs["in_shardings"] = in_shardings
            jit_kwargs["out_shardings"] = out_shardings

        @partial(jax.jit, **jit_kwargs)
        def _step(
            optimizer_tx: optax.GradientTransformation,
            jit_state: JitTrainingState,
            batch: TrainingBatch,
        ) -> Tuple[JitTrainingState, MetricsDict]:
            model = nnx.merge(graphdef, jit_state.model_state)

            def loss_for_grad(
                model_arg: LczeroModel, sample_arg: TrainingSample
            ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
                return loss_fn(model_arg, sample_arg)

            loss_vfn = jax.vmap(
                loss_for_grad,
                in_axes=(None, 0),  # (model_arg, sample_arg)
                out_axes=0,
            )

            def mean_loss_for_grad(
                model_arg: LczeroModel, batch_arg: TrainingBatch
            ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
                # vmap automatically distributes TrainingBatch over batch dimension,
                # calling loss_for_grad with TrainingSample (single samples).
                per_sample_data_loss, unweighted_losses = loss_vfn(
                    model_arg,
                    batch_arg,  # type: ignore[arg-type]
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
                [optax.GradientTransformation, JitTrainingState, TrainingBatch],
                Tuple[JitTrainingState, MetricsDict],
            ],
            _step,
        )

    @staticmethod
    @jax.jit
    def _swa_tree_map(
        alpha: jax.Array,
        beta: jax.Array,
        swa_state: nnx.State,
        model_state: nnx.State,
    ) -> nnx.State:
        return tree_util.tree_map(
            lambda a, b: alpha * a + beta * b, swa_state, model_state
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
        new_swa_state = self._swa_tree_map(
            jnp.array(alpha),
            jnp.array(beta),
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
        if self._swa_config is None:
            return jit_state
        period_steps = self._swa_config.period_steps
        assert period_steps > 0
        if steps_completed % period_steps == 0:
            return self.update_swa(jit_state, 1.0)
        if steps_completed == total_steps:
            remainder = total_steps % period_steps
            return self.update_swa(jit_state, remainder / period_steps)
        return jit_state

    def _validate_and_prepare_batch(
        self, tensor_tuple: tuple[np.ndarray, ...]
    ) -> TrainingBatch:
        logger.info("Fetched batch from dataloader")

        # Convert tuple to TrainingBatch
        batch = TrainingBatch.from_tuple(tensor_tuple)

        # Ensure batch.inputs is jax.Array for shape access
        assert isinstance(batch.inputs, jax.Array)
        batch_size = batch.inputs.shape[0]
        if self._dp_sharding is not None:
            num_devices = jax.device_count()
            if batch_size % num_devices != 0:
                raise ValueError(
                    f"Batch size {batch_size} must be divisible by device "
                    f"count {num_devices} for multi-GPU training. "
                    f"Per-device batch size would be "
                    f"{batch_size / num_devices:.2f}"
                )
            per_device_batch_size = batch_size // num_devices
            logger.info(
                f"Multi-GPU batch: {batch_size} total "
                f"({per_device_batch_size} per device)"
            )

        if self._dp_sharding is not None:
            batch = jax.device_put(batch, self._dp_sharding)

        return batch

    def _log_step_metrics(
        self,
        step_value: int,
        local_step: int,
        num_steps: int,
        metrics: MetricsDict,
    ) -> None:
        loss = float(metrics["loss"])
        unweighted_losses = {
            k: float(v) for k, v in metrics["unweighted_losses"].items()
        }
        grad_norm = float(metrics["grad_norm"])
        logger.info(
            f"Step {step_value} ({local_step}/{num_steps}), Loss: {loss}, "
            f"Unweighted losses: {unweighted_losses}, Grad norm: {grad_norm}"
        )

    def _execute_step_hook(
        self,
        step_hook: Optional[StepHook],
        step_value: int,
        local_step: int,
        num_steps: int,
        metrics: MetricsDict,
        jit_state: JitTrainingState,
    ) -> None:
        if step_hook is None:
            return
        hook_data = StepHookData(
            global_step=step_value,
            local_step=local_step,
            steps_per_epoch=num_steps,
            metrics=metrics,
            jit_state=jit_state,
        )
        step_hook(hook_data)

    def run(
        self,
        jit_state: JitTrainingState,
        datagen: Generator[tuple[np.ndarray, ...], None, None],
        num_steps: int,
        step_hook: Optional[StepHook] = None,
        memory_profile_dir: Optional[str] = None,
    ) -> JitTrainingState:
        assert jit_state.opt_state is not None
        if self._dp_sharding is not None:
            replicated = jshard.NamedSharding(self._dp_sharding.mesh, P())
            jit_state = jax.device_put(jit_state, replicated)
        for local_step in range(num_steps):
            logger.info(f"Starting step {jit_state.step}")
            if memory_profile_dir is not None:
                jax.profiler.save_device_memory_profile(
                    f"{memory_profile_dir}/before_{int(jit_state.step)}.prof"
                )
            batch = self._validate_and_prepare_batch(next(datagen))
            jit_state, metrics = self.train_step(
                self.optimizer_tx, jit_state, batch
            )
            step_value = int(
                np.asarray(jax.device_get(jit_state.step)).reshape(())
            )
            jit_state = self.maybe_update_swa(
                jit_state, local_step + 1, num_steps
            )
            self._execute_step_hook(
                step_hook, step_value, local_step, num_steps, metrics, jit_state
            )
            self._log_step_metrics(step_value, local_step, num_steps, metrics)
        return jit_state
