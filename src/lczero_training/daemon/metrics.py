"""Metrics collection and logging for training daemon."""

import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from lczero_training._lczero_training import DataLoader
from lczero_training.model.loss_function import LczeroLoss
from lczero_training.model.model import LczeroModel
from lczero_training.training.state import JitTrainingState
from lczero_training.training.tensorboard import TensorboardLogger
from lczero_training.training.training import StepHookData
from proto.metrics_config_pb2 import MetricConfig, MetricsConfig

logger = logging.getLogger(__name__)

Batch = Tuple[np.ndarray, ...]


def load_batch_from_npz(npz_filename: str) -> Batch:
    """Load a batch from an NPZ file.

    Args:
        npz_filename: Path to the NPZ file.

    Returns:
        Batch tuple of (inputs, policy, values, _, movesleft).

    Raises:
        ValueError: If the NPZ file doesn't contain exactly one batch.
    """
    with np.load(npz_filename) as npz_file:
        batches = npz_file["batches"]
        if batches.size != 1:
            raise ValueError(
                f"Expected 1 batch in npz '{npz_filename}', got {batches.size}"
            )
        return batches[0]


class _Metric(ABC):
    """Base class for individual metric tracking."""

    def __init__(self, config: MetricConfig, logger: TensorboardLogger):
        self.config = config
        self.logger = logger

    def should_log(
        self, global_step: int, local_step: int, steps_per_epoch: int
    ) -> bool:
        """Check if it's time to log this metric."""
        if self.config.after_epoch and local_step + 1 == steps_per_epoch:
            return True
        step = global_step if self.config.use_global_steps else local_step
        return (step + 1) % self.config.period == 0

    @abstractmethod
    def log(self, hook_data: StepHookData, graphdef: nnx.GraphDef) -> None:
        """Log the metric for the current step."""


class _TrainingBatchMetric(_Metric):
    """Metric that logs training batch data."""

    def __init__(self, config: MetricConfig, logger: TensorboardLogger):
        super().__init__(config, logger)
        if config.use_swa_model:
            raise ValueError(
                f"Metric '{config.name}': Cannot use SWA model for "
                "training_batch metrics"
            )

    def log(self, hook_data: StepHookData, graphdef: nnx.GraphDef) -> None:
        self.logger.log(hook_data.global_step, hook_data.metrics)


class _EvaluatingMetric(_Metric, ABC):
    """Base class for metrics that evaluate loss on data."""

    def __init__(
        self,
        config: MetricConfig,
        logger: TensorboardLogger,
        loss_fn: Optional[LczeroLoss],
    ):
        super().__init__(config, logger)
        if not loss_fn:
            raise ValueError(f"Metric '{config.name}': Loss function required")
        self.loss_fn = loss_fn

    @abstractmethod
    def get_batch(self) -> Batch:
        """Get the batch data to evaluate."""

    def log(self, hook_data: StepHookData, graphdef: nnx.GraphDef) -> None:
        batch = self.get_batch()
        metrics = self._evaluate(batch, hook_data.jit_state, graphdef)
        self.logger.log(hook_data.global_step, metrics)

    def _evaluate(
        self, batch: Batch, jit_state: JitTrainingState, graphdef: nnx.GraphDef
    ) -> Dict[str, jax.Array]:
        """Evaluate loss function on a batch of data."""
        return evaluate_batch(
            batch, jit_state, graphdef, self.loss_fn, self.config.use_swa_model
        )


def evaluate_batch(
    batch: Batch,
    jit_state: JitTrainingState,
    graphdef: nnx.GraphDef,
    loss_fn: LczeroLoss,
    use_swa_model: bool = False,
) -> Dict[str, jax.Array]:
    """Evaluate loss function on a batch of data.

    Args:
        batch: Tuple of (inputs, policy, values, _, movesleft).
        jit_state: JIT training state containing model and optimizer state.
        graphdef: Graph definition of the model.
        loss_fn: Loss function to evaluate.
        use_swa_model: If True, use SWA model state instead of regular model.

    Returns:
        Dictionary of metrics with loss and unweighted losses.
    """
    b_inputs, b_policy, b_values, _, b_movesleft = batch

    model_state = (
        jit_state.swa_state if use_swa_model else jit_state.model_state
    )
    if use_swa_model and model_state is None:
        raise RuntimeError("SWA state not available")

    model = nnx.merge(graphdef, model_state)

    def loss_fn_inner(
        m: LczeroModel, b: dict
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        return loss_fn(
            m,
            inputs=b["inputs"],
            value_targets=b["value_targets"],
            policy_targets=b["policy_targets"],
            movesleft_targets=b["movesleft_targets"],
        )

    loss_vfn = jax.vmap(loss_fn_inner, in_axes=(None, 0), out_axes=0)

    batch_dict = {
        "inputs": jax.device_put(b_inputs),
        "value_targets": jax.device_put(b_values),
        "policy_targets": jax.device_put(b_policy),
        "movesleft_targets": jax.device_put(b_movesleft),
    }

    per_sample_loss, unweighted = loss_vfn(model, batch_dict)
    return {
        "loss": jnp.mean(per_sample_loss),
        "unweighted_losses": jax.tree_util.tree_map(jnp.mean, unweighted),
    }


class _DataLoaderMetric(_EvaluatingMetric):
    """Metric that evaluates loss on dataloader output."""

    def __init__(
        self,
        config: MetricConfig,
        logger: TensorboardLogger,
        loss_fn: Optional[LczeroLoss],
        data_loader: Optional[DataLoader],
        dataloader_name: str,
    ):
        super().__init__(config, logger, loss_fn)
        if not data_loader:
            raise ValueError(f"Metric '{config.name}': DataLoader required")
        self.data_loader = data_loader
        self.dataloader_name = dataloader_name
        self.cached_batch: Optional[Batch] = None

    def get_batch(self) -> Batch:
        batch = self.data_loader.maybe_get_next(self.dataloader_name)
        if batch is not None:
            self.cached_batch = batch
        elif self.cached_batch is None:
            raise RuntimeError(
                f"No data for metric '{self.config.name}' "
                f"from dataloader '{self.dataloader_name}'"
            )
        return self.cached_batch


class _NpzMetric(_EvaluatingMetric):
    """Metric that evaluates loss on pre-loaded NPZ data."""

    def __init__(
        self,
        config: MetricConfig,
        logger: TensorboardLogger,
        loss_fn: Optional[LczeroLoss],
        npz_filename: str,
    ):
        super().__init__(config, logger, loss_fn)
        self.npz_data: Batch = load_batch_from_npz(npz_filename)

    def get_batch(self) -> Batch:
        return self.npz_data


class Metrics:
    """Manages metrics collection and logging for training."""

    def __init__(
        self,
        config: MetricsConfig,
        loss_fn: Optional[LczeroLoss] = None,
        data_loader: Optional[DataLoader] = None,
    ):
        self._metrics: Dict[str, _Metric] = {}

        for mc in config.metric:
            tb_logger = TensorboardLogger(
                os.path.join(config.tensorboard_path, mc.name)
            )

            metric: _Metric
            if mc.HasField("training_batch"):
                metric = _TrainingBatchMetric(mc, tb_logger)
            elif mc.HasField("dataloader_output"):
                metric = _DataLoaderMetric(
                    mc, tb_logger, loss_fn, data_loader, mc.dataloader_output
                )
            elif mc.HasField("npz_filename"):
                metric = _NpzMetric(mc, tb_logger, loss_fn, mc.npz_filename)
            else:
                raise ValueError(f"Metric '{mc.name}' has no sample source")

            self._metrics[mc.name] = metric

    def on_step(self, hook_data: StepHookData, graphdef: nnx.GraphDef) -> None:
        """Process metrics for the current step."""
        for metric in self._metrics.values():
            if metric.should_log(
                hook_data.global_step,
                hook_data.local_step,
                hook_data.steps_per_epoch,
            ):
                metric.log(hook_data, graphdef)

    def close(self) -> None:
        """Close all TensorBoard loggers."""
        for metric in self._metrics.values():
            metric.logger.close()
