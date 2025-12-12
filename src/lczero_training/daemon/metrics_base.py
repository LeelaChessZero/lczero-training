"""Base classes for metrics."""

from abc import ABC, abstractmethod

from flax import nnx

from lczero_training.training.tensorboard import TensorboardLogger
from lczero_training.training.training import StepHookData
from proto.metrics_config_pb2 import MetricConfig


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
