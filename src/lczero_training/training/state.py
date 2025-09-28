import dataclasses
import logging
from typing import Any, Optional

import optax
from flax import nnx
from flax.struct import dataclass

from lczero_training.model.model import LczeroModel
from lczero_training.training.optimizer import make_gradient_transformation
from proto.model_config_pb2 import ModelConfig
from proto.training_config_pb2 import TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class JitTrainingState:
    step: int
    model_state: nnx.State
    opt_state: Optional[optax.OptState]

    def replace(self, **changes: Any) -> "JitTrainingState":
        """Returns a new instance of the class with the specified changes."""
        return dataclasses.replace(self, **changes)


@dataclass
class TrainingState:
    jit_state: JitTrainingState
    # Last chunk source that was available when the last epoch started training.
    num_heads: int
    last_chunk_source: str = ""

    def replace(self, **changes: Any) -> "TrainingState":
        """Returns a new instance of the class with the specified changes."""
        return dataclasses.replace(self, **changes)

    @staticmethod
    def new_from_config(
        model_config: ModelConfig, training_config: TrainingConfig
    ) -> "TrainingState":
        rngs = nnx.Rngs(params=42)
        model_state = nnx.state(LczeroModel(config=model_config, rngs=rngs))
        opt_state = make_gradient_transformation(
            training_config.optimizer,
            max_grad_norm=getattr(training_config, "max_grad_norm", 0.0),
        ).init(model_state)
        jit_state = JitTrainingState(
            step=0,
            model_state=model_state,
            opt_state=opt_state,
        )
        return TrainingState(
            jit_state=jit_state,
            num_heads=model_config.encoder.heads,
        )
