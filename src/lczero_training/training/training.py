import logging
import sys
from typing import Generator, Tuple

import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from google.protobuf import text_format

from lczero_training.dataloader import DataLoader, make_dataloader
from lczero_training.model.model import LczeroModel
from lczero_training.training.optimizer import make_gradient_transformation
from lczero_training.training.state import TrainingState
from proto.root_config_pb2 import RootConfig
from proto.training_config_pb2 import TrainingConfig

logger = logging.getLogger(__name__)


def from_dataloader(
    loader: DataLoader,
) -> Generator[Tuple[np.ndarray, ...], None, None]:
    while True:
        yield loader.get_next()


class Training:
    config: TrainingConfig
    model: nnx.GraphDef
    datagen: Generator[Tuple[np.ndarray, ...], None, None]
    optimizer: optax.GradientTransformation
    step: int
    model_state: nnx.State
    opt_state: optax.OptState

    def __init__(
        self,
        config: TrainingConfig,
        model: nnx.GraphDef,
        training_state: TrainingState,
        datagen: Generator[Tuple[np.ndarray, ...], None, None],
    ):
        self.config = config
        self.model = model
        self.step = training_state.step
        self.model_state = training_state.model_state
        assert training_state.opt_state is not None
        self.opt_state = training_state.opt_state

        self.datagen = datagen
        self.optimizer = make_gradient_transformation(config.optimizer)

    def run(self) -> TrainingState:
        model_and_state = nnx.merge(self.model, self.model_state)

        # ...

        return TrainingState(
            step=self.step,
            model_state=nnx.state(model_and_state),
            opt_state=self.opt_state,
        )


def train(config_filename: str) -> None:
    print("A")
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
    training = Training(
        config=config.training,
        model=model,
        training_state=training_state,
        datagen=from_dataloader(make_dataloader(config.data_loader)),
    )
    training.run()
