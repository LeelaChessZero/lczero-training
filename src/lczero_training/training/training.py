import logging
from typing import Generator, Tuple

import numpy as np
import orbax.checkpoint as ocp
from absl import app
from absl import logging as absl_logging
from flax import nnx
from google.protobuf import text_format

from lczero_training.dataloader import DataLoader, make_dataloader
from lczero_training.model.model import LczeroModel
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
    model: LczeroModel
    checkpointer: ocp.StandardCheckpointer
    datagen: Generator[Tuple[np.ndarray, ...], None, None]

    def __init__(
        self,
        config: TrainingConfig,
        model: LczeroModel,
        datagen: Generator[Tuple[np.ndarray, ...], None, None],
    ):
        self.config = config
        self.model = model
        self.checkpointer = ocp.StandardCheckpointer()
        self.datagen = datagen

        assert config.checkpoint.path, "Checkpoint path must be set"
        logger.info(f"Loading checkpoint from {config.checkpoint.path}")
        state = nnx.state(model)
        state = self.checkpointer.restore(config.checkpoint.path, state)
        nnx.update(model, state)

    def run(self) -> None:
        pass


def main(argv: list[str]) -> None:
    del argv  # Unused.
    config = RootConfig()
    logger.info("Reading configuration from proto file")
    with open(
        "/home/crem/tmp/2025-08/lc0_training/training.textproto", "r"
    ) as f:
        text_format.Parse(f.read(), config)

    logger.info("Creating model from configuration")
    rngs = nnx.Rngs(params=42)
    model = LczeroModel(config=config.model, rngs=rngs)

    logger.info("Creating dataloader from configuration")
    datagen = make_dataloader(config.data_loader)

    logger.info("Creating training instance")
    training = Training(config.training, model, from_dataloader(datagen))
    training.run()


if __name__ == "__main__":
    absl_logging.set_verbosity(absl_logging.INFO)
    app.run(main)
