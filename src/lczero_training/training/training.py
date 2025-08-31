import logging
import sys

from flax import nnx
from google.protobuf import text_format

from lczero_training.model.model import LczeroModel
from proto.root_config_pb2 import RootConfig
from proto.training_config_pb2 import TrainingConfig

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Training:
    def __init__(self, config: TrainingConfig, model: LczeroModel):
        self.config = config
        self.model = model

    def run(self) -> None:
        pass


if __name__ == "__main__":
    config = RootConfig()
    logger.info("Reading configuration from proto file")
    with open(
        "/home/crem/tmp/2025-08/lc0_training/training.textproto", "r"
    ) as f:
        text_format.Parse(f.read(), config)

    logger.info("Creating model from configuration")
    rngs = nnx.Rngs(params=42)
    model = LczeroModel(config=config.model, rngs=rngs)

    logger.info("Creating training instance")
    training = Training(config.training, model)
    training.run()
