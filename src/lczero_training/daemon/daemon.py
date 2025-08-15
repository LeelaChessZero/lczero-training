# ABOUTME: TrainingDaemon class that acts as subprocess for training operations.
# ABOUTME: Handles IPC communication via Communicator and implements message handlers.

import logging
import sys
from pathlib import Path
from google.protobuf import text_format
from lczero_training._lczero_training import DataLoader
import proto.data_loader_config_pb2 as config_pb2
from ..protocol.communicator import Communicator
from ..protocol.messages import StartTrainingPayload, TrainingStatusPayload


class TrainingDaemon:
    _data_loader: DataLoader | None = None

    def __init__(self):
        self._setup_logging()
        self._communicator = Communicator(self, sys.stdin, sys.stdout)

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format=(
                "%(levelname).1s%(asctime)s.%(msecs)03d %(name)s "
                "%(filename)s:%(lineno)d] %(message)s"
            ),
            datefmt="%m%d %H:%M:%S",
            stream=sys.stderr,
        )
        logging.info("TrainingDaemon starting up")

    def run(self):
        logging.info("TrainingDaemon ready for IPC communication")
        self._communicator.run()

    def on_start_training(self, payload: StartTrainingPayload):
        assert self._data_loader is None, "DataLoader already exists"

        config_path = Path(payload.config_filepath)
        config_text = config_path.read_text()

        training_config = config_pb2.TrainingConfig()
        text_format.Parse(config_text, training_config)

        data_loader_config_bytes = (
            training_config.data_loader.SerializeToString()
        )
        self._data_loader = DataLoader(data_loader_config_bytes)

    def on_training_status(self, payload: TrainingStatusPayload):
        pass
