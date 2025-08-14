# ABOUTME: TrainingDaemon class that acts as subprocess for training operations.
# ABOUTME: Handles IPC communication via Communicator and implements message handlers.

import logging
import sys
from ..protocol.communicator import Communicator
from ..protocol.messages import StartTrainingPayload, TrainingStatusPayload


class TrainingDaemon:
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
        pass

    def on_training_status(self, payload: TrainingStatusPayload):
        pass
