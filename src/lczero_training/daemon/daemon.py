# ABOUTME: TrainingDaemon class that acts as subprocess for training operations.
# ABOUTME: Handles IPC communication via Communicator and implements message handlers.

import logging
import sys
import threading
import time
from pathlib import Path

import anyio
from google.protobuf import text_format

import lczero_training.proto.training_config_pb2 as config_pb2
from lczero_training._lczero_training import DataLoader
from lczero_training.proto import training_metrics_pb2

from ..protocol.communicator import Communicator
from ..protocol.messages import StartTrainingPayload, TrainingStatusPayload


class TrainingDaemon:
    _data_loader: DataLoader | None = None

    def __init__(self) -> None:
        self._setup_logging()
        self._communicator = Communicator(self, sys.stdin, sys.stdout)
        self._communicator_thread = threading.Thread(
            target=lambda: self._communicator.run(), daemon=True
        )
        self._communicator_thread.start()
        self._async_thread = threading.Thread(
            target=lambda: anyio.run(self._metrics_main), daemon=True
        )
        self._async_thread.start()

    def _setup_logging(self) -> None:
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

    async def _metrics_main(self) -> None:
        async with anyio.create_task_group() as tg:
            tg.start_soon(self._metrics_task)

    async def _metrics_task(self) -> None:
        while True:
            await anyio.sleep(1.1)

            dataloader_1_second = None
            dataloader_total = None
            dataloader_update_secs = None

            if self._data_loader is not None:
                stats_1_second_bytes, _ = self._data_loader.get_bucket_metrics(
                    0, False
                )  # k1Second = 0
                stats_total_bytes, dataloader_update_secs = (
                    self._data_loader.get_aggregate_ending_now(
                        float("inf"), False
                    )
                )

                dataloader_1_second = (
                    training_metrics_pb2.DataLoaderMetricsProto()
                )
                dataloader_1_second.ParseFromString(stats_1_second_bytes)

                dataloader_total = training_metrics_pb2.DataLoaderMetricsProto()
                dataloader_total.ParseFromString(stats_total_bytes)

            payload = TrainingStatusPayload(
                dataloader_update_secs=dataloader_update_secs,
                dataloader_1_second=dataloader_1_second,
                dataloader_total=dataloader_total,
            )
            self._communicator.send(payload)

    def run(self) -> None:
        while self._data_loader is None:
            logging.info("DataLoader is not ready")
            time.sleep(1)
        logging.info("DataLoader is ready")
        while True:
            self._data_loader.get_next()
            logging.info("DataLoader processed a batch")

    def on_start_training(self, payload: StartTrainingPayload) -> None:
        assert self._data_loader is None, "DataLoader already exists"

        config_path = Path(payload.config_filepath)
        config_text = config_path.read_text()

        training_config = config_pb2.TrainingConfig()
        text_format.Parse(config_text, training_config)

        data_loader_config_bytes = (
            training_config.data_loader.SerializeToString()
        )
        self._data_loader = DataLoader(data_loader_config_bytes)
