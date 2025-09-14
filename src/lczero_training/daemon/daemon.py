import logging
import signal
import sys
import threading
import time

import anyio

import proto.training_metrics_pb2 as training_metrics_pb2

from .pipeline import TrainingPipeline
from .protocol.communicator import Communicator
from .protocol.messages import StartTrainingPayload, TrainingStatusPayload


class TrainingDaemon:
    _training_pipeline: TrainingPipeline | None = None
    _config_filepath: str | None = None
    _daemon_start_time: float

    def __init__(self) -> None:
        self._daemon_start_time = time.time()
        self._setup_logging()
        self._setup_signal_handling()
        self._communicator = Communicator(self, sys.stdin, sys.stdout)
        self._communicator_thread = threading.Thread(
            target=lambda: self._communicator.run(), daemon=True
        )
        self._communicator_thread.start()
        self._async_thread = threading.Thread(
            target=lambda: anyio.run(self._metrics_main), daemon=True
        )
        self._async_thread.start()
        self._signal_thread = threading.Thread(
            target=self._signal_handler_thread, daemon=True
        )
        self._signal_thread.start()

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

    def _setup_signal_handling(self) -> None:
        # Block SIGINT and SIGTERM on all threads
        signal.pthread_sigmask(
            signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM}
        )

    def _signal_handler_thread(self) -> None:
        # Wait for SIGINT or SIGTERM
        signum = signal.sigwait({signal.SIGINT, signal.SIGTERM})
        self._shutdown(signum)

    def _shutdown(self, signum: int) -> None:
        logging.info(f"Received signal {signum}, shutting down...")
        if self._training_pipeline:
            self._training_pipeline.stop()

    async def _metrics_main(self) -> None:
        async with anyio.create_task_group() as tg:
            tg.start_soon(self._metrics_task)

    async def _metrics_task(self) -> None:
        while True:
            await anyio.sleep(1.1)

            dataloader_1_second = None
            dataloader_total = None
            dataloader_update_secs = None
            training_schedule_data = None

            data_loader = None
            if self._training_pipeline:
                data_loader = self._training_pipeline.get_data_loader()
                training_schedule_data = (
                    self._training_pipeline.get_training_schedule_data(
                        self._daemon_start_time
                    )
                )

            if data_loader is not None:
                stats_1_second_bytes, _ = data_loader.get_bucket_metrics(
                    0, False
                )  # k1Second = 0
                stats_total_bytes, dataloader_update_secs = (
                    data_loader.get_aggregate_ending_now(float("inf"), False)
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
                training_schedule=training_schedule_data,
            )
            self._communicator.send(payload)

    def run(self) -> None:
        while self._config_filepath is None:
            logging.info("Waiting for training config...")
            time.sleep(1)

        logging.info("Config received. Starting training pipeline.")
        self._training_pipeline = TrainingPipeline(self._config_filepath)
        self._training_pipeline.run()

    def on_start_training(self, payload: StartTrainingPayload) -> None:
        self._config_filepath = payload.config_filepath
