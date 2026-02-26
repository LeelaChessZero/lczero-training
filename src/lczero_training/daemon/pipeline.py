import dataclasses
import datetime
import gzip
import logging
import os
import threading
import time
from pathlib import Path
from typing import cast

import jax
import orbax.checkpoint as ocp
import requests
from dotenv import load_dotenv
from flax import nnx
from google.protobuf import text_format

from lczero_training._lczero_training import DataLoader
from lczero_training.convert.jax_to_leela import (
    LeelaExportOptions,
    jax_to_leela,
)
from lczero_training.model.loss_function import LczeroLoss
from lczero_training.model.model import LczeroModel
from lczero_training.training.lr_schedule import make_lr_schedule
from lczero_training.training.optimizer import make_gradient_transformation
from lczero_training.training.state import JitTrainingState, TrainingState
from lczero_training.training.training import (
    StepHookData,
    Training,
    from_dataloader,
)
from proto.data_loader_config_pb2 import DataLoaderConfig
from proto.root_config_pb2 import RootConfig
from proto.stage_control_pb2 import StageControlRequest, StageControlResponse
from proto.training_config_pb2 import ScheduleConfig

from .metrics import Metrics
from .protocol.messages import TrainingScheduleData, TrainingStage

logger = logging.getLogger(__name__)


def _read_config_file(config_filepath: str) -> RootConfig:
    config_path = Path(config_filepath)
    config_text = config_path.read_text()

    root_config = RootConfig()
    text_format.Parse(config_text, root_config)
    return root_config


def _make_dataloader(config: DataLoaderConfig) -> DataLoader:
    config_bytes = config.SerializeToString()
    return DataLoader(config_bytes)


def _configure_file_logging(config: RootConfig) -> None:
    """Configure file logging if log_filename is specified in config."""
    if config.HasField("log_filename"):
        file_handler = logging.FileHandler(config.log_filename)
        file_handler.setFormatter(
            logging.Formatter(
                "%(levelname).1s%(asctime)s.%(msecs)03d %(name)s "
                "%(filename)s:%(lineno)d] %(message)s",
                datefmt="%m%d %H:%M:%S",
            )
        )
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Added file logging to {config.log_filename}")


def _log_jax_system_info() -> None:
    """Log JAX system information including devices and backend details."""
    devices = jax.devices()
    local_devices = jax.local_devices()
    device_counts: dict[str, int] = {}
    for device in devices:
        device_type = device.device_kind
        device_counts[device_type] = device_counts.get(device_type, 0) + 1

    logger.info(f"JAX Backend: {jax.default_backend()}")
    logger.info(
        f"JAX Devices: {len(devices)} total, {len(local_devices)} local"
    )
    for device_type, count in device_counts.items():
        logger.info(f"  {device_type}: {count}")
    for i, device in enumerate(local_devices):
        logger.info(f"  Local device {i}: {device}")


@dataclasses.dataclass
class _TrainingCycleState:
    start_time: float = dataclasses.field(default_factory=time.time)
    current_stage: TrainingStage = TrainingStage.WAITING_FOR_DATA
    completed_epochs: int = 0
    current_cycle_start_time: float = dataclasses.field(
        default_factory=time.time
    )
    current_training_start_time: float | None = None
    previous_training_duration: float = 0.0
    previous_cycle_duration: float = 0.0
    chunks_at_training_start: int = 0


class TrainingPipeline:
    _data_loader: DataLoader
    _schedule: ScheduleConfig
    _chunks_to_wait: int
    _model: LczeroModel
    _checkpoint_mgr: ocp.CheckpointManager
    _training_state: TrainingState
    _cycle_state: _TrainingCycleState
    _metrics: Metrics | None

    def __init__(self, config_filepath: str) -> None:
        logger.info(f"Loading config from {config_filepath}")
        self._config = self._load_config(config_filepath)
        _configure_file_logging(self._config)
        self._schedule = self._config.training.schedule
        self._chunks_per_network = self._schedule.chunks_per_network
        self._num_steps_per_epoch = self._schedule.steps_per_network
        self._chunks_to_wait = self._chunks_per_network
        self._cycle_state = _TrainingCycleState()
        self._force_training_event = threading.Event()
        self._metrics = None
        logger.info("Creating empty model")
        self._model = LczeroModel(self._config.model, rngs=nnx.Rngs(params=42))
        logger.info(
            f"Creating checkpoint manager at {self._config.training.checkpoint.path}"
        )
        self._checkpoint_mgr = ocp.CheckpointManager(
            self._config.training.checkpoint.path,
            options=ocp.CheckpointManagerOptions(
                max_to_keep=self._config.training.checkpoint.max_to_keep
                or None,
            ),
        )

        logger.info("Restoring checkpoint")
        optimizer_config = self._config.training.optimizer
        max_grad_norm = getattr(self._config.training, "max_grad_norm", 0.0)
        self._lr_schedule = make_lr_schedule(self._config.training.lr_schedule)
        optimizer_tx = make_gradient_transformation(
            optimizer_config,
            max_grad_norm=max_grad_norm,
            lr_schedule=self._lr_schedule,
        )
        model_state = nnx.state(self._model)
        jit_state = JitTrainingState(
            step=0,
            model_state=model_state,
            opt_state=optimizer_tx.init(model_state),
            swa_state=model_state,
            num_averages=0.0,
        )
        empty_state = TrainingState(
            jit_state=jit_state,
            num_heads=self._config.model.encoder.heads,
        )
        self._training_state = cast(
            TrainingState,
            self._checkpoint_mgr.restore(
                step=None,
                args=ocp.args.PyTreeRestore(
                    item=empty_state,
                ),
            ),
        )

        logger.info("Creating training session")
        loss_fn = LczeroLoss(config=self._config.training.losses)
        self._training = Training(
            optimizer_tx=make_gradient_transformation(
                self._config.training.optimizer,
                max_grad_norm=max_grad_norm,
                lr_schedule=self._lr_schedule,
            ),
            graphdef=nnx.graphdef(self._model),
            loss_fn=loss_fn,
            swa_config=(
                self._config.training.swa
                if self._config.training.HasField("swa")
                else None
            ),
        )

        logger.info("Creating data loader")
        self._data_loader = _make_dataloader(self._config.data_loader)
        self._set_chunk_anchor(self._training_state.last_chunk_source)

        # Create metrics if configured.
        if self._config.HasField("metrics"):
            logger.info("Creating metrics")
            self._metrics = Metrics(
                config=self._config.metrics,
                loss_fn=loss_fn,
                data_loader=self._data_loader,
            )
        else:
            logger.info("No metrics configured")

        _log_jax_system_info()

    def start_training_immediately(self) -> None:
        """Request the next training cycle to start without waiting for chunks."""

        logger.info("Received request to start training immediately.")
        self._force_training_event.set()

    def run(self) -> None:
        logging.info("Starting DataLoader")
        self._data_loader.start()

        while True:
            self._wait_for_chunks()
            new_anchor, used_chunks = self._reset_chunk_anchor()
            logging.info(f"{new_anchor=} {used_chunks=}")
            self._training_state = self._training_state.replace(
                last_chunk_source=new_anchor
            )
            self._chunks_to_wait = max(
                self._chunks_to_wait + self._chunks_per_network - used_chunks,
                self._chunks_per_network // 2,
            )
            self._train_one_network()
            self._save_checkpoint()
            network_bytes = self._export_network()
            if network_bytes:
                self._save_network(network_bytes)
                self._upload_network(network_bytes)

    def _export_network(self) -> bytes | None:
        if (
            not self._config.export.destination_filename
            and not self._config.export.HasField("upload_training_run")
        ):
            return None

        logging.info("Exporting network")

        options = LeelaExportOptions(
            min_version="0.31",
            num_heads=self._training_state.num_heads,
            license=None,
        )
        export_state = (
            self._training_state.jit_state.swa_state
            if self._config.export.export_swa_model
            else self._training_state.jit_state.model_state
        )
        assert isinstance(export_state, nnx.State)
        net = jax_to_leela(jax_weights=export_state, export_options=options)
        return gzip.compress(net.SerializeToString())

    def _save_network(self, network_bytes: bytes) -> None:
        date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        step = self._training_state.jit_state.step

        for destination_template in self._config.export.destination_filename:
            destination = destination_template.format(
                datetime=date_str, step=step
            )
            logging.info(f"Writing network to {destination}")
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            with open(destination, "wb") as f:
                f.write(network_bytes)
            logging.info(f"Finished writing network to {destination}")

    def _upload_network(self, network_bytes: bytes) -> None:
        if not self._config.export.HasField("upload_training_run"):
            return

        load_dotenv()
        upload_pwd = os.getenv("UPLOAD_PWD")
        if not upload_pwd:
            logging.error(
                "UPLOAD_PWD not found in environment variables, skipping upload."
            )
            return

        try:
            state = cast(nnx.State, nnx.state(self._model))
            layers = len(state["encoders"]["encoders"]["layers"])
            filters = state["embedding"]["embedding"]["bias"].shape[0]
            training_id = self._config.export.upload_training_run

            logging.info(
                f"Uploading network to training website (ID: {training_id}, "
                f"layers: {layers}, filters: {filters})"
            )

            data = {
                "pwd": upload_pwd,
                "training_id": training_id,
                "layers": layers,
                "filters": filters,
            }
            response = requests.post(
                "http://api.lczero.org/upload_network",
                files={"file": network_bytes},
                data=data,
            )
            response.raise_for_status()

            logging.info(f"Successfully uploaded network: {response.text}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to upload network: {e}")
        except (KeyError, AttributeError, IndexError) as e:
            logging.error(f"Failed to extract model metadata for upload: {e}")

    def _step_hook(self, hook_data: StepHookData) -> None:
        # Append current learning rate from schedule to metrics.
        hook_data.metrics["lr"] = self._lr_schedule(hook_data.global_step)
        if self._metrics is not None:
            self._metrics.on_step(hook_data, nnx.graphdef(self._model))

    def _train_one_network(self) -> None:
        logging.info("Training one network!")

        # Record training start
        self._cycle_state.current_training_start_time = time.time()
        self._cycle_state.current_stage = TrainingStage.TRAINING
        self._cycle_state.chunks_at_training_start = self._chunks_since_anchor()

        new_jit_state = self._training.run(
            jit_state=self._training_state.jit_state,
            datagen=from_dataloader(self._data_loader),
            num_steps=self._schedule.steps_per_network,
            step_hook=self._step_hook,
        )
        self._training_state = self._training_state.replace(
            jit_state=new_jit_state
        )

        # Record training end
        current_time = time.time()
        if self._cycle_state.current_training_start_time:
            self._cycle_state.previous_training_duration = (
                current_time - self._cycle_state.current_training_start_time
            )
        self._cycle_state.previous_cycle_duration = (
            current_time - self._cycle_state.current_cycle_start_time
        )
        self._cycle_state.completed_epochs += 1
        self._cycle_state.current_training_start_time = None
        self._cycle_state.current_stage = TrainingStage.WAITING_FOR_DATA
        self._cycle_state.current_cycle_start_time = current_time

        logging.info("Done training")

    def _save_checkpoint(self) -> None:
        logging.info("Saving checkpoint")
        self._checkpoint_mgr.save(
            step=self._training_state.jit_state.step,
            args=ocp.args.PyTreeSave(item=self._training_state),
        )
        logging.info("Checkpoint saved")

    def stop(self) -> None:
        self._data_loader.stop()
        if self._metrics is not None:
            self._metrics.close()

    def get_data_loader(self) -> DataLoader:
        return self._data_loader

    def _wait_for_chunks(self) -> None:
        current_chunks = self._chunks_since_anchor()
        logger.info(
            f"Waiting for {self._chunks_to_wait} chunks. "
            f"got {current_chunks} so far"
        )
        while True:
            if self._force_training_event.is_set():
                logger.info(
                    "Force start requested; skipping remaining chunk wait."
                )
                self._force_training_event.clear()
                self._chunks_to_wait = self._chunks_since_anchor()
                return

            if self._chunks_since_anchor() >= self._chunks_to_wait:
                logger.info("Done waiting for enough chunks")
                return

            time.sleep(1)

    def get_training_schedule_data(
        self, daemon_start_time: float
    ) -> TrainingScheduleData:
        """Return current training schedule data for TUI display."""
        current_time = time.time()

        # Calculate current training time if currently training
        current_training_time = 0.0
        if self._cycle_state.current_training_start_time is not None:
            current_training_time = (
                current_time - self._cycle_state.current_training_start_time
            )

        # Calculate current cycle time
        current_cycle_time = (
            current_time - self._cycle_state.current_cycle_start_time
        )

        # Calculate new chunks since training start
        new_chunks_since_training_start = max(
            0,
            self._chunks_since_anchor()
            - self._cycle_state.chunks_at_training_start,
        )

        return TrainingScheduleData(
            current_stage=self._cycle_state.current_stage,
            completed_epochs_since_start=self._cycle_state.completed_epochs,
            new_chunks_since_training_start=new_chunks_since_training_start,
            chunks_to_wait=self._chunks_to_wait,
            total_uptime_seconds=current_time - daemon_start_time,
            current_training_time_seconds=current_training_time,
            previous_training_time_seconds=self._cycle_state.previous_training_duration,
            current_cycle_time_seconds=current_cycle_time,
            previous_cycle_time_seconds=self._cycle_state.previous_cycle_duration,
        )

    def _send_chunk_pool_control(
        self, request: StageControlRequest
    ) -> StageControlResponse | None:
        responses = self._data_loader.send_control_message(request)
        for _, response in responses:
            if response.HasField("chunk_pool_response"):
                return response
        return None

    def _reset_chunk_anchor(self) -> tuple[str, int]:
        request = StageControlRequest()
        request.chunk_pool_request.reset_chunk_anchor = True
        response = self._send_chunk_pool_control(request)
        if not response or not response.HasField("chunk_pool_response"):
            return "", 0
        chunk_response = response.chunk_pool_response
        return chunk_response.chunk_anchor, chunk_response.chunks_since_anchor

    def _chunks_since_anchor(self) -> int:
        request = StageControlRequest()
        request.chunk_pool_request.SetInParent()
        response = self._send_chunk_pool_control(request)
        if not response or not response.HasField("chunk_pool_response"):
            return 0
        return response.chunk_pool_response.chunks_since_anchor

    def _set_chunk_anchor(self, anchor: str) -> None:
        request = StageControlRequest()
        request.chunk_pool_request.set_chunk_anchor = anchor or ""
        self._send_chunk_pool_control(request)

    def _load_config(self, config_filepath: str) -> RootConfig:
        config_path = Path(config_filepath)
        config_text = config_path.read_text()

        root_config = RootConfig()
        text_format.Parse(config_text, root_config)
        return root_config
