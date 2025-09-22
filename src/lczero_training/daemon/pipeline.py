import dataclasses
import gzip
import logging
import os
import time
from pathlib import Path
from typing import cast

import orbax.checkpoint as ocp
from flax import nnx
from google.protobuf import text_format

from lczero_training._lczero_training import DataLoader
from lczero_training.convert.jax_to_leela import (
    LeelaExportOptions,
    jax_to_leela,
)
from lczero_training.model.loss_function import LczeroLoss
from lczero_training.model.model import LczeroModel
from lczero_training.training.optimizer import make_gradient_transformation
from lczero_training.training.state import JitTrainingState, TrainingState
from lczero_training.training.training import Training, from_dataloader
from proto.data_loader_config_pb2 import DataLoaderConfig
from proto.root_config_pb2 import RootConfig
from proto.training_config_pb2 import ScheduleConfig

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

    def __init__(self, config_filepath: str) -> None:
        logger.info(f"Loading config from {config_filepath}")
        self._config = self._load_config(config_filepath)
        self._schedule = self._config.training.schedule
        self._chunks_per_network = self._schedule.chunks_per_network
        self._num_steps_per_epoch = self._schedule.steps_per_network
        self._chunks_to_wait = self._chunks_per_network
        self._cycle_state = _TrainingCycleState()
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
        optimizer_tx = make_gradient_transformation(
            self._config.training.optimizer
        )
        jit_state = JitTrainingState(
            step=0,
            model_state=nnx.state(self._model),
            opt_state=optimizer_tx.init(nnx.state(self._model)),
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
        self._training = Training(
            optimizer_tx=make_gradient_transformation(
                self._config.training.optimizer
            ),
            graphdef=nnx.graphdef(self._model),
            loss_fn=LczeroLoss(config=self._config.training.losses),
        )

        logger.info("Creating data loader")
        self._data_loader = _make_dataloader(self._config.data_loader)
        self._data_loader.set_chunk_anchor(
            self._training_state.last_chunk_source
        )

    def run(self) -> None:
        logging.info("Starting DataLoader")
        self._data_loader.start()

        while True:
            self._wait_for_chunks()
            new_anchor, used_chunks = self._data_loader.reset_chunk_anchor()
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
            self._export_model()

    def _export_model(self) -> None:
        if not self._config.export.HasField("path"):
            return
        export_filename = os.path.join(
            self._config.export.path,
            f"lc0-{self._training_state.jit_state.step:08d}.pb.gz",
        )

        logging.info(f"Exporting model to {export_filename}")

        options = LeelaExportOptions(
            min_version="0.28",
            num_heads=self._training_state.num_heads,
            license=None,
        )
        net = jax_to_leela(
            jax_weights=nnx.state(self._model),
            export_options=options,
        )
        logging.info(f"Writing model to {export_filename}")
        os.makedirs(self._config.export.path, exist_ok=True)
        with gzip.open(export_filename, "wb") as f:
            f.write(net.SerializeToString())
        logging.info(f"Finished writing model to {export_filename}")

    def _train_one_network(self) -> None:
        logging.info("Training one network!")

        # Record training start
        self._cycle_state.current_training_start_time = time.time()
        self._cycle_state.current_stage = TrainingStage.TRAINING
        self._cycle_state.chunks_at_training_start = (
            self._data_loader.chunks_since_anchor()
        )

        new_jit_state = self._training.run(
            jit_state=self._training_state.jit_state,
            datagen=from_dataloader(self._data_loader),
            num_steps=self._schedule.steps_per_network,
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

    def get_data_loader(self) -> DataLoader:
        return self._data_loader

    def _wait_for_chunks(self) -> None:
        logger.info(
            f"Waiting for {self._chunks_to_wait} chunks. "
            f"got {self._data_loader.chunks_since_anchor()} so far"
        )
        while self._data_loader.chunks_since_anchor() < self._chunks_to_wait:
            time.sleep(1)
        logger.info("Done waiting for enough chunks")

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
            self._data_loader.chunks_since_anchor()
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

    def _load_config(self, config_filepath: str) -> RootConfig:
        config_path = Path(config_filepath)
        config_text = config_path.read_text()

        root_config = RootConfig()
        text_format.Parse(config_text, root_config)
        return root_config
