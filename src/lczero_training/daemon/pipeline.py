import logging
import time
from pathlib import Path
from typing import cast

import orbax.checkpoint as ocp
from flax import nnx
from google.protobuf import text_format

from lczero_training._lczero_training import DataLoader
from lczero_training.model.loss_function import LczeroLoss
from lczero_training.model.model import LczeroModel
from lczero_training.training.optimizer import make_gradient_transformation
from lczero_training.training.state import JitTrainingState, TrainingState
from lczero_training.training.training import Training, from_dataloader
from proto.data_loader_config_pb2 import DataLoaderConfig
from proto.root_config_pb2 import RootConfig
from proto.training_config_pb2 import ScheduleConfig

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


class TrainingPipeline:
    _data_loader: DataLoader
    _schedule: ScheduleConfig
    _chunks_to_wait: int
    _model: LczeroModel
    _checkpoint_mgr: ocp.CheckpointManager
    _training_state: TrainingState

    def __init__(self, config_filepath: str) -> None:
        logger.info(f"Loading config from {config_filepath}")
        config = _read_config_file(config_filepath)
        self._schedule = config.training.schedule
        self._chunks_per_network = self._schedule.chunks_per_network
        self._num_steps_per_epoch = self._schedule.steps_per_network
        self._chunks_to_wait = self._chunks_per_network
        logger.info("Creating empty model")
        self._model = LczeroModel(config.model, rngs=nnx.Rngs(params=42))
        logger.info(
            f"Creating checkpoint manager at {config.training.checkpoint.path}"
        )
        self._checkpoint_mgr = ocp.CheckpointManager(
            config.training.checkpoint.path,
            options=ocp.CheckpointManagerOptions(
                max_to_keep=config.training.checkpoint.max_to_keep or None,
            ),
        )

        logger.info("Restoring checkpoint")
        optimizer_tx = make_gradient_transformation(config.training.optimizer)
        jit_state = JitTrainingState(
            step=0,
            model_state=nnx.state(self._model),
            opt_state=optimizer_tx.init(nnx.state(self._model)),
        )
        empty_state = TrainingState(
            jit_state=jit_state,
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
                config.training.optimizer
            ),
            graphdef=nnx.graphdef(self._model),
            loss_fn=LczeroLoss(config=config.training.losses),
        )

        logger.info("Creating data loader")
        self._data_loader = _make_dataloader(config.data_loader)
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

    def _train_one_network(self) -> None:
        logging.info("Training one network!")
        new_jit_state = self._training.run(
            jit_state=self._training_state.jit_state,
            datagen=from_dataloader(self._data_loader),
            num_steps=self._schedule.steps_per_network,
        )
        self._training_state = self._training_state.replace(
            jit_state=new_jit_state
        )
        logging.info("Done training")

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
