import logging
from pathlib import Path

from google.protobuf import text_format

import proto.root_config_pb2 as config_pb2
from lczero_training._lczero_training import DataLoader


class TrainingPipeline:
    _data_loader: DataLoader

    def __init__(self, config_filepath: str) -> None:
        config_path = Path(config_filepath)
        config_text = config_path.read_text()

        root_config = config_pb2.RootConfig()
        text_format.Parse(config_text, root_config)

        data_loader_config_bytes = root_config.data_loader.SerializeToString()
        self._data_loader = DataLoader(data_loader_config_bytes)
        self._data_loader.start()

    def run(self) -> None:
        logging.info("DataLoader is ready")
        while True:
            self._data_loader.get_next()
            logging.info("DataLoader processed a batch")

    def stop(self) -> None:
        self._data_loader.stop()

    def get_data_loader(self) -> DataLoader:
        return self._data_loader
