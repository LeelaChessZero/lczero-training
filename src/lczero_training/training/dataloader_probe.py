"""Utilities for exercising the training data loader."""

import logging
import time
from contextlib import suppress
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from google.protobuf import text_format

from lczero_training.dataloader import DataLoader, make_dataloader
from proto.root_config_pb2 import RootConfig

logger = logging.getLogger(__name__)


def _stop_loader(loader: DataLoader) -> None:
    with suppress(Exception):
        loader.stop()


def _materialize_batch(batch: Sequence[np.ndarray]) -> Tuple[np.ndarray, ...]:
    return tuple(np.asarray(tensor).copy() for tensor in batch)


def _store_batches(path: str, batches: List[Tuple[np.ndarray, ...]]) -> None:
    output = Path(path)
    if output.parent:
        output.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing %d batches to %s", len(batches), output)
    container = np.empty(len(batches), dtype=object)
    container[:] = batches
    np.savez(output, batches=container)


def probe_dataloader(
    config_filename: str, num_batches: int, npz_output: Optional[str] = None
) -> None:
    """Measure latency and throughput for the configured data loader.

    Args:
        config_filename: Path to the root configuration proto file.
        num_batches: Total number of batches to fetch from the loader.
        npz_output: Optional path to store fetched batches as an .npz archive.
    """

    if num_batches < 1:
        raise ValueError("num_batches must be at least 1")

    config = RootConfig()
    logger.info("Reading configuration from proto file")
    with open(config_filename, "r") as config_file:
        text_format.Parse(config_file.read(), config)

    logger.info("Creating data loader")
    loader = make_dataloader(config.data_loader)

    collected_batches: List[Tuple[np.ndarray, ...]] = []
    collect_enabled = npz_output is not None
    first_batch_time = 0.0
    remaining_batches = num_batches - 1
    try:
        logger.info("Fetching first batch")
        start_time = time.perf_counter()
        first_batch = loader.get_next()
        if collect_enabled:
            collected_batches.append(_materialize_batch(first_batch))
        first_batch_time = time.perf_counter() - start_time
        logger.info("Time to first batch: %.3f seconds", first_batch_time)

        if remaining_batches <= 0:
            logger.info("Only fetched first batch; skipping throughput")
            return

        logger.info(
            "Fetching %d additional batches for throughput measurement",
            remaining_batches,
        )
        throughput_start = time.perf_counter()
        for _ in range(remaining_batches):
            batch = loader.get_next()
            if collect_enabled:
                collected_batches.append(_materialize_batch(batch))
        throughput_duration = time.perf_counter() - throughput_start

        if throughput_duration <= 0:
            logger.warning("Measured non-positive duration; skipping rate")
            return

        batches_per_second = remaining_batches / throughput_duration
        logger.info(
            "Throughput excluding first batch: %.2f batches/second",
            batches_per_second,
        )
        logger.info(
            "Total time excluding first batch: %.3f seconds",
            throughput_duration,
        )
    finally:
        if collect_enabled and npz_output is not None and collected_batches:
            _store_batches(npz_output, collected_batches)
        _stop_loader(loader)
