"""Utilities for writing training metrics to TensorBoard event files."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Dict

import jax
import numpy as np
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MetricsDict = Dict[str, Any]


def _to_ndarray(value: Any) -> np.ndarray:
    try:
        return np.asarray(jax.device_get(value))
    except TypeError:
        return np.asarray(value)


def _to_scalar(value: Any) -> float | None:
    array = _to_ndarray(value)
    if array.ndim == 0 or array.size == 1:
        return float(array.reshape(()))
    logger.warning(
        "Skipping non-scalar metric with shape %s when logging to TensorBoard.",
        array.shape,
    )
    return None


def _flatten_metrics(
    metrics: Mapping[str, Any], prefix: str = ""
) -> Dict[str, float]:
    scalars: Dict[str, float] = {}
    for key, value in metrics.items():
        tag = f"{prefix}{key}" if prefix else key
        if isinstance(value, Mapping):
            scalars.update(_flatten_metrics(value, f"{tag}/"))
            continue
        scalar = _to_scalar(value)
        if scalar is not None:
            scalars[tag] = scalar
    return scalars


def _to_step(step: Any) -> int:
    return int(_to_ndarray(step).reshape(()))


class TensorboardLogger:
    """Writes scalar training metrics to TensorBoard."""

    def __init__(self, logdir: str) -> None:
        self._writer = SummaryWriter(logdir)

    def log(self, step: int, metrics: MetricsDict) -> None:
        global_step = _to_step(step)
        for tag, value in _flatten_metrics(metrics).items():
            self._writer.add_scalar(
                tag=tag, scalar_value=value, global_step=global_step
            )

    def close(self) -> None:
        self._writer.close()
