# ABOUTME: Payload dataclass definitions for JSONL IPC protocol messages.
# ABOUTME: Defines minimal event types for training daemon communication.

from dataclasses import dataclass

from ..proto import training_metrics_pb2
from .registry import register

# --- Notifications from UI (Parent) to Trainer (Child) ---


@register("start_training")
@dataclass
class StartTrainingPayload:
    config_filepath: str


# --- Notifications from Trainer (Child) to UI (Parent) ---


@register("training_status")
@dataclass
class TrainingStatusPayload:
    dataloader_update_secs: float | None = None
    dataloader_1_second: training_metrics_pb2.DataLoaderMetricsProto | None = (
        None
    )
    dataloader_total: training_metrics_pb2.DataLoaderMetricsProto | None = None
