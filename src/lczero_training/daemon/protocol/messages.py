# ABOUTME: Payload dataclass definitions for JSONL IPC protocol messages.
# ABOUTME: Defines minimal event types for training daemon communication.

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import proto.training_metrics_pb2 as training_metrics_pb2

from .registry import register


class TrainingStage(Enum):
    WAITING_FOR_DATA = "WAITING FOR DATA"
    TRAINING = "TRAINING"


@dataclass
class TrainingScheduleData:
    current_stage: TrainingStage
    completed_epochs_since_start: int
    new_chunks_since_training_start: int
    chunks_to_wait: int
    total_uptime_seconds: float
    current_training_time_seconds: float
    previous_training_time_seconds: float
    current_cycle_time_seconds: float
    previous_cycle_time_seconds: float


# --- Notifications from UI (Parent) to Trainer (Child) ---


@register("start_training")
@dataclass
class StartTrainingPayload:
    config_filepath: str


# --- Notifications from Trainer (Child) to UI (Parent) ---


@register("training_status")
@dataclass
class TrainingStatusPayload:
    dataloader_update_secs: Optional[float] = None
    dataloader_1_second: Optional[
        training_metrics_pb2.DataLoaderMetricsProto
    ] = None
    dataloader_total: Optional[training_metrics_pb2.DataLoaderMetricsProto] = (
        None
    )
    training_schedule: Optional[TrainingScheduleData] = None
