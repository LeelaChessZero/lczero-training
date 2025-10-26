"""Backfill metrics for existing checkpoints."""

import logging
from typing import Any

from flax import nnx
from google.protobuf import text_format

from lczero_training.daemon.metrics import (
    evaluate_batch,
    load_batch_from_npz,
)
from lczero_training.model.loss_function import LczeroLoss
from lczero_training.model.model import LczeroModel
from lczero_training.training.migrate_checkpoint import (
    Migration,
    get_checkpoint_steps,
    load_checkpoint,
    load_migration_rules,
)
from lczero_training.training.state import TrainingState
from lczero_training.training.tensorboard import TensorboardLogger
from proto.root_config_pb2 import RootConfig

logger = logging.getLogger(__name__)


def _load_config(config_path: str) -> RootConfig:
    """Load RootConfig from textproto file."""
    config = RootConfig()
    with open(config_path, "r") as f:
        text_format.Parse(f.read(), config)
    return config


def _validate_and_get_metrics(
    root_config: RootConfig, metric_names: list[str]
) -> dict[str, Any]:
    """Validate metrics exist and are NPZ type."""
    if not root_config.HasField("metrics"):
        raise ValueError("No metrics configuration found in root config")

    metric_configs = {
        mc.name: mc
        for mc in root_config.metrics.metric
        if mc.name in metric_names and mc.HasField("npz_filename")
    }

    non_npz = [
        mc.name
        for mc in root_config.metrics.metric
        if mc.name in metric_names and not mc.HasField("npz_filename")
    ]
    if non_npz:
        raise ValueError(f"Non-NPZ metrics: {', '.join(non_npz)}")

    missing = set(metric_names) - set(metric_configs.keys())
    if missing:
        raise ValueError(f"Metrics not found: {', '.join(sorted(missing))}")

    return metric_configs


def _load_and_migrate_checkpoint(
    checkpoint_path: str,
    step: int,
    template: TrainingState | None,
    rules: list[tuple],
) -> TrainingState:
    """Load checkpoint and apply migration if rules provided."""
    state, _ = load_checkpoint(checkpoint_path, step)
    if not rules:
        return state
    return Migration(state, template).run(rules)


def backfill_metrics(
    config_path: str,
    metric_names: list[str],
    min_step: int | None = None,
    max_step: int | None = None,
    migration_config_path: str | None = None,
) -> None:
    """Backfill metrics for existing checkpoints.

    Args:
        config_path: Path to the RootConfig textproto file.
        metric_names: Names of metrics to backfill (must be NPZ metrics).
        min_step: Minimum checkpoint step (inclusive), or None.
        max_step: Maximum checkpoint step (inclusive), or None.
        migration_config_path: Path to CheckpointMigrationConfig file, or None.

    Raises:
        ValueError: If any metric is not an NPZ metric or doesn't exist.
    """
    config = _load_config(config_path)
    metric_configs = _validate_and_get_metrics(config, metric_names)

    # Load batches and create loggers.
    batches = {
        name: load_batch_from_npz(mc.npz_filename)
        for name, mc in metric_configs.items()
    }
    loggers = {
        name: TensorboardLogger(f"{config.metrics.tensorboard_path}/{name}")
        for name in metric_configs
    }

    # Initialize model components.
    loss_fn = LczeroLoss(config=config.training.losses)
    graphdef, _ = nnx.split(
        LczeroModel(config=config.model, rngs=nnx.Rngs(params=42))
    )

    # Prepare migration if needed.
    rules = load_migration_rules(migration_config_path)
    template = (
        TrainingState.new_from_config(config.model, config.training)
        if rules
        else None
    )

    # Get and process checkpoints.
    steps = get_checkpoint_steps(
        config.training.checkpoint.path, min_step, max_step
    )
    logger.info(f"Processing {len(steps)} checkpoints")

    if not steps:
        logger.warning("No checkpoints found in range")
        return

    try:
        for step in steps:
            logger.info(f"Step {step}")
            state = _load_and_migrate_checkpoint(
                config.training.checkpoint.path, step, template, rules
            )

            for name, mc in metric_configs.items():
                metrics = evaluate_batch(
                    batches[name],
                    state.jit_state,
                    graphdef,
                    loss_fn,
                    mc.use_swa_model,
                )
                loggers[name].log(step, metrics)
                logger.info(f"  {name}: loss={metrics['loss']:.6f}")
    finally:
        for tb_logger in loggers.values():
            tb_logger.close()

    logger.info("Backfill complete")
