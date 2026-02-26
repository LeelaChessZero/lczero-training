"""RMS metrics for model parameters."""

from typing import Any, cast

import jax
import jax.numpy as jnp
from flax import nnx

from lczero_training.daemon.metrics_base import _Metric
from lczero_training.model.encoder import EncoderBlock
from lczero_training.model.model import LczeroModel
from lczero_training.training.tensorboard import TensorboardLogger
from lczero_training.training.training import StepHookData
from proto.metrics_config_pb2 import MetricConfig


def compute_rms(state_subtree: nnx.State) -> float:
    """Compute RMS of all parameters in a state subtree."""
    leaves = jax.tree_util.tree_leaves(state_subtree)
    flat = jnp.concatenate([jnp.asarray(p).ravel() for p in leaves])
    return float(jnp.sqrt(jnp.mean(jnp.square(flat))))


def extract_attention_components(model: LczeroModel) -> dict[str, Any]:
    """Extract Q, K, V, output_dense, smolgen from all encoder layers.

    Args:
        model: LczeroModel instance.

    Returns:
        Dict with keys 'q', 'k', 'v', 'output_dense', optionally 'smolgen'.
    """
    components: dict[str, Any] = {
        "q": {},
        "k": {},
        "v": {},
        "output_dense": {},
    }

    encoder_layers = cast(list[EncoderBlock], model.encoders.encoders.layers)
    for i, encoder_block in enumerate(encoder_layers):
        mha = encoder_block.mha
        components["q"][f"layer_{i}"] = nnx.state(mha.q)
        components["k"][f"layer_{i}"] = nnx.state(mha.k)
        components["v"][f"layer_{i}"] = nnx.state(mha.v)
        components["output_dense"][f"layer_{i}"] = nnx.state(mha.output_dense)

        if mha.smolgen is not None:
            if "smolgen" not in components:
                components["smolgen"] = {}
            components["smolgen"][f"layer_{i}"] = nnx.state(mha.smolgen)

    return components


def collect_rms_metrics(model: LczeroModel) -> dict[str, Any]:
    """Collect all RMS metrics for the model.

    Args:
        model: LczeroModel instance.

    Returns:
        Nested dict with RMS values for different model components.
    """
    model_state = nnx.state(model)

    metrics: dict[str, Any] = {
        "all_params": compute_rms(model_state),
        "embedding": compute_rms(nnx.state(model.embedding)),
        "encoder_body": compute_rms(nnx.state(model.encoders)),
    }

    # Attention components
    attn_components = extract_attention_components(model)
    metrics["attention"] = {
        name: compute_rms(component)
        for name, component in attn_components.items()
    }

    # Policy heads
    metrics["policy_heads"] = {
        name: compute_rms(nnx.state(head))
        for name, head in model.policy_heads.items()
    }

    # Value heads
    metrics["value_heads"] = {
        name: compute_rms(nnx.state(head))
        for name, head in model.value_heads.items()
    }

    # Movesleft heads
    metrics["movesleft_heads"] = {
        name: compute_rms(nnx.state(head))
        for name, head in model.movesleft_heads.items()
    }

    return metrics


class _RmsMetric(_Metric):
    """Metric that computes RMS of model parameters."""

    def __init__(self, config: MetricConfig, logger: TensorboardLogger):
        super().__init__(config, logger)

    def log(self, hook_data: StepHookData, graphdef: nnx.GraphDef) -> None:
        model_state = (
            hook_data.jit_state.swa_state
            if self.config.use_swa_model
            else hook_data.jit_state.model_state
        )
        model = nnx.merge(graphdef, model_state)
        metrics = collect_rms_metrics(model)
        self.logger.log(hook_data.global_step, metrics)
