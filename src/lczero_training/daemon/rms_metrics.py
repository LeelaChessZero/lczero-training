"""RMS metrics for model parameters."""

from typing import Any, cast
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

from lczero_training.daemon.metrics_base import _Metric
from lczero_training.model.encoder import EncoderBlock
from lczero_training.model.model import LczeroModel
from lczero_training.training.tensorboard import TensorboardLogger
from lczero_training.training.training import StepHookData
from lczero_training.training.state import JitTrainingState
from proto.metrics_config_pb2 import MetricConfig


def compute_rms(state_subtree: nnx.State) -> jax.Array:
    """Compute RMS of all parameters in a state subtree."""
    leaves = jax.tree_util.tree_leaves(state_subtree)
    total_sq = sum(jnp.sum(jnp.square(p)) for p in leaves)
    total_n = sum(p.size for p in leaves)
    return jnp.sqrt(total_sq / total_n)


def extract_attention_components(encoders_layers: nnx.State) -> dict[str, Any]:
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

    for i in range(len(encoders_layers)):
        mha = encoders_layers[i].mha
        components["q"][f"layer_{i}"] = nnx.state(mha.q)
        components["k"][f"layer_{i}"] = nnx.state(mha.k)
        components["v"][f"layer_{i}"] = nnx.state(mha.v)
        components["output_dense"][f"layer_{i}"] = nnx.state(mha.output_dense)

        if mha.smolgen is not None:
            if "smolgen" not in components:
                components["smolgen"] = {}
            components["smolgen"][f"layer_{i}"] = nnx.state(mha.smolgen)

    return components


@partial(jax.jit, static_argnames=("use_swa_model",))
def collect_rms_metrics(
        jit_state: JitTrainingState,
        use_swa_model: bool,
    ) -> dict[str, Any]:
    """Collect all RMS metrics for the model.

    Args:
        model: LczeroModel instance.

    Returns:
        Nested dict with RMS values for different model components.
    """
    model_state = jit_state.swa_state if use_swa_model else jit_state.model_state

    embedding_state = model_state['embedding']
    encoders_state = model_state['encoders']
    policy_heads = model_state['policy_heads']
    value_heads = model_state['value_heads']
    movesleft_heads = model_state['movesleft_heads']

    metrics: dict[str, Any] = {
        "all_params": compute_rms(model_state),
        "embedding": compute_rms(embedding_state),
        "encoder_body": compute_rms(encoders_state),
    }

    encoders_layers = encoders_state['encoders']['layers']

    # Attention components
    attn_components = extract_attention_components(encoders_layers)
    metrics["attention"] = {
        name: compute_rms(component)
        for name, component in attn_components.items()
    }

    # Policy heads
    metrics["policy_heads"] = {
        name: compute_rms(nnx.state(head))
        for name, head in policy_heads.items()
    }

    # Value heads
    metrics["value_heads"] = {
        name: compute_rms(nnx.state(head))
        for name, head in value_heads.items()
    }

    # Movesleft heads
    metrics["movesleft_heads"] = {
        name: compute_rms(nnx.state(head))
        for name, head in movesleft_heads.items()
    }

    return metrics


class _RmsMetric(_Metric):
    """Metric that computes RMS of model parameters."""

    def __init__(self, config: MetricConfig, logger: TensorboardLogger):
        super().__init__(config, logger)
        self.previous_metrics = None

    def __del__(self):
        self._do_log(self.previous_metrics)

    def log(self, hook_data: StepHookData, graphdef: nnx.GraphDef) -> None:
        metrics = collect_rms_metrics(hook_data.jit_state, self.config.use_swa_model)
        previous = self.previous_metrics
        self.previous_metrics = {
            'step': hook_data.global_step,
            'metrics': jax.copy_to_host_async(metrics),
        }
        self._do_log(previous)

    def _do_log(self, previous: dict[str, Any] | None) -> None:
        if previous is not None:
            self.logger.log(previous['step'], previous['metrics'])
