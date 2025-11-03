from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from proto.training_config_pb2 import (
    NadamwOptimizerConfig,
    OptimizerConfig,
)


def _make_nadamw_weight_decay_mask(
    config: NadamwOptimizerConfig, params: nnx.State
) -> nnx.State:
    """Creates a mask that excludes bias and LayerNorm parameters from decay."""

    def is_layer_norm(path: tuple[object, ...]) -> bool:
        return any(str(s).startswith("ln") for s in path)

    def is_embedding(path: tuple[object, ...]) -> bool:
        return ("embedding", "embedding") in zip(path, path[1:])

    def is_bias(path: tuple[object, ...]) -> bool:
        return str(path[-1]).lower() == "bias"

    def mask_fn(path: tuple[object, ...], variable: nnx.Variable) -> bool:
        if is_bias(path) and not config.decay_biases:
            return False
        if is_layer_norm(path) and not config.decay_layer_norms:
            return False
        if is_embedding(path) and not config.decay_embedding:
            return False
        return True

    return nnx.map_state(mask_fn, params)


def update_optimizer_step(
    opt_state: optax.OptState, step: int
) -> optax.OptState:
    """Updates all step counters in the optimizer state tree."""
    step_array = jnp.array(step, dtype=jnp.int32)

    def update_count(x: optax.OptState) -> optax.OptState:
        if isinstance(
            x,
            (
                optax.ScaleByAdamState,
                optax.ScaleByScheduleState,
            ),
        ):
            return x._replace(count=step_array)
        return x

    return jax.tree_util.tree_map(
        update_count, opt_state, is_leaf=lambda x: hasattr(x, "_replace")
    )


def make_gradient_transformation(
    config: OptimizerConfig,
    *,
    max_grad_norm: float | None = None,
    lr_schedule: optax.Schedule,
) -> optax.GradientTransformation:
    if config.HasField("nadamw"):
        conf = config.nadamw
        tx = optax.nadamw(
            lr_schedule,
            b1=conf.beta_1,
            b2=conf.beta_2,
            eps=conf.epsilon,
            weight_decay=conf.weight_decay,
            mask=partial(_make_nadamw_weight_decay_mask, conf),
        )
        if max_grad_norm is not None and max_grad_norm > 0:
            tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), tx)
        return tx
    else:
        raise ValueError(
            "Unsupported optimizer type: {}".format(
                config.WhichOneof("optimizer_type")
            )
        )
