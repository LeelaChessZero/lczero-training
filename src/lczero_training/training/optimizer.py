from functools import partial

import jax
import jax.numpy as jnp
import optax

from lczero_training.training.utils import make_weights_mask
from proto.training_config_pb2 import OptimizerConfig


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
        nadamw = config.nadamw
        tx = optax.nadamw(
            lr_schedule,
            b1=nadamw.beta_1,
            b2=nadamw.beta_2,
            eps=nadamw.epsilon,
            weight_decay=nadamw.weight_decay,
            mask=partial(make_weights_mask, nadamw.decay_selector),
        )
    elif config.HasField("nadam"):
        nadam = config.nadam
        tx = optax.nadam(
            lr_schedule,
            b1=nadam.beta_1,
            b2=nadam.beta_2,
            eps=nadam.epsilon,
        )
    elif config.HasField("sgd"):
        sgd = config.sgd
        tx = optax.sgd(
            lr_schedule,
            momentum=sgd.momentum if sgd.momentum else None,
            nesterov=sgd.nesterov,
        )
    else:
        raise ValueError(
            "Unsupported optimizer type: {}".format(
                config.WhichOneof("optimizer_type")
            )
        )
    if max_grad_norm is not None and max_grad_norm > 0:
        tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), tx)
    if config.HasField("freeze_selector"):
        tx = optax.masked(
            tx,
            mask=lambda p: jax.tree.map(
                lambda x: not x, make_weights_mask(config.freeze_selector, p)
            ),
        )
    return tx
